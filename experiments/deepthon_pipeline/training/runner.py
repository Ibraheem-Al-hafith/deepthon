from pathlib import Path
import json
from typing import Optional

from ..data.loader import build_dataset
from ..data.base import DataModule  # Import the types for hinting
from ..models.builder import build_model_from_config
from ..utils.logging import get_logger
from ..models.optimizer_builder import build_optimizer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


from deepthon.pipeline.trainer import Trainer
from deepthon.nn.losses import CrossEntropy, MAE, BCE

logger = get_logger(__name__)

class ExperimentRunner:
    """
    Orchestrates full training lifecycle using structured DataModules.
    """

    def __init__(self, cfg, dataset_name: str, model_name: str):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.model_name = model_name
        
        # Hierarchical Path: runs/exp_name/dataset_model
        exp_name = cfg.get("experiment", "run")
        self.exp_dir = Path('runs') / exp_name / f"{dataset_name}_{model_name}"

        logger.info(f"Runner Initialized for {dataset_name} + {model_name}")
        logger.info(f"Artifacts will be saved to: {self.exp_dir}")

    def build_data(self):
        """Builds dataset and stores it in a structured DataModule."""
        # Dynamically fetch the dataset cofig based on name
        dataset_cfg = self.cfg.datasets.__getattr__(self.dataset_name)
        self.input_dim = dataset_cfg.input_dim
        self.output_dim = dataset_cfg.output_dim

        self.dm = build_dataset(dataset_cfg)
        logger.info(f"Dataset {self.dataset_name} ready with input {self.input_dim},{self.output_dim}")

    def build_model(self):
        model_cfg = self.cfg.model.__getattr__(self.model_name)
        self.model = build_model_from_config(
            self.cfg,
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            model_name=self.model_name
            )

    def build_optimizer(self):
        logger.info("Building optimizer...")
        self.optimizer = build_optimizer(self.cfg.training.optimizer)

    def build_trainer(self):
        logger.info("Initializing Trainer...")
        d_train_cfg = self.cfg.datasets.__getattr__(self.dataset_name).train_config # dataset training config
        t_cfg = self.cfg.training # global config

        # Logic map string to loss
        loss_map= {
            "CCE":CrossEntropy(from_logits=True),
            "BCE": BCE(from_logits=True),
            "MAE": MAE()
        }

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=loss_map.get(d_train_cfg.loss_fn, MAE()),
            batch_size=t_cfg.batch_size,
            early_stopping=t_cfg.get("early_stopping", False),
            patience=t_cfg.get("patience", 5),
            metric_fn=d_train_cfg.get("metric", None),
            min_delta=d_train_cfg.get("min_delata", 1e-4),
            checkpoint_dir=self.exp_dir,
            save_every=1
        )

    def test(self, checkpoint_path: Optional[Path] = None):
        """
        Evaluate the model on the test dataset.
        If checkpoint_path is provided, it loads the model weights first.
        """
        logger.info("Starting evaluation on test set...")

        # 1. Handle standalone evaluation (optional loading)
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded weights from {checkpoint_path}")

        # 2. Check if test data exists
        if not self.dm or not self.dm.test:
            logger.warning("No test split found in DataModule. Skipping test phase.")
            return

        # 3. Perform evaluation
        # Assuming your Trainer has an evaluate method, 
        # or we use the model directly:
        test_results,output = self.trainer.validate(
            X_val=self.dm.test.x,
            y_val=self.dm.test.y
        )
        msg = f"Test Results: {test_results} "
        if self.trainer.metric is not None:
            msg+=f"| {self.trainer.metric_fn}: {self.trainer.metric(self.dm.test.y, output)}"
        # Calculate metric
        metric_fn = self.trainer.metric
        score = metric_fn(self.dm.test.y, output) if metric_fn else 0.0
        
        # New: Plotting
        self._plot_task_results(self.dm.test.y, output, score, stage="test")
        
        results = {"loss": float(test_results), "metric": float(score)}
        self._save_test_artifacts(results)
        return results
    def _save_test_artifacts(self, results):
        """Save test metrics to a separate file."""
        path = self.exp_dir / "test_results.json"
        path.write_text(json.dumps(results, indent=2))
        logger.info(f"Test artifacts saved to {path}")

    def run(self, resume: bool = False):
        """The full 'Train + Test' pipeline."""
        logger.info("Starting experiment...")
        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.build_trainer()

        # --- RESUME LOGIC ---
        if resume:
            ckpt_path = self.exp_dir / "checkpoint.pkl"
            if ckpt_path.exists():
                logger.info(f"Resuming training from {ckpt_path}...")
                self.trainer.load_checkpoint(ckpt_path)
                logger.info(f"Resuming from Epoch {self.trainer.start_epoch + 1}")
            else:
                logger.warning(f"No checkpoint found at {ckpt_path}. Starting from scratch.")

        assert isinstance(self.dm, DataModule)
        # Phase 1: Training
        self.trainer.train(
            X_train=self.dm.train.x,
            y_train=self.dm.train.y,
            X_val=self.dm.val.x,
            y_val=self.dm.val.y,
            epochs=self.cfg.training.epochs,
        )
        

        # Phase 2: Post-Training Save
        self.trainer.save_checkpoint(filename="last_model.pkl")
        logger.info("Experiment completed.")

        self._plot_loss()
        # Phase 3: Immediate Testing
        self.test()

    def _plot_loss(self):
        """Plots Training vs Validation Loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.trainer.train_losses, label='Train Loss')
        plt.plot(self.trainer.val_losses, label='Val Loss')
        plt.title(f"Loss Plot: {self.dataset_name} + {self.model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        save_path = self.exp_dir / "loss_plot.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Loss plot saved to {save_path}")

    def _plot_task_results(self, y_true, y_pred, metric_score, stage="test"):
        """Generates task-specific plots (CM or Scatter)."""
        plt.figure(figsize=(8, 6))
        metric_name = self.cfg.datasets.__getattr__(self.dataset_name).train_config.get("metric", "Score")
        title = f"{self.dataset_name} ({self.model_name}) | {metric_name}: {metric_score:.4f}"

        # 1. Classification Plot (Confusion Matrix)
        if self.dataset_name in ["mnist", "cancer"]:
            # Convert one-hot to labels if necessary
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
            elif y_pred.ndim > 1 and y_pred.shape[1] == 1: # Binary BCE case
                y_pred = (y_pred > 0.5).astype(int)

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix\n{title}")

        # 2. Regression Plot (Scatter)
        else:
            plt.scatter(y_true, y_pred, alpha=0.5)
            # Add identity line
            limit = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(limit, limit, 'r--', label='Ideal')
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            plt.title(f"Regression Scatter Plot\n{title}")
            plt.legend()

        save_path = self.exp_dir / f"{stage}_visualization.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Task visualization saved to {save_path}")