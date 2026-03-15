from pathlib import Path
from ..config.loader import load_config
from ..training.runner import ExperimentRunner
from ..utils.logging import get_logger
logger = get_logger(__name__)


def cmd_train(config_path, dataset_name, model_name, resume: bool = False):
    cfg = load_config(config_path)
    assert isinstance(cfg.datasets,dict)
    assert isinstance(cfg.model,dict)
    datasets = list(cfg.datasets.keys()) if dataset_name == "all" else dataset_name if isinstance(dataset_name, list) else [dataset_name]
    models = list(cfg.model.keys()) if model_name =="all" else model_name if isinstance(model_name, list)  else [model_name]

    for d_name in datasets:
        for m_name in models:
            logger.info(f"\n{'='*20}\nTARGET: {d_name} | MODEL: {m_name}\n{'='*20}")
            try:
                runner = ExperimentRunner(cfg, dataset_name=d_name, model_name=m_name)
                runner.run(resume=resume)
            except Exception as e:
                logger.error(f"Failed training {d_name}_{m_name}: {e}")
                raise


def cmd_test(config_path: str,checkpoint_path:str, dataset_name:str, model_name: str):
    """
    Run evaluation only. 
    Usage: deepthon-cli test --config my_cfg.yaml --ckpt runs/exp_1/checkpoint.pkl
    """
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg, dataset_name=dataset_name, model_name=model_name)
    
    # 1. Only build what we need for inference
    runner.build_data()
    runner.build_model()
    runner.build_optimizer() # Usually needed to init Trainer state
    runner.build_trainer()

    # 2. Run the test with the specific checkpoint
    runner.test(checkpoint_path=Path(checkpoint_path))


def cmd_test_all(config_path: str):
    """
    Run evaluation only. 
    Usage: deepthon-cli test --config my_cfg.yaml --ckpt runs/exp_1/checkpoint.pkl
    """
    cfg = load_config(config_path)
    exp_name = cfg.get("experiment", "run")
    base_dir = Path("runs") / exp_name

    if not base_dir.exists():
        logger.error(f"Experiment direcroty {base_dir} not found.")
        return
    
    # Find all directories that contain a best_model.pkl
    for run_dir in base_dir.iterdir():
        if run_dir.is_dir():
            ckpt = run_dir / "last_model.pkl"
            if not ckpt.exists():
                continue
            d_name, m_name = run_dir.name.split("_")
            runner = ExperimentRunner(cfg, dataset_name=d_name, model_name=m_name)
            runner.build_data()
            runner.build_model()
            runner.build_optimizer()
            runner.build_trainer()
            runner.test(checkpoint_path=ckpt)

