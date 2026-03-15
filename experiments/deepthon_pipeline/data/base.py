import numpy as np
import gzip
import requests
import kagglehub
from pathlib import Path
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Tuple
from kagglehub import KaggleDatasetAdapter
from sklearn.datasets import load_breast_cancer
from deepthon.utils.split import train_test_split

from ..utils.logging import get_logger
from .registry import register_dataset

logger = get_logger(__name__)

# --- Structured Data Containers ---

@dataclass
class DataSplit:
    """Container for a single split (e.g. Train, Val, or Test)."""
    x: np.ndarray
    y: np.ndarray

    @property
    def shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.x.shape, self.y.shape

@dataclass
class DataModule:
    """Container for the entire dataset lifecycle."""
    train: DataSplit
    val: DataSplit
    test: Optional[DataSplit] = None

# --- Base Abstract Loader ---

# data/base.py

class BaseLoader(ABC):
    def __init__(self, ds_cfg) -> None:
        self.ds_cfg = ds_cfg
        self.processed_dir = Path(ds_cfg.paths.processed_dir)
        self.marker_path = self.processed_dir / ds_cfg.paths.marker_file

    def get_data(self) -> DataModule:
        if not self.marker_path.exists():
            logger.info(f"Processing raw data for {self.__class__.__name__}...")
            data_module = self.process()
            # Save all 3 potential splits
            assert isinstance(data_module, DataModule)
            self._save_to_disk(data_module)
            self.marker_path.touch()
            return data_module
        
        logger.info(f"Loading cached dataset from {self.processed_dir}")
        return self._load_from_disk()

    def _save_to_disk(self, dm: DataModule) -> None:
        """Saves all available splits using a clear naming convention."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to save a split if it exists
        def save_split(split, prefix):
            if split:
                np.save(self.processed_dir / f"{prefix}_x.npy", split.x)
                np.save(self.processed_dir / f"{prefix}_y.npy", split.y)

        save_split(dm.train, "train")
        save_split(dm.val, "val")
        save_split(dm.test, "test")

    def _load_from_disk(self) -> DataModule:
        """Loads splits from disk, reconstructs DataModule with optional test set."""
        
        # Standard Train/Val
        dm = DataModule(
            train=DataSplit(
                x=np.load(self.processed_dir / "train_x.npy"),
                y=np.load(self.processed_dir / "train_y.npy")
            ),
            val=DataSplit(
                x=np.load(self.processed_dir / "val_x.npy"),
                y=np.load(self.processed_dir / "val_y.npy")
            )
        )

        # Check if optional Test split exists on disk
        test_x_path = self.processed_dir / "test_x.npy"
        if test_x_path.exists():
            dm.test = DataSplit(
                x=np.load(test_x_path),
                y=np.load(self.processed_dir / "test_y.npy")
            )
        
        return dm
    def process(self) -> DataModule|None:
        pass

    def download_file(self, url: str, destination: Path) -> None:
        """Utility to download raw files."""
        if destination.exists():
            logger.info(f"File exists: {destination}. Skipping.")
            return
        
        try:
            logger.info(f"Downloading: {url}")
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with open(destination, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    def _prepare_splits(self, x: np.ndarray, y: np.ndarray) -> DataModule:
        """
        Logic to split data into Train, Val, and optionally Test based on config.
        Expected config structure:
        split:
            train: 0.8
            val: 0.1
            test: 0.1
            stratify: true
        """
        s_cfg = self.ds_cfg.get("split", {})
        train_r = s_cfg.get("train", 0.8)
        val_r = s_cfg.get("val", 0.2)
        test_r = s_cfg.get("test", 0.0)
        should_stratify = s_cfg.get("stratify", False)
        
        # 1. Split off the Test set first (if exists)
        if test_r > 0:
            # The test_size for the first split is just the test_r
            x_tv, x_test, y_tv, y_test = train_test_split(
                x, y, 
                test_size=test_r, 
                stratify=y if should_stratify else None
            )
            # 2. Split remaining (Train+Val) into Train and Val
            # We need to adjust the val ratio relative to the remaining data
            relative_val_size = val_r / (train_r + val_r)
            x_train, x_val, y_train, y_val = train_test_split(
                x_tv, y_tv, 
                test_size=relative_val_size, 
                stratify=y_tv if should_stratify else None
            )
            
            return DataModule(
                train=DataSplit(x_train, y_train),
                val=DataSplit(x_val, y_val),
                test=DataSplit(x_test, y_test)
            )
        
        else:
            # Only 2 splits: Train and Val
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, 
                test_size=val_r, 
                stratify=y if should_stratify else None
            )
            return DataModule(
                train=DataSplit(x_train, y_train),
                val=DataSplit(x_val, y_val)
            )

# --- Concrete Implementations ---
@register_dataset("mnist")
class MNISTLoader(BaseLoader):
    def process(self) -> DataModule:
        # 1. Define Raw Paths
        raw_dir = Path(self.ds_cfg.paths.raw_dir)
        paths = {
            "tr_x": raw_dir / "train-images", "tr_y": raw_dir / "train-labels",
            "ts_x": raw_dir / "t10k-images",  "ts_y": raw_dir / "t10k-labels",
        }
        
        # 2. Download
        self.download_file(self.ds_cfg.urls.train_images, paths["tr_x"])
        self.download_file(self.ds_cfg.urls.train_labels, paths["tr_y"])
        self.download_file(self.ds_cfg.urls.test_images, paths["ts_x"])
        self.download_file(self.ds_cfg.urls.test_labels, paths["ts_y"])

        # 3. Transform Raw Bytes
        full_train_x = transform_images(paths["tr_x"].read_bytes())
        full_train_y = transform_labels(paths["tr_y"].read_bytes())
        
        # 4. Use internal splitter to create Train/Val from the official training set
        # This respects your 'split' configuration in the YAML
        dm = self._prepare_splits(full_train_x, full_train_y)

        # 5. Attach the official MNIST Test set to the .test attribute
        dm.test = DataSplit(
            x=transform_images(paths["ts_x"].read_bytes()),
            y=transform_labels(paths["ts_y"].read_bytes())
        )
        
        return dm
    

@register_dataset("cancer")
class BreastCancerLoader(BaseLoader):
    def process(self) -> DataModule:
        X, y = load_breast_cancer(return_X_y=True)
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        
        return self._prepare_splits(X,y)

@register_dataset("turbines")
class TurbinesLoader(BaseLoader):
    def process(self) -> DataModule:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "ishank2005/wind-turbines-data-csv",
            "Data.csv",
        )
        X, y = df.drop(columns="PE").values, df["PE"].values.reshape(-1, 1)
        return self._prepare_splits(X,y)

# --- Pure Functional Helpers ---

def transform_images(raw_bytes: bytes) -> np.ndarray:
    decompressed = gzip.decompress(raw_bytes)
    data = np.frombuffer(decompressed, dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

def transform_labels(raw_bytes: bytes) -> np.ndarray:
    decompressed = gzip.decompress(raw_bytes)
    data = np.frombuffer(decompressed, dtype=np.uint8, offset=8)
    return one_hot_numpy(data, 10)

def one_hot_numpy(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels.reshape(-1)]