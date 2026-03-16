"""
Centralized logging utilities for the pipeline.

This module exposes a configurable logger factory to ensure consistent
logging behavior across pipeline components (data, models, training,
evaluation, orchestration, CLI, UI, etc.).
"""
import logging
from pathlib import Path
from typing import Optional
from ..config.loader import load_config

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)

BASE_LOGGER_NAME = "deepthon"

def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger prefixed with the base project name.
    Usage: get_logger(__name__) -> deepthon.experiments.data.loader
    """
    # Ensure the name is always deepthon.something
    if module_name and not module_name.startswith(BASE_LOGGER_NAME):
        full_name = f"{BASE_LOGGER_NAME}.{module_name}"
    else:
        full_name = module_name or BASE_LOGGER_NAME
        
    return logging.getLogger(full_name)

def setup_logging(config_path: str | Path):
    """
    Configures the ROOT 'deepthon' logger. 
    Call this ONCE at the start of your CLI or App.
    """
    cfg = load_config(config_path)
    # Get the 'logging' section or default to empty dict
    log_cfg = cfg.get("logging", {}) 
    
    root_logger = logging.getLogger(BASE_LOGGER_NAME)
    root_logger.setLevel(log_cfg.get("level", "INFO").upper())
    
    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    # 1. Console Output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. File Output
    if log_cfg.get("to_file", False):
        log_dir = Path(log_cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use experiment name from config for the filename
        filename = f"{cfg.get('experiment', 'deepthon_run')}.log"
        file_handler = logging.FileHandler(log_dir / filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.info(f"Logging initialized at level {root_logger.level}")