"""
Centralized logging utilities for the pipeline.

This module exposes a configurable logger factory to ensure consistent
logging behavior across pipeline components (data, models, training,
evaluation, orchestration, CLI, UI, etc.).
"""

from __future__ import annotations
import logging
from logging import Logger
from pathlib import Path
from ..config.loader import load_config

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s |"
    "%(funcName)s:%(lineno)d | %(message)s"
)

_LOGGERS: dict[str, Logger] = {}    # cache to avoid duplicate handlers
# src/deepthon_pipeline/utils/logging.py

# ... (imports and _DEFAULT_FORMAT remain same) ...

# 1. Use a consistent base name for the whole project
BASE_LOGGER_NAME = "src" 

def get_logger(name: str = BASE_LOGGER_NAME) -> Logger:
    """Returns a logger. If it's a child, it will propagate to 'src'."""
    logger = logging.getLogger(name)
    
    # IMPORTANT: Do not add handlers or set propagate=False here 
    # if it's a child logger. Just set the level.
    logger.setLevel(logging.INFO) 
    return logger

def logger_from_config(config_path: str|Path) -> Logger:
    cfg = load_config(config_path)
    logging_cfg = cfg.get("logging", {})
    
    # 2. We always configure the TOP LEVEL logger ("src")
    root_logger = logging.getLogger(BASE_LOGGER_NAME)
    root_logger.setLevel(logging_cfg.get("level", "INFO"))
    
    # Clear existing handlers to prevent duplicates if called twice
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File Handler
    if logging_cfg.get("to_file", False):
        log_dir = Path(logging_cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # We can still name the FILE after the experiment
        exp_name = cfg.get("experiment", "run")
        fh = logging.FileHandler(log_dir / f"{exp_name}.log")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger