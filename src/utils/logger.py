# -*- coding: utf-8 -*-
"""Logging utilities."""
import logging, os, sys
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except:
    HAS_TB = False

def setup_logging(log_dir, level="INFO", name="glocal"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def get_logger(name="glocal"):
    return logging.getLogger(name)

class TensorBoardLogger:
    def __init__(self, log_dir, exp_name):
        self.exp_name = exp_name
        self.writer = SummaryWriter(os.path.join(log_dir, exp_name)) if HAS_TB else None
    def log_scalar(self, tag, value, step):
        if self.writer: self.writer.add_scalar(tag, value, step)
    def log_scalars(self, tag, values, step):
        if self.writer: self.writer.add_scalars(tag, values, step)
    def close(self):
        if self.writer: self.writer.close()
