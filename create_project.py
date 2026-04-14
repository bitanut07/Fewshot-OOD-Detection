#!/usr/bin/env python3
import os
R = "/Users/lap14568/Fewshot-OOD-Detection"

def mk(name, content):
    path = os.path.join(R, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {name}")

# ═══ 1. UTILS ════════════════════════════════════════════════════════════════════
mk("src/utils/__init__.py", """# Utility modules for GLOCAL FSL-OOD framework
from .config import load_config, merge_configs, Config
from .logger import get_logger, setup_logging
from .seed import set_seed
from .checkpoint import save_checkpoint, load_checkpoint
from .registry import Registry

__all__ = [
    "load_config", "merge_configs", "Config",
    "get_logger", "setup_logging",
    "set_seed", "save_checkpoint", "load_checkpoint",
    "Registry",
]
""")

# 2. config.py
mk("src/utils/config.py", '''# -*- coding: utf-8 -*-
"""Configuration management with YAML loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


class Config(dict):
    """Dict-like config object with dot-notation access."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise KeyError(f"Config key \\'{key}\\' not found")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        val = self
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                result[k] = v
        return result


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load a YAML config file with inheritance via includes field.
    
    Args:
        config_path: Path to the experiment YAML file.
        
    Returns:
        Merged Config object.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("includes", [])
    base_dir = config_path.parent
    base: Dict[str, Any] = {}
    default_path = base_dir.parent / "default.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
    for inc in includes:
        inc_path = base_dir / inc
        if not inc_path.exists():
            inc_path = base_dir.parent.parent / inc
        if not inc_path.exists():
            raise FileNotFoundError(f"Included config not found: {inc}")
        with open(inc_path, "r", encoding="utf-8") as f:
            base = _merge(base, yaml.safe_load(f) or {})
    base = _merge(base, cfg)
    return Config(base)
''')

# 3. logger.py
mk("src/utils/logger.py", '''# -*- coding: utf-8 -*-
"""Logging utilities."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def setup_logging(log_dir: str, level: str = "INFO", name: str = "glocal") -> logging.Logger:
    """Setup logging with file and console handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def get_logger(name: str = "glocal") -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.writer = (
            SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
            if HAS_TB else None
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int) -> None:
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
''')

# 4. seed.py
mk("src/utils/seed.py", '''# -*- coding: utf-8 -*-
"""Random seed utilities for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
''')

# 5. checkpoint.py
mk("src/utils/checkpoint.py", '''# -*- coding: utf-8 -*-
"""Checkpoint saving and loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
) -> str:
    """Save a model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    if is_best:
        torch.save(state, checkpoint_dir / "best.pt")
    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    load_finetuned_only: bool = False,
) -> Dict[str, Any]:
    """Load a model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device or torch.device("cpu")
    )
    if model is not None:
        if not load_finetuned_only:
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
''')

# 6. registry.py
mk("src/utils/registry.py", '''# -*- coding: utf-8 -*-
"""Registry pattern for modular component registration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """
    Generic registry for models, datasets, losses, evaluators, etc.
    Enables config-driven instantiation without hardcoded class names.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, type] = {}
        self._factory: Dict[str, Callable] = {}

    def register(
        self, name: Optional[str] = None, factory: Optional[Callable] = None
    ) -> Callable:
        """Register a class or factory function."""

        def wrapper(cls_or_fn: Any) -> Any:
            key = name or cls_or_fn.__name__.lower()
            self._registry[key] = cls_or_fn
            if factory is not None:
                self._factory[key] = factory
            return cls_or_fn

        return wrapper

    def get(self, name: str) -> type:
        """Get registered class by name."""
        if name not in self._registry:
            raise KeyError(f"\\'{name}\\' not found in {self.name} registry")
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class with kwargs."""
        cls = self.get(name)
        if name in self._factory:
            return self._factory[name](**kwargs)
        return cls(**kwargs)

    def list_registered(self) -> list:
        return list(self._registry.keys())


MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
EVALUATOR_REGISTRY = Registry("evaluator")


def register_model(name: Optional[str] = None) -> Callable:
    return MODEL_REGISTRY.register(name)


def register_dataset(name: Optional[str] = None) -> Callable:
    return DATASET_REGISTRY.register(name)


def register_loss(name: Optional[str] = None) -> Callable:
    return LOSS_REGISTRY.register(name)


def register_evaluator(name: Optional[str] = None) -> Callable:
    return EVALUATOR_REGISTRY.register(name)
''')

print("\n=== Utils done ===")