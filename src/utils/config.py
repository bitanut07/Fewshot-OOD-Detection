# -*- coding: utf-8 -*-
"""Configuration management with YAML loading and merging."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Union
import yaml

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise KeyError(f"Config key '{key}' not found")

    def __setattr__(self, key, value):
        self[key] = value

    def get_nested(self, *keys, default=None):
        val = self
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                result[k] = v
        return result

def _merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("includes", [])
    base_dir = config_path.parent
    base = {}
    dp = base_dir.parent / "default.yaml"
    if dp.exists():
        with open(dp) as f:
            base = yaml.safe_load(f) or {}
    for inc in includes:
        # Resolve include path relative to:
        #   1) the current config directory (base_dir)
        #   2) the configs/ directory (base_dir.parent)
        #   3) the project root (base_dir.parent.parent)
        ip = base_dir / inc
        if not ip.exists():
            ip = base_dir.parent / inc
        if not ip.exists():
            ip = base_dir.parent.parent / inc
        if not ip.exists():
            raise FileNotFoundError(f"Config not found: {inc}")
        with open(ip) as f:
            base = _merge(base, yaml.safe_load(f) or {})
    base = _merge(base, cfg)
    return Config(base)
