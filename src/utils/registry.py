# -*- coding: utf-8 -*-
"""Registry pattern."""
from typing import Any, Callable, Dict, Optional

class Registry:
    def __init__(self, name):
        self.name = name
        self._r = {}
        self._f = {}

    def register(self, name=None, factory=None):
        def w(cls):
            k = name or cls.__name__.lower()
            self._r[k] = cls
            if factory: self._f[k] = factory
            return cls
        return w

    def get(self, name):
        if name not in self._r:
            raise KeyError(f"'{name}' not in {self.name} registry. Available: {list(self._r.keys())}")
        return self._r[name]

    def create(self, name, **kw):
        cls = self.get(name)
        return self._f[name](**kw) if name in self._f else cls(**kw)

    def list(self):
        return list(self._r.keys())

MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
EVALUATOR_REGISTRY = Registry("evaluator")

def register_model(name=None): return MODEL_REGISTRY.register(name)
def register_dataset(name=None): return DATASET_REGISTRY.register(name)
def register_loss(name=None): return LOSS_REGISTRY.register(name)
def register_evaluator(name=None): return EVALUATOR_REGISTRY.register(name)
