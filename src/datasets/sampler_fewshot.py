# -*- coding: utf-8 -*-
"""Few-shot episodic sampler."""

from __future__ import annotations

from typing import Dict, Iterator, List
import torch
from torch.utils.data import Sampler
import numpy as np


class FewShotSampler(Sampler):
    """
    Episodic sampler for few-shot learning.

    Each episode samples n_way classes, k_shot support samples, and n_query query samples per class.
    """

    def __init__(
        self,
        dataset,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        episodes_per_epoch: int = 200,
        shuffle: bool = True,
        num_classes: int = 10,
    ) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.shuffle = shuffle
        self.num_classes = num_classes

        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.episodes_per_epoch):
            yield self._sample_episode()

    def _sample_episode(self) -> List[int]:
        available = list(self.class_to_indices.keys())
        if len(available) < self.n_way:
            selected = available * (self.n_way // len(available) + 1)
            selected = selected[:self.n_way]
        else:
            selected = np.random.choice(available, self.n_way, replace=False).tolist()

        episode = []
        for cls_idx in selected:
            indices = self.class_to_indices[cls_idx]
            need = self.k_shot + self.n_query
            if len(indices) >= need:
                sampled = np.random.choice(indices, need, replace=False)
            else:
                sampled = np.array(indices)
            episode.extend(sampled.tolist())

        if self.shuffle:
            np.random.shuffle(episode)
        return episode

    def __len__(self) -> int:
        return self.episodes_per_epoch
