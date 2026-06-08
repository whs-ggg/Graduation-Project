# -*- coding: utf-8 -*-
"""
MDL4Microbiome — multimodal MLP from Lee & Rho, Sci Rep 12, 824 (2022).
https://doi.org/10.1038/s41598-022-04773-3

Per-modality dense towers (ReLU), last hidden layer as embedding; embeddings are
concatenated and passed through a two-layer fusion head. Original paper used three
modalities (taxonomy, genome-level RPKM, KEGG functional); here we map the first
modality in ``large_branch_modal_indices`` to the "functional" wider first layer
(500-100-50) and others to taxonomy-style (200-100-50), matching the paper's Fig. 2.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


InputsDim = Union[OrderedDict, Dict[str, Tuple[int, ...]]]


def _to_int_tuple(x: Sequence[int] | int | None, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    raise TypeError(f"Expected list/tuple of ints, got {type(x)}")


class _ModalityTower(nn.Module):
    """Dense-ReLU tower; output is the last hidden activation (embedding)."""

    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            h = int(h)
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            d = h
        self.net = nn.Sequential(*layers)
        self.embed_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MDL4Microbiome(nn.Module):
    def __init__(
        self,
        inputs_dim: InputsDim,
        small_hidden: Sequence[int] | None = None,
        large_hidden: Sequence[int] | None = None,
        head_hidden: Sequence[int] | None = None,
        large_branch_modal_indices: Sequence[int] | None = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            inputs_dim: OrderedDict of {modality_key: (N, F)} shapes; keys like f1_input, ...
            small_hidden: taxonomy / genome-style tower (default 200, 100, 50).
            large_hidden: functional-style tower (default 500, 100, 50).
            head_hidden: fusion MLP after concat (default 50, 25).
            large_branch_modal_indices: 0-based indices into modality order for ``large_hidden``.
                Default ``(0,)`` → first CSV in feature list (e.g. ko) uses large tower.
        """
        super().__init__()
        self.modality_order: Tuple[str, ...] = tuple(inputs_dim.keys())
        sh = _to_int_tuple(small_hidden, (200, 100, 50))
        lh = _to_int_tuple(large_hidden, (500, 100, 50))
        hh = _to_int_tuple(head_hidden, (50, 25))
        if large_branch_modal_indices is None:
            large_idx = frozenset({0})
        else:
            large_idx = frozenset(int(i) for i in large_branch_modal_indices)

        towers: Dict[str, _ModalityTower] = {}
        for i, name in enumerate(self.modality_order):
            shape = inputs_dim[name]
            in_dim = int(shape[1]) if len(shape) > 1 else int(shape[0])
            dims = lh if i in large_idx else sh
            towers[name] = _ModalityTower(in_dim, dims, dropout)
        self.towers = nn.ModuleDict(towers)

        emb_dim = sum(self.towers[k].embed_dim for k in self.modality_order)
        head_layers: List[nn.Module] = []
        d = emb_dim
        for h in hh:
            h = int(h)
            head_layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                head_layers.append(nn.Dropout(p=float(dropout)))
            d = h
        head_layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        for name, feat in list(features.items()):
            if isinstance(feat, np.ndarray):
                features[name] = torch.from_numpy(feat).to(device=device, dtype=torch.float32)

        parts = []
        for name in self.modality_order:
            parts.append(self.towers[name](features[name]))
        z = torch.cat(parts, dim=1)
        return self.head(z)
