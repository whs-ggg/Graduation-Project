# -*- coding: utf-8 -*-
"""简单全连接基线：与 FT-Concat 一样接收按模态拼接后的二维数值矩阵。"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn


HiddenSpec = Union[int, Sequence[int], None]


class TabularMLP(nn.Module):
    """
    多模态特征在数据管道中已沿特征维拼接为 (N, in_dim)，本模块直接输出 (N, 1) logits，
    与仓库中 MTMFTransformer / FT 等二分类设置一致，配合 skorch 的 BCEWithLogitsLoss。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: HiddenSpec = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_list = _normalize_hidden_dims(hidden_dims)
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in [0, 1]")
        layers: List[nn.Module] = []
        d = int(in_dim)
        if not hidden_list:
            layers.append(nn.Linear(d, 1))
        else:
            for h in hidden_list:
                h = int(h)
                layers += [
                    nn.Linear(d, h),
                    nn.LayerNorm(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(dropout)),
                ]
                d = h
            layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_hidden_dims(hidden_dims: HiddenSpec) -> List[int]:
    if hidden_dims is None:
        return [512]
    if isinstance(hidden_dims, int):
        return [hidden_dims] if hidden_dims > 0 else []
    if isinstance(hidden_dims, (list, tuple)):
        return [int(h) for h in hidden_dims if int(h) > 0]
    raise TypeError("hidden_dims must be int, list/tuple of int, or None")
