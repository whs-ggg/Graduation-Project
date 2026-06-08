# -*- coding: utf-8 -*-
"""
Multimodal Variational Information Bottleneck (MVIB), adapted from:
Grazioli et al., PLOS Comput Biol 2022, https://doi.org/10.1371/journal.pcbi.1010050

Per-modality Gaussian encoders; joint approximate posterior via Product-of-Experts
with spherical Gaussian prior N(0, I); linear decoder for binary logits.
Continuous omics tables use the paper's "abundance" MLP when dim is moderate;
high-dimensional modalities use the "marker" MLP (dim/2 -> 1024 -> 1024).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def _kl_diag_gaussian_to_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ), summed over latent dims, per sample."""
    logvar = logvar.clamp(min=-10.0, max=6.0)
    var = torch.exp(logvar)
    return 0.5 * torch.sum(var + mu.pow(2) - 1.0 - logvar, dim=1)


def _gaussian_poe_diag(
    mus: List[torch.Tensor],
    logvars: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MVAE-style PoE: q(z|x) ∝ ∏_m q_m(z|x_m) / p(z)^(M-1) with p(z)=N(0,I), all diagonal.
    Returns fused (mu, logvar) with logvar = log(sigma^2).
    """
    if len(mus) == 1:
        return mus[0], logvars[0]

    m = len(mus)
    prec_sum = torch.zeros_like(mus[0])
    weighted_mean = torch.zeros_like(mus[0])

    for mu, lv in zip(mus, logvars):
        var = torch.exp(lv).clamp(min=1e-8, max=1e8)
        prec = 1.0 / var
        prec_sum = prec_sum + prec
        weighted_mean = weighted_mean + mu * prec

    # 专家冲突时 fused precision 可接近 0 → 方差爆炸 → KL 与 skorch 记录的 valid_loss 达 1e10+。
    # 对 fused precision 设下界，使融合后后验方差有上限（常见 MVAE 实现中的稳定化）。
    prec_fused = (prec_sum - float(m - 1)).clamp(min=0.5)
    var_f = 1.0 / prec_fused
    mu_f = weighted_mean / prec_fused
    logvar_f = torch.log(var_f.clamp(min=1e-6, max=50.0))
    return mu_f, logvar_f


class _AbundanceEncoder(nn.Module):
    """Paper: input_dim -> input_dim/2 -> input_dim/2 -> mu, logvar (K each)."""

    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        h = max(in_dim // 2, 1)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(h, latent_dim)
        self.fc_logvar = nn.Linear(h, latent_dim)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _MarkerEncoder(nn.Module):
    """Paper (strain markers): input_dim/2 -> 1024 -> 1024 -> mu, logvar."""

    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        h0 = max(in_dim // 2, 1)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h0),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h0, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MVIB(nn.Module):
    def __init__(
        self,
        inputs_dim: OrderedDict | Dict[str, Tuple[int, ...]],
        latent_dim: int = 256,
        beta: float = 1e-5,
        dropout: float = 0.4,
        marker_encoder_threshold: int = 2048,
    ):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.dropout = float(dropout)
        self.marker_encoder_threshold = int(marker_encoder_threshold)

        self.modality_order: Tuple[str, ...] = tuple(inputs_dim.keys())
        encoders: Dict[str, nn.Module] = {}
        for name in self.modality_order:
            shape = inputs_dim[name]
            in_dim = int(shape[1]) if len(shape) > 1 else int(shape[0])
            if in_dim > self.marker_encoder_threshold:
                encoders[name] = _MarkerEncoder(in_dim, self.latent_dim, self.dropout)
            else:
                encoders[name] = _AbundanceEncoder(in_dim, self.latent_dim, self.dropout)
        self.encoders = nn.ModuleDict(encoders)
        self.decoder = nn.Linear(self.latent_dim, 1)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.01)
        nn.init.zeros_(self.decoder.bias)
        self.last_kl: torch.Tensor | None = None

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        for name, feat in list(features.items()):
            if isinstance(feat, np.ndarray):
                features[name] = torch.from_numpy(feat).to(device=device, dtype=torch.float32)

        mus, logvars = [], []
        for name in self.modality_order:
            x = features[name]
            mu, lv = self.encoders[name](x)
            lv = lv.clamp(min=-10.0, max=10.0)
            mus.append(mu)
            logvars.append(lv)

        mu_z, logvar_z = _gaussian_poe_diag(mus, logvars)
        logvar_z = logvar_z.clamp(min=-10.0, max=6.0)
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        z = mu_z + eps * std

        kl = _kl_diag_gaussian_to_std_normal(mu_z, logvar_z)
        self.last_kl = kl.mean()

        return self.decoder(z)
