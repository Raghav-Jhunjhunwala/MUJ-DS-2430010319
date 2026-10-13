"""TabM-mini (BatchEnsemble-style parameter-efficient ensemble) with periodic embeddings."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicEmbedding(nn.Module):
    """x_j -> [sin(2*pi*c*x_j), cos(2*pi*c*x_j)] -> per-feature linear -> ReLU
    (Gorishniy et al., NeurIPS 2022)."""
    def __init__(self, n_features, n_freq=8, d_embed=8, sigma=0.1):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        bound = 1.0 / np.sqrt(2 * n_freq)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_freq, d_embed))
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(n_features, d_embed))

    def forward(self, x):
        z = 2 * np.pi * self.freq[None] * x[..., None]
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        out = torch.einsum("bfn,fnd->bfd", z, self.weight) + self.bias
        return F.relu(out).flatten(1)


class TabMMini(nn.Module):
    """k implicit MLPs sharing weights via random +-1 input scalings (BatchEnsemble),
    averaged at inference.  k=1 + use_embedding=False gives a plain MLP (ablation)."""
    def __init__(self, n_features, k=8, use_embedding=True, d_embed=8, n_freq=8,
                 sigma=0.1, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embed = PeriodicEmbedding(n_features, n_freq, d_embed, sigma)
            d_in = n_features * d_embed
        else:
            d_in = n_features
        self.r = nn.Parameter(torch.sign(torch.randn(k, d_in)))
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head_w = nn.Parameter(torch.randn(k, hidden) / np.sqrt(hidden))
        self.head_b = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        h = self.embed(x) if self.use_embedding else x
        h = h[:, None, :] * self.r[None]          # (B, k, d_in)
        h = self.backbone(h)                       # (B, k, hidden)
        return (h * self.head_w[None]).sum(-1) + self.head_b   # (B, k) logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
