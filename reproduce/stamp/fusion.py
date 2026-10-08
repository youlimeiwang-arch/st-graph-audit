"""M1 fusion.

identity        return gene. Sanity check that the wrap matches frozen STAIG.

residual        gene + scale * g * adapter(img). Gate bias starts at -4, so this
                arm is conservative and often a near no-op.

residual_open   gene + scale * adapter(img). No gate. img_out still zero-init, so
                epoch 0 matches STAIG; gradients to the image path are not
                squashed by sigmoid(-4).

late / stream   do not touch the 3000-d gene input. Image is added in the 64-d
                encoder space (see ImageStream). Phikon residual_open showed
                that gene-space residuals are noise on 151673.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GatedFusion(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        img_dim: int,
        hidden: int = 64,
        mode: str = "residual",
        res_scale: float = 0.05,
    ):
        super().__init__()
        self.mode = mode
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale)))
        if mode == "identity":
            return
        self.img_adapter = nn.Sequential(
            nn.Linear(img_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.img_out = nn.Linear(hidden, gene_dim)
        self.gate = None
        if mode != "residual_open":
            self.gate = nn.Sequential(
                nn.Linear(gene_dim + hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, gene_dim),
                nn.Sigmoid(),
            )
        nn.init.xavier_uniform_(self.img_adapter[0].weight)
        nn.init.zeros_(self.img_adapter[0].bias)
        nn.init.xavier_uniform_(self.img_adapter[2].weight)
        nn.init.zeros_(self.img_adapter[2].bias)
        nn.init.zeros_(self.img_out.weight)
        nn.init.zeros_(self.img_out.bias)
        if self.gate is not None:
            nn.init.constant_(self.gate[-2].bias, -4.0)

    def forward(self, gene: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return gene
        h = self.img_adapter(img)
        h_img = self.img_out(h)
        if self.mode == "residual_open":
            return gene + self.res_scale * h_img
        g = self.gate(torch.cat([gene.detach(), h], dim=-1))
        residual = g * h_img
        if self.mode == "residual":
            return gene + self.res_scale * residual
        return g * gene + (1.0 - g) * residual


class ImageStream(nn.Module):
    """Map frozen FM features into STAIG's 64-d hidden space.

    late     MLP only, added after the full backbone (encoder + projection).
    stream   MLP → 1-layer GCN on the same edges, added to encoder output,
             then the STAIG projection. Histology gets message passing
             without rewriting gene node features.
    Last linear is zero-init, so epoch 0 matches STAIG embeddings.
    """

    def __init__(
        self,
        img_dim: int,
        hid_dim: int,
        mode: str = "stream",
        res_scale: float = 0.1,
    ):
        super().__init__()
        if mode not in ("late", "stream"):
            raise ValueError(f"ImageStream mode must be late|stream, got {mode}")
        self.mode = mode
        self.proj = nn.Sequential(nn.Linear(img_dim, hid_dim), nn.GELU())
        self.conv = GCNConv(hid_dim, hid_dim) if mode == "stream" else None
        self.out = nn.Linear(hid_dim, hid_dim)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale)))
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, img: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.proj(img)
        if self.conv is not None:
            x = F.elu(self.conv(x, edge_index))
        return self.out(x)
