"""STAMP trainer: M1 fusion in front of the STAIG MVmodel. Optional M2 proto CE."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from staig.net import drop_feature, multiple_dropout_average, training_graph_views, training_feature_views
from staig.staig import STAIG, adj_to_edge_index, convert_edge_probabilities, generate_pseudo_labels

from .fusion import GatedFusion, ImageStream


def proto_ce_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Pull each spot toward the stop-grad mean of its prototype in embedding space."""
    labels = labels.long()
    z = F.normalize(z, dim=1)
    k = int(labels.max().item()) + 1
    proto = torch.zeros(k, z.size(1), device=z.device, dtype=z.dtype)
    counts = torch.zeros(k, device=z.device, dtype=z.dtype)
    proto.index_add_(0, labels, z.detach())
    counts.index_add_(0, labels, torch.ones(labels.size(0), device=z.device, dtype=z.dtype))
    proto = proto / counts.clamp_min(1.0).unsqueeze(1)
    proto = F.normalize(proto, dim=1)
    return F.cross_entropy(z @ proto.t() / tau, labels)


class STAMPModel(nn.Module):
    """STAIG MVmodel plus optional gene-space fusion and/or 64-d image stream."""

    def __init__(self, fusion: GatedFusion, backbone: nn.Module, stream: ImageStream | None = None):
        super().__init__()
        self.fusion = fusion
        self.backbone = backbone
        self.stream = stream

    def forward(self, gene, img, edge_index):
        x = self.fusion(gene, img)
        if self.stream is None:
            return self.backbone(x, edge_index)
        delta = self.stream.res_scale * self.stream(img, edge_index)
        if self.stream.mode == "late":
            return self.backbone(x, edge_index) + delta
        h = self.backbone.encoder(x, edge_index) + delta
        return self.backbone.projection(h)

    def contrastive_loss_bias(self, z1, z2, adj, pseudo_labels, mean: bool = True):
        return self.backbone.contrastive_loss_bias(z1, z2, adj, pseudo_labels, mean=mean)


class STAMP(STAIG):
    def __init__(self, args, config, single: bool = False, refine: bool = True):
        super().__init__(args, config, single=single, refine=refine)
        self.ckpt_dir = config.get("ckpt_dir", "stamp/checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{args.slide}.pt")
        self.img_key = config.get("img_key", "img_emb")
        img_dim = int(config.get("img_dim", 16))
        fusion_mode = config.get("fusion_mode", "residual")
        self.fusion_warmup = int(config.get("fusion_warmup", 0))
        # identity must be an unmodified STAIG: wrapping + a second Adam group
        # drifted 151673 from 0.687 to 0.638.
        self.plain = fusion_mode == "identity"
        if self.plain:
            return
        early_mode = fusion_mode if fusion_mode in ("residual", "residual_open", "convex") else "identity"
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        fusion = GatedFusion(
            self.num_gene,
            img_dim,
            mode=early_mode,
            res_scale=float(config.get("res_scale", 0.05)),
        ).to(self.device).double()
        stream = None
        if fusion_mode in ("late", "stream"):
            stream = ImageStream(
                img_dim,
                int(self.num_hidden),
                mode=fusion_mode,
                res_scale=float(config.get("res_scale", 0.1)),
            ).to(self.device).double()
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        self.model = STAMPModel(fusion, self.model, stream).to(self.device).double()
        fusion_lr = float(config.get("fusion_lr", self.learning_rate * 0.1))
        extra = list(self.model.fusion.parameters())
        if self.model.stream is not None:
            extra += list(self.model.stream.parameters())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": self.learning_rate},
                {"params": extra, "lr": fusion_lr},
            ],
            weight_decay=self.weight_decay,
        )

    def train(self):
        if self.plain:
            proto_w = float(self.config.get("proto_loss_w", 0.0))
            look_w = float(self.config.get("lookalike_w", 0.0))
            if proto_w <= 0 and look_w <= 0:
                return super().train()
            return self._train_plain_extra(proto_w, look_w)
        if self.adata is None:
            raise ValueError("adata not load!")
        gene = torch.tensor(self.adata.obsm["feat"].copy(), device=self.device, dtype=torch.float64)
        img = torch.tensor(self.adata.obsm[self.img_key].copy(), device=self.device, dtype=torch.float64)
        graph_neigh = torch.tensor(self.adata.obsm["graph_neigh"].copy(), device=self.device, dtype=torch.float64)
        edge_probabilities = torch.tensor(
            self.adata.obsm["edge_probabilities"].copy(), device=self.device, dtype=torch.float64
        )
        proto_key = self.config.get("proto_key", "img_emb")
        proto_k = int(self.config.get("proto_k", self.config.get("k", 40)))
        use_debiased = proto_key not in ("none", "off", "")
        if use_debiased:
            proto_feat = self.adata.obsm[proto_key]
            pseudo_labels = generate_pseudo_labels(proto_feat, proto_k).to(self.device)
        else:
            pseudo_labels = None
        proto_loss_w = float(self.config.get("proto_loss_w", 0.0))
        proto_loss_labels = None
        if proto_loss_w > 0:
            loss_key = self.config.get("proto_loss_key", "phikon_emb")
            loss_k = int(self.config.get("proto_loss_k", 40))
            npy_state = np.random.get_state()
            py_state = __import__("random").getstate()
            proto_loss_labels = generate_pseudo_labels(self.adata.obsm[loss_key], loss_k).to(self.device)
            np.random.set_state(npy_state)
            __import__("random").setstate(py_state)
            proto_loss_tau = float(self.config.get("proto_loss_tau", 0.5))
        edge_index = adj_to_edge_index(graph_neigh)
        edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)
        stream_mode = self.model.stream.mode if self.model.stream is not None else "off"
        print(
            f"=== train STAMP  slide={self.args.slide} fusion={self.model.fusion.mode} "
            f"stream={stream_mode} gene={tuple(gene.shape)} img={tuple(img.shape)} proto={proto_key}:{proto_k} "
            f"warmup={self.fusion_warmup} proto_loss_w={proto_loss_w} ==="
        )
        for epoch in tqdm.tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
            freeze = epoch <= self.fusion_warmup
            heads = list(self.model.fusion.parameters())
            if self.model.stream is not None:
                heads += list(self.model.stream.parameters())
            for p in heads:
                p.requires_grad = not freeze
            self.model.train()
            self.optimizer.zero_grad()
            edge_index_1, edge_index_2 = training_graph_views(
                edge_index, edge_probs, self.config, force_undirected=True
            )
            x_1, x_2 = training_feature_views(gene, self.drop_feature_rate_1, self.drop_feature_rate_2, self.config)
            z1 = self.model(x_1, img, edge_index_1)
            z2 = self.model(x_2, img, edge_index_2)
            if epoch == 1 or epoch % 50 == 0:
                with torch.no_grad():
                    fused = self.model.fusion(gene, img)
                    delta = (fused - gene).abs().mean().item()
                    stream = self.model.stream
                    if stream is None:
                        extra = f"res_scale={float(self.model.fusion.res_scale):.4f}"
                    else:
                        mix = (stream.res_scale * stream(img, edge_index)).abs().mean().item()
                        extra = (
                            f"stream={stream.mode} |img-hid|={mix:.4e} "
                            f"res_scale={float(stream.res_scale):.4f}"
                        )
                    print(f"  epoch {epoch} fusion |x-gene|={delta:.4e} {extra}", flush=True)
            if use_debiased:
                loss = self.model.contrastive_loss_bias(z1, z2, graph_neigh, pseudo_labels)
            else:
                loss = self.model.backbone.contrastive_loss(z1, z2, graph_neigh)
            if proto_loss_w > 0 and proto_loss_labels is not None:
                loss = loss + proto_loss_w * 0.5 * (
                    proto_ce_loss(z1, proto_loss_labels, proto_loss_tau)
                    + proto_ce_loss(z2, proto_loss_labels, proto_loss_tau)
                )
            lookalike_w = float(self.config.get("lookalike_w", 0.0))
            if lookalike_w > 0:
                if not hasattr(self, "_lookalike_edge_index"):
                    from stamp.edges import _fm_matrix
                    from stamp.lookalike_contrastive import build_lookalike_edge_index

                    tau_fm = float(self.config.get("lookalike_tau_fm", 0.80))
                    tau_gene = float(self.config.get("lookalike_tau_gene", 0.20))
                    fm_key = self.config.get("lookalike_fm_key", "phikonv2_emb")
                    fm = _fm_matrix(self.adata, fm_key, pca_dim=16)
                    gene_np = _fm_matrix(self.adata, "feat", pca_dim=32)
                    self._lookalike_edge_index, look_stats = build_lookalike_edge_index(
                        self.adata.obsm["graph_neigh"],
                        fm,
                        gene_np,
                        tau_fm=tau_fm,
                        tau_gene=tau_gene,
                        spatial=self.adata.obsm.get("spatial"),
                        min_dist_mult=float(self.config.get("lookalike_min_dist_mult", 3.0)),
                        device=self.device,
                    )
                    self._lookalike_margin = float(self.config.get("lookalike_margin", 0.5))
                    print(f"  lookalike edges: {look_stats}", flush=True)
                from stamp.lookalike_contrastive import lookalike_repulsion_loss

                loss = loss + lookalike_w * 0.5 * (
                    lookalike_repulsion_loss(z1, self._lookalike_edge_index, margin=self._lookalike_margin)
                    + lookalike_repulsion_loss(z2, self._lookalike_edge_index, margin=self._lookalike_margin)
                )
            loss.backward()
            self.optimizer.step()
        torch.save(self.model.state_dict(), self.ckpt_path)

    def _train_plain_extra(self, proto_loss_w: float, lookalike_w: float = 0.0) -> None:
        """Identity STAIG loop + optional proto CE + optional lookalike repulsion."""
        if self.adata is None:
            raise ValueError("adata not load!")
        features_matrix = torch.tensor(self.adata.obsm["feat"].copy(), device=self.device, dtype=torch.float64)
        graph_neigh = torch.tensor(self.adata.obsm["graph_neigh"].copy(), device=self.device, dtype=torch.float64)
        edge_probabilities = torch.tensor(
            self.adata.obsm["edge_probabilities"].copy(), device=self.device, dtype=torch.float64
        )
        if "k" in self.config:
            pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"], self.config["k"]).to(self.device)
        else:
            pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"]).to(self.device)
        proto_labels = None
        tau = 0.5
        if proto_loss_w > 0:
            loss_key = self.config.get("proto_loss_key", "phikonv2_emb")
            loss_k = int(self.config.get("proto_loss_k", 20))
            tau = float(self.config.get("proto_loss_tau", 0.5))
            npy_state = np.random.get_state()
            proto_labels = generate_pseudo_labels(self.adata.obsm[loss_key], loss_k).to(self.device)
            np.random.set_state(npy_state)
        lookalike_edge_index = None
        lookalike_margin = float(self.config.get("lookalike_margin", 0.5))
        if lookalike_w > 0:
            from stamp.edges import _fm_matrix
            from stamp.lookalike_contrastive import build_lookalike_edge_index

            fm_key = self.config.get("lookalike_fm_key", "phikonv2_emb")
            fm = _fm_matrix(self.adata, fm_key, pca_dim=16)
            gene_np = _fm_matrix(self.adata, "feat", pca_dim=32)
            lookalike_edge_index, look_stats = build_lookalike_edge_index(
                self.adata.obsm["graph_neigh"],
                fm,
                gene_np,
                tau_fm=float(self.config.get("lookalike_tau_fm", 0.80)),
                tau_gene=float(self.config.get("lookalike_tau_gene", 0.20)),
                spatial=self.adata.obsm.get("spatial"),
                min_dist_mult=float(self.config.get("lookalike_min_dist_mult", 3.0)),
                device=self.device,
            )
            print(f"  lookalike edges: {look_stats}", flush=True)
        edge_index = adj_to_edge_index(graph_neigh)
        edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)
        print(
            f"=== train STAMP-plain slide={self.args.slide} proto_w={proto_loss_w} "
            f"lookalike_w={lookalike_w} ===",
            flush=True,
        )
        from stamp.lookalike_contrastive import lookalike_repulsion_loss

        for _epoch in tqdm.tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
            self.model.train()
            self.optimizer.zero_grad()
            edge_index_1, edge_index_2 = training_graph_views(
                edge_index, edge_probs, self.config, force_undirected=True
            )
            x_1, x_2 = training_feature_views(
                features_matrix, self.drop_feature_rate_1, self.drop_feature_rate_2, self.config
            )
            z1 = self.model(x_1, edge_index_1)
            z2 = self.model(x_2, edge_index_2)
            if hasattr(self.model, "contrastive_loss_bias"):
                loss = self.model.contrastive_loss_bias(z1, z2, graph_neigh, pseudo_labels)
            else:
                loss = self.model.contrastive_loss(z1, z2, graph_neigh)
            if proto_loss_w > 0 and proto_labels is not None:
                loss = loss + proto_loss_w * 0.5 * (
                    proto_ce_loss(z1, proto_labels, tau) + proto_ce_loss(z2, proto_labels, tau)
                )
            if lookalike_w > 0 and lookalike_edge_index is not None:
                loss = loss + lookalike_w * 0.5 * (
                    lookalike_repulsion_loss(z1, lookalike_edge_index, margin=lookalike_margin)
                    + lookalike_repulsion_loss(z2, lookalike_edge_index, margin=lookalike_margin)
                )
            loss.backward()
            self.optimizer.step()
        torch.save(self.model.state_dict(), "model.pt")

    def _train_plain_proto(self, proto_loss_w: float) -> None:
        return self._train_plain_extra(proto_loss_w, 0.0)

    def eva(self):
        if self.plain:
            return super().eva()
        print("=== load ===")
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval()
        gene = torch.tensor(self.adata.obsm["feat"].copy(), device=self.device, dtype=torch.float64)
        img = torch.tensor(self.adata.obsm[self.img_key].copy(), device=self.device, dtype=torch.float64)
        graph_neigh = torch.tensor(self.adata.obsm["graph_neigh"].copy(), device=self.device, dtype=torch.float64)
        edge_index = adj_to_edge_index(graph_neigh)
        with torch.no_grad():
            self.adata.obsm["fused_feat"] = self.model.fusion(gene, img).detach().cpu().numpy()
            self.adata.obsm["emb"] = self.model(gene, img, edge_index).detach().cpu().numpy()
        print("embedding generated, go clustering")
