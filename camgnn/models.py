"""Model definitions for CaMGNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_max_pool, global_mean_pool


class BackdoorCausalModule(nn.Module):
    """Backdoor-adjusted classifier over learned clinical confounder strata."""

    def __init__(self, graph_dim: int, clin_hidden: int, num_strata: int = 8) -> None:
        super().__init__()
        if graph_dim < 2:
            raise ValueError("graph_dim must be at least 2.")

        self.num_strata = num_strata
        self.confounder_dict = nn.Parameter(torch.empty(num_strata, clin_hidden))
        nn.init.normal_(self.confounder_dict, std=0.1)

        attn_dim = max(graph_dim // 2, 1)
        self.q_proj = nn.Linear(graph_dim, attn_dim)
        self.k_proj = nn.Linear(clin_hidden, attn_dim)

    def forward(self, graph_embedding: torch.Tensor, clinical_embedding: torch.Tensor, cls_head: nn.Module) -> torch.Tensor:
        q = self.q_proj(graph_embedding)
        k = self.k_proj(self.confounder_dict)
        attention = F.softmax(torch.matmul(q, k.T) / (q.size(-1) ** 0.5), dim=1)

        logits = None
        batch_size = graph_embedding.size(0)
        for stratum_idx in range(self.num_strata):
            confounder = self.confounder_dict[stratum_idx].unsqueeze(0).expand(batch_size, -1)
            fused = torch.cat([graph_embedding, confounder, clinical_embedding], dim=1)
            stratum_logits = cls_head(fused)
            weighted_logits = attention[:, stratum_idx].unsqueeze(1) * stratum_logits
            logits = weighted_logits if logits is None else logits + weighted_logits

        if logits is None:
            raise RuntimeError("No confounder strata were evaluated.")
        return logits


class CaMGNN(nn.Module):
    """Causal multimodal graph neural network for AD classification."""

    def __init__(
        self,
        x_dim: int,
        clin_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.4,
        num_layers: int = 3,
        clin_hidden: int = 32,
        num_strata: int = 8,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Keep the original attribute name so checkpoints produced by the
        # monolithic research script remain directly compatible.
        self.embed = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )
            self.norms.append(LayerNorm(hidden_dim))

        self.clin_mlp = nn.Sequential(
            nn.Linear(clin_dim, clin_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        graph_dim = 2 * hidden_dim
        self.causal_module = BackdoorCausalModule(
            graph_dim=graph_dim,
            clin_hidden=clin_hidden,
            num_strata=num_strata,
        )

        self.cls_head = nn.Sequential(
            nn.Linear(graph_dim + clin_hidden + clin_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        clin: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = conv(x, edge_index)
            residual = norm(residual)
            residual = F.elu(residual)
            residual = F.dropout(residual, p=self.dropout, training=self.training)
            x = x + residual

        graph_embedding = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )

        if clin.dim() == 1:
            clin = clin.view(1, -1)
        elif clin.dim() > 2:
            clin = clin.view(clin.size(0), -1)

        clinical_embedding = self.clin_mlp(clin)
        return self.causal_module(graph_embedding, clinical_embedding, self.cls_head)


# Backward-compatible alias for checkpoints or notebooks based on the old name.
GATv2MultiTask_MRIClin = CaMGNN


def load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
):
    """Load original or early refactor checkpoints into CaMGNN.

    Early refactor checkpoints used `node_embed.*`; the original script and
    current public model use `embed.*`.
    """

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("node_embed."):
            key = f"embed.{key[len('node_embed.'):]}"
        remapped[key] = value
    return model.load_state_dict(remapped, strict=strict)
