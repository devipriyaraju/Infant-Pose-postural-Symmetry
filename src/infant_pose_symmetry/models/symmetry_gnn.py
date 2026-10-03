from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from infant_pose_symmetry.constants import FLIP_PERM, LIMB_JOINTS, MIRROR_EDGES, SKELETON
from infant_pose_symmetry.symmetry.geometry import canonicalize, geometric_features


def build_adjacency(mirror=True):
    adjacency = np.eye(17, dtype=bool)
    edges = SKELETON + (MIRROR_EDGES if mirror else [])
    for a, b in edges:
        adjacency[a, b] = True
        adjacency[b, a] = True
    return torch.tensor(adjacency)


class DenseGAT(nn.Module):
    def __init__(self, input_dim, output_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = output_dim // heads
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_source = nn.Parameter(
            torch.randn(heads, self.head_dim) * 0.1
        )
        self.attention_target = nn.Parameter(
            torch.randn(heads, self.head_dim) * 0.1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        batch, nodes, _ = x.shape
        projected = self.projection(x).view(
            batch,
            nodes,
            self.heads,
            self.head_dim,
        )

        score = (
            (projected * self.attention_target).sum(-1)[:, :, None, :]
            + (projected * self.attention_source).sum(-1)[:, None, :, :]
        )
        score = F.leaky_relu(score, 0.2)
        score = score.masked_fill(
            ~adjacency[None, :, :, None],
            float("-inf"),
        )
        attention = self.dropout(torch.softmax(score, dim=2))
        self.last_attention = attention.detach()

        return torch.einsum(
            "bijh,bjhd->bihd",
            attention,
            projected,
        ).reshape(batch, nodes, -1)


class DenseGCN(nn.Module):
    def __init__(self, input_dim, output_dim, **_):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency):
        matrix = adjacency.float()
        degree = matrix.sum(1).rsqrt()
        normalized = degree[:, None] * matrix * degree[None, :]
        return normalized @ self.projection(x)


class SymmetryGNN(nn.Module):
    def __init__(
        self,
        layer="gat",
        hidden=64,
        n_layers=3,
        use_geometry=False,
        dropout=0.1,
        mirror=True,
    ):
        super().__init__()
        self.register_buffer("adjacency", build_adjacency(mirror))
        self.joint_embedding = nn.Embedding(17, 16)
        self.input_projection = nn.Linear(3 + 16, hidden)

        layer_class = DenseGAT if layer == "gat" else DenseGCN
        self.layers = nn.ModuleList(
            [
                layer_class(hidden, hidden, heads=4, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )

        self.limb_embedding = nn.Embedding(4, 16)
        self.use_geometry = use_geometry
        self.register_buffer(
            "limb_index",
            torch.tensor(list(LIMB_JOINTS.values())),
        )

        readout_dim = hidden * 7 + 16 + (3 if use_geometry else 0)
        self.head = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

    def forward(self, x, geometry=None):
        batch = x.shape[0]
        joint_embedding = self.joint_embedding.weight[None].expand(
            batch,
            -1,
            -1,
        )
        hidden = self.input_projection(
            torch.cat([x, joint_embedding], dim=-1)
        )

        for layer, norm in zip(self.layers, self.norms):
            hidden = norm(
                hidden + F.elu(layer(hidden, self.adjacency))
            )

        limb_hidden = hidden[:, self.limb_index]
        features = [
            limb_hidden[:, :, 0],
            limb_hidden[:, :, 1],
            limb_hidden[:, :, 2],
            limb_hidden[:, :, 3],
            (limb_hidden[:, :, 0] - limb_hidden[:, :, 2]).abs(),
            (limb_hidden[:, :, 1] - limb_hidden[:, :, 3]).abs(),
            hidden.mean(1, keepdim=True).expand(-1, 4, -1),
            self.limb_embedding.weight[None].expand(batch, -1, -1),
        ]

        if self.use_geometry:
            if geometry is None:
                raise ValueError("geometry features are required")
            features.append(geometry)

        output = self.head(torch.cat(features, dim=-1))
        return output[..., 0], output[..., 1:]


FLIP_INDEX = torch.tensor(FLIP_PERM)


def flip_batch(x, geometry):
    flipped = x[:, FLIP_INDEX].clone()
    flipped[..., 0] = -flipped[..., 0]

    flipped_geometry = geometry.clone()
    flipped_geometry[..., 1] = -flipped_geometry[..., 1]
    return flipped, flipped_geometry


def augment_batch(x, geometry, generator):
    batch = x.shape[0]
    flip_mask = torch.rand(batch, generator=generator) < 0.5
    flipped_x, flipped_geometry = flip_batch(x, geometry)

    x = torch.where(flip_mask[:, None, None], flipped_x, x)
    geometry = torch.where(
        flip_mask[:, None, None],
        flipped_geometry,
        geometry,
    )

    theta = (
        torch.rand(batch, generator=generator) - 0.5
    ) * math.radians(30)
    rotation = torch.stack(
        [
            torch.stack([theta.cos(), -theta.sin()], dim=1),
            torch.stack([theta.sin(), theta.cos()], dim=1),
        ],
        dim=1,
    )

    scale = 0.9 + 0.2 * torch.rand(
        batch,
        1,
        1,
        generator=generator,
    )
    xy = torch.einsum(
        "bij,bnj->bni",
        rotation,
        x[..., :2],
    ) * scale

    noise = torch.randn(
        xy.shape,
        generator=generator,
    ) * 0.03 * (1.5 - x[..., 2:3].clamp(0, 1))
    xy = xy + noise

    drop = (
        torch.rand(batch, 17, 1, generator=generator) < 0.05
    ).float()

    x = torch.cat(
        [
            xy * (1 - drop),
            x[..., 2:3] * (1 - drop),
        ],
        dim=-1,
    )
    return x, geometry


def make_graph_inputs(keypoints, scores):
    normalized, _, _ = canonicalize(keypoints)
    _, limb_features = geometric_features(keypoints, scores)
    node_features = np.concatenate(
        [normalized, scores[..., None]],
        axis=-1,
    )
    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(limb_features, dtype=torch.float32),
    )
