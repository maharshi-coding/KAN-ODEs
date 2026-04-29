#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
KAN-PINN (PyTorch) for the strain-limiting PDE (Equation 40):

div( ∇Φ / [ 2μ (1 + β |∇Φ|^α)^(1/α) ] ) = 0

Features implemented per task:
- Exact PDE residual via autograd (no finite differences)
- Notched geometry sampling (rectangle minus V-notch void)
- Dirichlet BCs on Γ1-Γ5 with Γ5=0 on both notch faces
- Optional distance-function hard Dirichlet ansatz
- Nonlinear energy pretraining followed by strong-residual training
- KAN network (Gaussian-basis Kolmogorov-Arnold layers), not an MLP
- Weighted PDE residual near tip: w(x)=1/(dist_to_tip+eps)
- Adam + LR schedule + optional L-BFGS polish + grad clipping + early stopping + validation
- Outputs: loss plot, Φ field heatmap, |∇Φ| line plot
- Diagnostics: PDE residual stats, symmetry, near/far gradient ratio, finite check

Run example:
  KAN_PINN_RUN_NAME=stable_v4 python StrainLimiting_KAN_PINN.py

Environment override examples:
  KAN_PINN_NTIP=256 KAN_PINN_VAL_NTIP=512 KAN_PINN_RUN_NAME=py_run python StrainLimiting_KAN_PINN.py
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


BOUNDARY_DISPLAY = {
    "G1": "Γ1",
    "G2": "Γ2",
    "G3": "Γ3",
    "G4": "Γ4",
    "G5a": "Γ5a",
    "G5b": "Γ5b",
}

OUTER_BOUNDARY_LABELS = ("G1", "G2", "G3", "G4")
NOTCH_FACE_LABELS = ("G5a", "G5b")
ALL_BOUNDARY_LABELS = OUTER_BOUNDARY_LABELS + NOTCH_FACE_LABELS


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class MaterialParams:
    mu: float = 1.0
    beta: float = 1.0
    alpha: float = 0.2


@dataclass
class GeometryParams:
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    tip: Tuple[float, float] = (0.5, 0.5)
    notch_angle_deg: float = 20.0
    notch_length: float = 0.50
    refine_half_width: float = 0.10

    @property
    def notch_angle(self) -> float:
        return math.radians(self.notch_angle_deg)


@dataclass
class BCParams:
    sigma0: float = 1.0
    L: float = 1.0


@dataclass
class TrainParams:
    adam_epochs: int = 10000
    finetune_epochs: int = 5000
    pretrain_epochs: int = 3000
    pde_ramp_epochs: int = 9000

    n_interior_uniform: int = 256
    n_interior_refine: int = 256
    n_interior_tip_strip: int = 1536
    n_interior_tip_annulus: int = 768
    n_boundary_each: int = 128

    val_n_interior_uniform: int = 256
    val_n_interior_refine: int = 256
    val_n_interior_tip_strip: int = 2048
    val_n_interior_tip_annulus: int = 1024
    val_n_boundary_each: int = 128

    lambda_bc: float = 10.0
    lambda_gauge: float = 0.01
    lambda_sym: float = 0.5
    lambda_pde: float = 0.1
    lambda_energy: float = 1.0
    lambda_tip: float = 0.0
    lambda_tip_ratio: float = 0.0

    tip_stress_c: float = 0.25
    tip_stress_eps: float = 1e-5
    tip_ratio_target: float = 1.2
    tip_strip_bias_power: float = 2.5
    tip_loss_r_weight_power: float = 0.5

    learning_rate: float = 5e-5
    finetune_lr: float = 1e-5

    print_every: int = 10
    validation_every: int = 10
    checkpoint_every: int = 50
    detailed_diag_every: int = 100
    early_stop_patience: int = 99999
    min_improve: float = 1e-5
    max_grad_norm: float = 0.25
    diagnostics_samples: int = 512
    pointwise_nx: int = 181
    pointwise_ny: int = 181
    pointwise_boundary_each: int = 256
    pointwise_batch_size: int = 512

    # Best-model selection (physics-aware)
    model_select_start_epoch: int = 2750
    model_select_pde_weight_floor: float = 0.25

    # Singular weighting w=1/(dist_to_tip+eps)
    tip_weight_eps: float = 2e-3
    tip_weight_clip: float = 25.0
    grad_norm_eps: float = 1e-10
    initial_pde_weight: float = 1e-6
    pde_loss_mode: str = "pseudo_huber"
    pde_residual_delta: float = 25.0
    notch_face_bc_mode: str = "dirichlet_zero"
    hard_bc_mode: str = "distance_all"
    hard_bc_eps: float = 1e-5
    hard_bc_distance_scale: float = 0.25
    hard_bc_distance_power: float = 2.0
    use_tip_enhanced_sampling: bool = True

    # Sampling around tip strip
    tip_strip_half_height: float = 0.02
    tip_strip_length: float = 0.12
    tip_annulus_rmin: float = 2e-3
    tip_annulus_rmax: float = 0.12
    tip_annulus_bias_power: float = 2.0

    # Scheduler
    lr_gamma_adam: float = 0.9998
    lr_gamma_finetune: float = 0.9999

    # L-BFGS polishing after Adam stages
    lbfgs_epochs: int = 0
    lbfgs_lr: float = 0.8
    lbfgs_history_size: int = 25
    lbfgs_max_iter: int = 1
    lbfgs_n_uniform: int = 128
    lbfgs_n_refine: int = 128
    lbfgs_n_tip_strip: int = 384
    lbfgs_n_tip_annulus: int = 192
    lbfgs_n_boundary_each: int = 96

    # Memory control
    train_pde_chunk_size: int = 256
    val_pde_chunk_size: int = 256

    # Adaptive residual sampling
    adaptive_sampling: bool = False
    adaptive_candidates: int = 4096
    adaptive_topk: int = 512
    adaptive_start_epoch: int = 2750

    # Reproducibility
    seed: int = 42

    # Model shape
    hidden: int = 96
    n_basis: int = 48

    # PDE tip weighting control (0 = plain MSE, no singular weighting)
    tip_weight_power: float = 1.0
    reference_line_tip_offset: float = 2e-3
    tip_ratio_n_near: int = 128
    tip_ratio_n_far: int = 128
    tip_ratio_near_dmin: float = 8e-3
    tip_ratio_near_dmax: float = 5e-2
    tip_ratio_far_dmin: float = 0.18
    tip_ratio_far_dmax: float = 0.30


# -----------------------------
# KAN model (Gaussian basis)
# -----------------------------

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_basis: int, scale: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis

        self.coeff = nn.Parameter(scale * torch.randn(out_dim, in_dim, n_basis))
        self.lin = nn.Parameter(scale * torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        centers = torch.linspace(0.0, 1.0, n_basis)
        self.centers = nn.Parameter(centers)
        self.logwidth = nn.Parameter(torch.full((n_basis,), math.log(0.15)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        widths = torch.exp(self.logwidth) + 1e-5
        lin_part = x @ self.lin.t()  # [N, out_dim]
        z = (x.unsqueeze(-1) - self.centers.view(1, 1, -1)) / widths.view(1, 1, -1)
        bi = torch.exp(-(z ** 2))  # [N,in_dim,n_basis]
        basis_part = torch.einsum("nib,oib->no", bi, self.coeff)

        return lin_part + basis_part + self.bias.view(1, -1)


class KANPINN(nn.Module):
    def __init__(self, hidden: int = 96, n_basis: int = 48):
        super().__init__()
        self.k1 = KANLayer(2, hidden, n_basis)
        self.k2 = KANLayer(hidden, hidden, n_basis)
        self.k3 = KANLayer(hidden, hidden, n_basis)
        self.k4 = KANLayer(hidden, 1, n_basis)
        self.hard_bc_mode = "none"
        self.hard_bc_eps = 1e-12
        self.hard_bc_distance_scale = 0.15
        self.hard_bc_distance_power = 2.0
        self.geo: GeometryParams | None = None
        self.bc: BCParams | None = None

    def configure_boundary_ansatz(self, geo: GeometryParams, bc: BCParams, trn: TrainParams) -> None:
        self.geo = geo
        self.bc = bc
        self.hard_bc_mode = canonical_hard_bc_mode(trn.hard_bc_mode)
        self.hard_bc_eps = trn.hard_bc_eps
        self.hard_bc_distance_scale = trn.hard_bc_distance_scale
        self.hard_bc_distance_power = trn.hard_bc_distance_power

    def raw_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.k1(x))
        h = torch.tanh(self.k2(h))
        h = torch.tanh(self.k3(h))
        out = self.k4(h)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_forward(x).squeeze(-1)
        if self.hard_bc_mode == "none":
            return raw.unsqueeze(-1)
        if self.geo is None or self.bc is None:
            raise RuntimeError("Hard boundary ansatz requested before model.configure_boundary_ansatz(...).")
        phi = hard_boundary_ansatz(
            raw,
            x,
            self.geo,
            self.bc,
            self.hard_bc_mode,
            self.hard_bc_eps,
            self.hard_bc_distance_scale,
            self.hard_bc_distance_power,
        )
        return phi.unsqueeze(-1)


# -----------------------------
# Geometry and sampling
# -----------------------------

def notch_face_directions(geo: GeometryParams) -> Tuple[np.ndarray, np.ndarray]:
    theta = geo.notch_angle
    d_upper = np.array([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=np.float32)
    d_lower = np.array([math.cos(theta / 2.0), -math.sin(theta / 2.0)], dtype=np.float32)
    return d_upper, d_lower


def notch_mouth_points(geo: GeometryParams) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0 = geo.tip
    d_upper, d_lower = notch_face_directions(geo)
    pu = np.array([x0, y0], dtype=np.float32) + geo.notch_length * d_upper
    pl = np.array([x0, y0], dtype=np.float32) + geo.notch_length * d_lower
    return pu, pl


def point_in_notch_void(x: float, y: float, geo: GeometryParams) -> bool:
    x0, y0 = geo.tip
    if x < x0:
        return False
    dx = x - x0
    if dx > geo.notch_length:
        return False
    half_open = math.tan(geo.notch_angle / 2.0) * dx
    return abs(y - y0) <= half_open


def canonical_g5_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases = {"dirichlet_zero", "dirichlet", "zero", "g5_dirichlet_zero"}
    if normalized in aliases:
        return "dirichlet_zero"
    raise ValueError(
        "KAN_PINN_G5_MODE must enforce Γ5 as Dirichlet zero for this formulation. "
        f"Got '{mode}'. Supported: dirichlet_zero."
    )


def canonical_hard_bc_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases_none = {"none", "off", "false", "0", "penalty"}
    aliases_outer = {"outer", "distance_outer", "hard_outer"}
    aliases_all = {"all", "distance_all", "hard_all", "dirichlet_all"}
    if normalized in aliases_none:
        return "none"
    if normalized in aliases_outer:
        return "distance_outer"
    if normalized in aliases_all:
        return "distance_all"
    raise ValueError(
        f"Invalid KAN_PINN_HARD_BC_MODE='{mode}'. "
        "Use 'distance_all', 'distance_outer', or 'none'."
    )


def canonical_pde_loss_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases_mse = {"mse", "l2", "squared"}
    aliases_pseudo_huber = {"pseudo_huber", "pseudohuber", "huber", "robust"}
    if normalized in aliases_mse:
        return "mse"
    if normalized in aliases_pseudo_huber:
        return "pseudo_huber"
    raise ValueError(
        f"Invalid KAN_PINN_PDE_LOSS_MODE='{mode}'. "
        "Use 'pseudo_huber' for stable training or 'mse' for the raw squared residual objective."
    )


def dirichlet_boundary_labels(trn: TrainParams) -> Tuple[str, ...]:
    _ = canonical_g5_mode(trn.notch_face_bc_mode)
    return ALL_BOUNDARY_LABELS


def boundary_roles(trn: TrainParams) -> Dict[str, str]:
    _ = canonical_g5_mode(trn.notch_face_bc_mode)
    roles = {label: "Dirichlet" for label in OUTER_BOUNDARY_LABELS}
    roles["G5a"] = "Dirichlet-zero"
    roles["G5b"] = "Dirichlet-zero"
    return roles


def validate_configuration(mat: MaterialParams, geo: GeometryParams, trn: TrainParams) -> None:
    canonical_g5_mode(trn.notch_face_bc_mode)
    canonical_hard_bc_mode(trn.hard_bc_mode)
    canonical_pde_loss_mode(trn.pde_loss_mode)

    if mat.mu <= 0.0:
        raise ValueError(f"Invalid KAN_PINN_MU={mat.mu}. Must be > 0.")
    if mat.beta <= 0.0:
        raise ValueError(f"Invalid KAN_PINN_BETA={mat.beta}. Must be > 0.")
    if mat.alpha <= 0.0:
        raise ValueError(f"Invalid KAN_PINN_ALPHA={mat.alpha}. Must be > 0.")

    if not (geo.xmin < geo.xmax and geo.ymin < geo.ymax):
        raise ValueError("Invalid geometry bounds: require xmin < xmax and ymin < ymax.")
    tip_x, tip_y = geo.tip
    if not (geo.xmin <= tip_x <= geo.xmax and geo.ymin <= tip_y <= geo.ymax):
        raise ValueError(f"Invalid notch tip={geo.tip}; tip must lie inside domain bounds.")
    if geo.notch_length <= 0.0:
        raise ValueError(f"Invalid notch_length={geo.notch_length}. Must be > 0.")
    if not (0.0 < geo.notch_angle_deg < 180.0):
        raise ValueError(f"Invalid notch_angle_deg={geo.notch_angle_deg}. Must be in (0, 180).")

    if trn.adam_epochs < 0 or trn.finetune_epochs < 0:
        raise ValueError("Training epochs must be non-negative.")
    if trn.n_boundary_each <= 0 or trn.val_n_boundary_each <= 0:
        raise ValueError("Boundary sample counts must be > 0.")
    if trn.train_pde_chunk_size <= 0 or trn.val_pde_chunk_size <= 0:
        raise ValueError("PDE chunk sizes must be > 0.")
    if trn.diagnostics_samples <= 0:
        raise ValueError("KAN_PINN_DIAGNOSTIC_SAMPLES must be > 0.")
    if trn.pointwise_nx <= 1 or trn.pointwise_ny <= 1:
        raise ValueError("KAN_PINN_POINTWISE_NX and KAN_PINN_POINTWISE_NY must be > 1.")
    if trn.pointwise_boundary_each <= 0:
        raise ValueError("KAN_PINN_POINTWISE_BOUNDARY_EACH must be > 0.")
    if trn.pointwise_batch_size <= 0:
        raise ValueError("KAN_PINN_POINTWISE_BATCH must be > 0.")
    if trn.hard_bc_eps <= 0.0:
        raise ValueError("KAN_PINN_HARD_BC_EPS must be > 0.")
    if trn.hard_bc_distance_scale <= 0.0:
        raise ValueError("KAN_PINN_HARD_BC_DISTANCE_SCALE must be > 0.")
    if trn.hard_bc_distance_power <= 0.0:
        raise ValueError("KAN_PINN_HARD_BC_DISTANCE_POWER must be > 0.")
    if trn.pde_residual_delta <= 0.0:
        raise ValueError("KAN_PINN_PDE_RESIDUAL_DELTA must be > 0.")
    if trn.lbfgs_epochs < 0:
        raise ValueError("KAN_PINN_LBFGS_EPOCHS must be >= 0.")


def sample_points_excluding_notch(
    geo: GeometryParams,
    n: int,
    xlo: float | None = None,
    xhi: float | None = None,
    ylo: float | None = None,
    yhi: float | None = None,
) -> np.ndarray:
    xlo = geo.xmin if xlo is None else xlo
    xhi = geo.xmax if xhi is None else xhi
    ylo = geo.ymin if ylo is None else ylo
    yhi = geo.ymax if yhi is None else yhi

    pts = np.empty((n, 2), dtype=np.float32)
    k = 0
    while k < n:
        x = xlo + (xhi - xlo) * random.random()
        y = ylo + (yhi - ylo) * random.random()
        if not point_in_notch_void(x, y, geo):
            pts[k, 0] = x
            pts[k, 1] = y
            k += 1
    return pts


def sample_tip_annulus_points(geo: GeometryParams, trn: TrainParams, n: int) -> np.ndarray:
    x0, y0 = geo.tip
    rmin = max(1e-6, float(trn.tip_annulus_rmin))
    rmax = max(rmin + 1e-6, float(trn.tip_annulus_rmax))
    bias = max(1e-6, float(trn.tip_annulus_bias_power))

    pts = np.empty((n, 2), dtype=np.float32)
    k = 0
    while k < n:
        u = random.random()
        r = rmin + (rmax - rmin) * ((1.0 - u) ** bias)
        theta = -math.pi + 2.0 * math.pi * random.random()
        x = x0 + r * math.cos(theta)
        y = y0 + r * math.sin(theta)
        if geo.xmin <= x <= geo.xmax and geo.ymin <= y <= geo.ymax and (not point_in_notch_void(x, y, geo)):
            pts[k, 0] = x
            pts[k, 1] = y
            k += 1
    return pts


def sample_interior_points(
    geo: GeometryParams,
    trn: TrainParams,
    counts_override: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    counts_cfg = {
        "uniform": trn.n_interior_uniform,
        "refine": trn.n_interior_refine,
        "tip_strip": trn.n_interior_tip_strip,
        "tip_annulus": trn.n_interior_tip_annulus,
    }
    if counts_override is not None:
        counts_cfg.update(counts_override)

    parts: List[np.ndarray] = []
    region_counts: Dict[str, int] = {}

    uniform_pts = sample_points_excluding_notch(geo, counts_cfg["uniform"])
    parts.append(uniform_pts)
    region_counts["uniform"] = int(uniform_pts.shape[0])

    x0, y0 = geo.tip
    hr = geo.refine_half_width
    refine_pts = sample_points_excluding_notch(
        geo,
        counts_cfg["refine"],
        xlo=max(geo.xmin, x0 - hr),
        xhi=min(geo.xmax, x0 + hr),
        ylo=max(geo.ymin, y0 - hr),
        yhi=min(geo.ymax, y0 + hr),
    )
    parts.append(refine_pts)
    region_counts["refine_box"] = int(refine_pts.shape[0])

    if trn.use_tip_enhanced_sampling:
        tip_pts = sample_tip_strip_points(geo, trn, counts_cfg["tip_strip"])
        annulus_pts = sample_tip_annulus_points(geo, trn, counts_cfg["tip_annulus"])
        parts.extend([tip_pts, annulus_pts])
        region_counts["tip_strip"] = int(tip_pts.shape[0])
        region_counts["tip_annulus"] = int(annulus_pts.shape[0])
    else:
        region_counts["tip_strip"] = 0
        region_counts["tip_annulus"] = 0

    points = np.vstack(parts).astype(np.float32)
    region_counts["total"] = int(points.shape[0])
    return points, region_counts


def sample_interior_points_val(geo: GeometryParams, trn: TrainParams) -> Tuple[np.ndarray, Dict[str, int]]:
    return sample_interior_points(
        geo,
        trn,
        counts_override={
            "uniform": trn.val_n_interior_uniform,
            "refine": trn.val_n_interior_refine,
            "tip_strip": trn.val_n_interior_tip_strip,
            "tip_annulus": trn.val_n_interior_tip_annulus,
        },
    )


def adaptive_residual_points(
    model: nn.Module,
    geo: GeometryParams,
    mat: MaterialParams,
    trn: TrainParams,
    device: torch.device,
    n_pick: int,
) -> np.ndarray:
    if n_pick <= 0:
        return np.empty((0, 2), dtype=np.float32)

    n_candidates = max(int(trn.adaptive_candidates), int(4 * n_pick))
    candidates = sample_points_excluding_notch(geo, n_candidates)

    chunk = max(16, min(int(trn.val_pde_chunk_size), n_candidates))
    residual_abs = np.empty((n_candidates,), dtype=np.float32)

    s = 0
    while s < n_candidates:
        e = min(s + chunk, n_candidates)
        xy = to_tensor(candidates[s:e], device, requires_grad=True)
        with torch.enable_grad():
            r = pde_residual(model, xy, mat, create_graph=False)
        residual_abs[s:e] = torch.abs(r).detach().cpu().numpy().astype(np.float32)
        del xy, r
        s = e

    if n_pick >= n_candidates:
        return candidates

    top_idx = np.argpartition(residual_abs, -n_pick)[-n_pick:]
    return candidates[top_idx].astype(np.float32)


def point_in_tip_strip_region(x: float, y: float, geo: GeometryParams, trn: TrainParams) -> bool:
    x0, y0 = geo.tip
    xlo = max(geo.xmin, x0 - trn.tip_strip_length)
    xhi = min(geo.xmax, x0)
    if x < xlo or x > xhi:
        return False
    if y < max(geo.ymin, y0 - trn.tip_strip_half_height) or y > min(geo.ymax, y0 + trn.tip_strip_half_height):
        return False
    half_open = math.tan(geo.notch_angle / 2.0) * (x0 - x)
    return abs(y - y0) <= half_open


def sample_tip_strip_points(geo: GeometryParams, trn: TrainParams, n: int) -> np.ndarray:
    x0, y0 = geo.tip
    xlo = max(geo.xmin, x0 - trn.tip_strip_length)
    xhi = min(geo.xmax, x0)
    ylo = max(geo.ymin, y0 - trn.tip_strip_half_height)
    yhi = min(geo.ymax, y0 + trn.tip_strip_half_height)

    pts = np.empty((n, 2), dtype=np.float32)
    k = 0
    bias = max(1e-6, float(trn.tip_strip_bias_power))
    span = max(1e-12, float(x0 - xlo))
    while k < n:
        u = random.random()
        x = x0 - span * (u ** bias)
        x = min(max(x, xlo), xhi)
        y = ylo + (yhi - ylo) * random.random()
        if point_in_tip_strip_region(x, y, geo, trn) and (not point_in_notch_void(x, y, geo)):
            pts[k, 0] = x
            pts[k, 1] = y
            k += 1
    return pts


def filter_tip_strip_points(points: np.ndarray, geo: GeometryParams, trn: TrainParams) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    mask = np.array(
        [point_in_tip_strip_region(float(x), float(y), geo, trn) for x, y in points],
        dtype=bool,
    )
    return points[mask].astype(np.float32)


def sample_tip_ratio_line_points(
    geo: GeometryParams,
    trn: TrainParams,
    n_near: int,
    n_far: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0 = geo.tip
    near_lo = max(geo.xmin, x0 - max(trn.tip_ratio_near_dmax, trn.tip_ratio_near_dmin + 1e-4))
    near_hi = max(near_lo + 1e-4, x0 - trn.tip_ratio_near_dmin)
    far_xlo = max(geo.xmin, x0 - max(trn.tip_ratio_far_dmax, trn.tip_ratio_far_dmin + 1e-4))
    far_xhi = max(far_xlo + 1e-4, x0 - trn.tip_ratio_far_dmin)
    xnear = np.linspace(near_lo, near_hi, n_near, dtype=np.float32)
    xfar = np.linspace(far_xlo, far_xhi, n_far, dtype=np.float32)
    ynear = np.full_like(xnear, y0)
    yfar = np.full_like(xfar, y0)
    near_pts = np.stack([xnear, ynear], axis=1).astype(np.float32)
    far_pts = np.stack([xfar, yfar], axis=1).astype(np.float32)
    return near_pts, far_pts


def notch_face_points(geo: GeometryParams, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0 = geo.tip
    d1, d2 = notch_face_directions(geo)
    s = np.random.rand(n).astype(np.float32) * np.float32(geo.notch_length)

    p1 = np.stack([x0 + s * d1[0], y0 + s * d1[1]], axis=1).astype(np.float32)
    p2 = np.stack([x0 + s * d2[0], y0 + s * d2[1]], axis=1).astype(np.float32)
    return p1, p2


def sample_boundary_points(geo: GeometryParams, n_each: int) -> Dict[str, np.ndarray]:
    y1 = geo.ymin + (geo.ymax - geo.ymin) * np.random.rand(n_each).astype(np.float32)
    g1 = np.stack([np.full(n_each, geo.xmin, dtype=np.float32), y1], axis=1)

    x3 = geo.xmin + (geo.xmax - geo.xmin) * np.random.rand(n_each).astype(np.float32)
    g3 = np.stack([x3, np.full(n_each, geo.ymin, dtype=np.float32)], axis=1)

    x4 = geo.xmin + (geo.xmax - geo.xmin) * np.random.rand(n_each).astype(np.float32)
    g4 = np.stack([x4, np.full(n_each, geo.ymax, dtype=np.float32)], axis=1)

    pu, pl = notch_mouth_points(geo)
    ylo = max(geo.ymin, min(float(pl[1]), float(pu[1])))
    yhi = min(geo.ymax, max(float(pl[1]), float(pu[1])))
    g2 = np.empty((n_each, 2), dtype=np.float32)
    for i in range(n_each):
        y = geo.ymin + (geo.ymax - geo.ymin) * random.random()
        while ylo <= y <= yhi:
            y = geo.ymin + (geo.ymax - geo.ymin) * random.random()
        g2[i, 0] = geo.xmax
        g2[i, 1] = y

    g5a, g5b = notch_face_points(geo, n_each)

    return {
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5a": g5a,
        "G5b": g5b,
    }


# -----------------------------
# PDE + losses (autograd)
# -----------------------------

def to_tensor(x: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(x, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def safe_l2_norm(vec: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(vec ** 2, dim=1) + eps)


def phi_scalar(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    # xy: [N,2], returns [N]
    return model(xy).squeeze(-1)


def flux_from_grad(grad_phi: torch.Tensor, mat: MaterialParams, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    gnorm = safe_l2_norm(grad_phi, eps)
    denom = 2.0 * mat.mu * torch.pow(1.0 + mat.beta * torch.pow(gnorm, mat.alpha), 1.0 / mat.alpha)
    flux = grad_phi / denom.unsqueeze(1)
    return flux, gnorm


def boundary_normals(geo: GeometryParams, label: str, n: int) -> np.ndarray:
    if label == "G1":
        normal = np.array([-1.0, 0.0], dtype=np.float32)
    elif label == "G2":
        normal = np.array([1.0, 0.0], dtype=np.float32)
    elif label == "G3":
        normal = np.array([0.0, -1.0], dtype=np.float32)
    elif label == "G4":
        normal = np.array([0.0, 1.0], dtype=np.float32)
    elif label == "G5a":
        tangent, _ = notch_face_directions(geo)
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float32)
    elif label == "G5b":
        _, tangent = notch_face_directions(geo)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown boundary label: {label}")
    return np.repeat(normal.reshape(1, 2), n, axis=0)


def torch_segment_distance(xy: torch.Tensor, a: Tuple[float, float], b: Tuple[float, float], eps: float) -> torch.Tensor:
    a_t = torch.tensor(a, dtype=xy.dtype, device=xy.device)
    b_t = torch.tensor(b, dtype=xy.dtype, device=xy.device)
    v = b_t - a_t
    denom = torch.sum(v * v).clamp_min(eps)
    t = torch.sum((xy - a_t.view(1, 2)) * v.view(1, 2), dim=1) / denom
    t = torch.clamp(t, 0.0, 1.0)
    closest = a_t.view(1, 2) + t.view(-1, 1) * v.view(1, 2)
    return safe_l2_norm(xy - closest, eps)


def hard_boundary_distances(
    xy: torch.Tensor,
    geo: GeometryParams,
    eps: float,
    include_notch: bool,
) -> Dict[str, torch.Tensor]:
    x = xy[:, 0]
    y = xy[:, 1]
    distances: Dict[str, torch.Tensor] = {
        "G1": torch.clamp(x - geo.xmin, min=0.0),
        "G2": torch.clamp(geo.xmax - x, min=0.0),
        "G3": torch.clamp(y - geo.ymin, min=0.0),
        "G4": torch.clamp(geo.ymax - y, min=0.0),
    }
    if include_notch:
        tip = (float(geo.tip[0]), float(geo.tip[1]))
        pu, pl = notch_mouth_points(geo)
        distances["G5a"] = torch_segment_distance(xy, tip, (float(pu[0]), float(pu[1])), eps)
        distances["G5b"] = torch_segment_distance(xy, tip, (float(pl[0]), float(pl[1])), eps)
    return distances


def dirichlet_target_values(label: str, xy: torch.Tensor, bc: BCParams) -> torch.Tensor:
    x = xy[:, 0]
    if label == "G1":
        return torch.full_like(x, bc.sigma0 * bc.L)
    if label == "G2":
        return torch.zeros_like(x)
    if label == "G3":
        return -bc.sigma0 * (x - bc.L)
    if label == "G4":
        return bc.sigma0 * (bc.L - x)
    if label in NOTCH_FACE_LABELS:
        return torch.zeros_like(x)
    raise ValueError(f"Unknown Dirichlet boundary label: {label}")


def hard_boundary_ansatz(
    raw_phi: torch.Tensor,
    xy: torch.Tensor,
    geo: GeometryParams,
    bc: BCParams,
    mode: str,
    eps: float,
    distance_scale: float,
    distance_power: float,
) -> torch.Tensor:
    """
    Distance-based trial function Phi = G + D*N.

    G is an inverse-distance Dirichlet extension. D is a smooth nearest-boundary
    factor, so the trainable correction vanishes on prescribed boundaries.
    """
    mode = canonical_hard_bc_mode(mode)
    if mode == "none":
        return raw_phi

    include_notch = mode == "distance_all"
    distances = hard_boundary_distances(xy, geo, eps, include_notch=include_notch)
    labels = tuple(distances.keys())
    d_stack = torch.stack([distances[label] for label in labels], dim=1)
    d_pos = torch.clamp(d_stack, min=0.0)

    weights = 1.0 / torch.pow(d_pos + eps, distance_power)
    target_stack = torch.stack([dirichlet_target_values(label, xy, bc) for label in labels], dim=1)
    extension = torch.sum(weights * target_stack, dim=1) / torch.sum(weights, dim=1).clamp_min(eps)

    inv_nearest = torch.sum(1.0 / (d_pos + eps), dim=1)
    nearest = 1.0 / inv_nearest.clamp_min(eps)
    vanish = nearest / (nearest + distance_scale)
    return extension + vanish * raw_phi


def compute_stress(
    model: nn.Module,
    xy: torch.Tensor,
    create_graph: bool = True,
    grad_norm_eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    phi = phi_scalar(model, xy)
    grad_phi = torch.autograd.grad(
        phi,
        xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    tau_xz = grad_phi[:, 1]
    tau_yz = -grad_phi[:, 0]
    tau_eq = safe_l2_norm(torch.stack([tau_xz, tau_yz], dim=1), grad_norm_eps)
    return tau_xz, tau_yz, tau_eq


def pde_residual(
    model: nn.Module,
    xy: torch.Tensor,
    mat: MaterialParams,
    create_graph: bool = True,
    grad_norm_eps: float = 1e-10,
) -> torch.Tensor:
    """
    Residual for Eq. 40:
      div( grad(phi) / (2*mu*(1+beta*|grad(phi)|^alpha)^(1/alpha)) )
    """
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    phi = phi_scalar(model, xy)
    grad_phi = torch.autograd.grad(
        phi,
        xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]  # [N,2]

    q, _ = flux_from_grad(grad_phi, mat, grad_norm_eps)

    qx = q[:, 0]
    qy = q[:, 1]
    dqx_dx = torch.autograd.grad(
        qx,
        xy,
        grad_outputs=torch.ones_like(qx),
        create_graph=create_graph,
        retain_graph=True,
    )[0][:, 0]
    dqy_dy = torch.autograd.grad(
        qy,
        xy,
        grad_outputs=torch.ones_like(qy),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0][:, 1]

    return dqx_dx + dqy_dy


def dirichlet_target(label: str, xy: torch.Tensor, bc: BCParams, trn: TrainParams) -> torch.Tensor:
    if label in NOTCH_FACE_LABELS:
        _ = canonical_g5_mode(trn.notch_face_bc_mode)
    return dirichlet_target_values(label, xy, bc)


def tip_residual_weights(interior_xy: torch.Tensor, geo: GeometryParams, trn: TrainParams) -> torch.Tensor:
    x0, y0 = geo.tip
    dist = safe_l2_norm(
        torch.stack([interior_xy[:, 0] - x0, interior_xy[:, 1] - y0], dim=1),
        trn.grad_norm_eps,
    )
    pw = max(0.0, float(trn.tip_weight_power))
    if pw <= 0.0:
        return torch.ones_like(dist)
    raw = 1.0 / (torch.pow(dist, pw) + trn.tip_weight_eps)
    raw = raw / raw.mean().detach().clamp_min(1e-12)
    if trn.tip_weight_clip > 0.0:
        raw = torch.clamp(raw, max=trn.tip_weight_clip)
    return raw


def pde_residual_objective(weighted_residual: torch.Tensor, trn: TrainParams) -> torch.Tensor:
    mode = canonical_pde_loss_mode(trn.pde_loss_mode)
    if mode == "mse":
        return weighted_residual ** 2

    delta = torch.as_tensor(
        trn.pde_residual_delta,
        dtype=weighted_residual.dtype,
        device=weighted_residual.device,
    )
    scaled = weighted_residual / delta
    return 2.0 * delta * delta * (torch.sqrt(1.0 + scaled * scaled) - 1.0)


def weighted_pde_loss(
    model: nn.Module,
    interior_xy: torch.Tensor,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    create_graph: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    n = interior_xy.shape[0]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        res = pde_residual(model, interior_xy, mat, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
        w = tip_residual_weights(interior_xy, geo, trn)
        return torch.mean(pde_residual_objective(w * res, trn))

    total = torch.zeros((), dtype=torch.float32, device=interior_xy.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xy_chunk = interior_xy[s:e]
        res = pde_residual(model, xy_chunk, mat, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
        w = tip_residual_weights(xy_chunk, geo, trn)
        chunk_loss = torch.mean(pde_residual_objective(w * res, trn))
        total = total + chunk_loss * (e - s)

    return total / n


def strain_limiting_energy_density(grad_phi: torch.Tensor, mat: MaterialParams, eps: float) -> torch.Tensor:
    """
    Convex potential whose gradient with respect to grad_phi is the nonlinear flux.
    psi(s) = integral_0^s t / (2*mu*(1 + beta*t^alpha)^(1/alpha)) dt.
    """
    gnorm = safe_l2_norm(grad_phi, eps)
    nodes = torch.tensor(
        [
            0.0198550717512319,
            0.1016667612931866,
            0.2372337950418355,
            0.4082826787521751,
            0.5917173212478249,
            0.7627662049581645,
            0.8983332387068134,
            0.9801449282487681,
        ],
        dtype=grad_phi.dtype,
        device=grad_phi.device,
    )
    weights = torch.tensor(
        [
            0.0506142681451881,
            0.1111905172266872,
            0.1568533229389436,
            0.1813418916891809,
            0.1813418916891809,
            0.1568533229389436,
            0.1111905172266872,
            0.0506142681451881,
        ],
        dtype=grad_phi.dtype,
        device=grad_phi.device,
    )
    t = gnorm.unsqueeze(1) * nodes.view(1, -1)
    denom = 2.0 * mat.mu * torch.pow(1.0 + mat.beta * torch.pow(t + eps, mat.alpha), 1.0 / mat.alpha)
    integrand = t / denom
    return gnorm * torch.sum(weights.view(1, -1) * integrand, dim=1)


def energy_loss(
    model: nn.Module,
    interior_xy: torch.Tensor,
    mat: MaterialParams,
    trn: TrainParams,
    create_graph: bool = True,
) -> torch.Tensor:
    if not interior_xy.requires_grad:
        interior_xy = interior_xy.clone().detach().requires_grad_(True)
    phi = phi_scalar(model, interior_xy)
    grad_phi = torch.autograd.grad(
        phi,
        interior_xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    density = strain_limiting_energy_density(grad_phi, mat, trn.grad_norm_eps)
    return torch.mean(density)


def tip_stress_loss(
    model: nn.Module,
    tip_xy: torch.Tensor,
    geo: GeometryParams,
    trn: TrainParams,
    create_graph: bool = True,
) -> torch.Tensor:
    if tip_xy.shape[0] == 0:
        return torch.zeros((), dtype=torch.float32, device=tip_xy.device)

    _, _, tau_eq = compute_stress(model, tip_xy, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
    x0, y0 = geo.tip
    r = safe_l2_norm(
        torch.stack([tip_xy[:, 0] - x0, tip_xy[:, 1] - y0], dim=1),
        trn.grad_norm_eps,
    )
    singular_scaled = tau_eq * torch.sqrt(r + trn.tip_stress_eps)
    mismatch2 = (singular_scaled - trn.tip_stress_c) ** 2
    if trn.tip_loss_r_weight_power <= 0.0:
        return torch.mean(mismatch2)
    w = 1.0 / torch.pow(r + trn.tip_stress_eps, trn.tip_loss_r_weight_power)
    return torch.sum(w * mismatch2) / (torch.sum(w) + 1e-12)


def tip_stress_ratio_loss(
    model: nn.Module,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    create_graph: bool = True,
    n_near: int | None = None,
    n_far: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_near = trn.tip_ratio_n_near if n_near is None else n_near
    n_far = trn.tip_ratio_n_far if n_far is None else n_far
    near_pts, far_pts = sample_tip_ratio_line_points(geo, trn, n_near, n_far)
    near_xy = to_tensor(near_pts, device, requires_grad=True)
    far_xy = to_tensor(far_pts, device, requires_grad=True)

    _, _, tnear = compute_stress(model, near_xy, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
    _, _, tfar = compute_stress(model, far_xy, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)

    near_mean = torch.mean(tnear)
    far_mean = torch.mean(tfar)
    ratio = near_mean / (far_mean + 1e-8)
    loss = torch.relu(trn.tip_ratio_target - ratio) ** 2
    return loss, ratio


def boundary_loss_terms(
    model: nn.Module,
    bdata_t: Dict[str, torch.Tensor],
    bc: BCParams,
    trn: TrainParams,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    for label in dirichlet_boundary_labels(trn):
        if label not in bdata_t:
            continue
        xy = bdata_t[label]
        pred = phi_scalar(model, xy)
        tgt = dirichlet_target(label, xy, bc, trn)
        losses[label] = torch.mean((pred - tgt) ** 2)
    return losses


def boundary_loss(
    model: nn.Module,
    bdata_t: Dict[str, torch.Tensor],
    bc: BCParams,
    trn: TrainParams,
) -> torch.Tensor:
    losses = boundary_loss_terms(model, bdata_t, bc, trn)
    if len(losses) == 0:
        device = next(model.parameters()).device
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(list(losses.values())).mean()


def notch_face_flux_diagnostics(
    model: nn.Module,
    notch_bdata_t: Dict[str, torch.Tensor],
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
) -> Dict[str, Dict[str, float]]:
    diagnostics: Dict[str, Dict[str, float]] = {}
    for label in NOTCH_FACE_LABELS:
        xy = notch_bdata_t.get(label)
        if xy is None or xy.shape[0] == 0:
            diagnostics[label] = {"mean_abs_flux_n": float("nan"), "max_abs_flux_n": float("nan")}
            continue
        xy_req = xy.clone().detach().requires_grad_(True)
        phi = phi_scalar(model, xy_req)
        grad_phi = torch.autograd.grad(
            phi,
            xy_req,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=False,
        )[0]
        flux, _ = flux_from_grad(grad_phi, mat, trn.grad_norm_eps)
        normals = to_tensor(boundary_normals(geo, label, xy.shape[0]), xy.device, requires_grad=False)
        flux_n = torch.sum(flux * normals, dim=1)
        diagnostics[label] = {
            "mean_abs_flux_n": float(torch.mean(torch.abs(flux_n)).detach().cpu()),
            "max_abs_flux_n": float(torch.max(torch.abs(flux_n)).detach().cpu()),
        }
    return diagnostics


def gauge_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    p = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    return phi_scalar(model, p).pow(2).mean()


def symmetry_loss(model: nn.Module, geo: GeometryParams, device: torch.device, n: int = 128) -> torch.Tensor:
    x0, y0 = geo.tip
    _ = x0
    pts = sample_points_excluding_notch(geo, n, ylo=y0, yhi=geo.ymax)

    pairs_a = []
    pairs_b = []
    for x, y in pts:
        ym = 2.0 * y0 - y
        if geo.ymin <= ym <= geo.ymax and (not point_in_notch_void(float(x), float(ym), geo)):
            pairs_a.append([x, y])
            pairs_b.append([x, ym])

    if len(pairs_a) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    a_t = to_tensor(np.asarray(pairs_a, dtype=np.float32), device, requires_grad=False)
    b_t = to_tensor(np.asarray(pairs_b, dtype=np.float32), device, requires_grad=False)

    pa = phi_scalar(model, a_t)
    pb = phi_scalar(model, b_t)
    return torch.mean((pa - pb) ** 2)


def compute_losses(
    model: nn.Module,
    interior_t: torch.Tensor,
    bdata_t: Dict[str, torch.Tensor],
    mat: MaterialParams,
    geo: GeometryParams,
    bc: BCParams,
    trn: TrainParams,
    device: torch.device,
    pde_weight: float,
    validation_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pde_weight > 0.0:
        lpde = weighted_pde_loss(
            model,
            interior_t,
            mat,
            geo,
            trn,
            create_graph=(not validation_mode),
            chunk_size=(trn.val_pde_chunk_size if validation_mode else trn.train_pde_chunk_size),
        )
    else:
        lpde = torch.zeros((), dtype=torch.float32, device=device)
    lbc = boundary_loss(model, bdata_t, bc, trn)
    lg = gauge_loss(model, device)
    lsym = symmetry_loss(model, geo, device)
    return lpde, lbc, lg, lsym


def streaming_pde_backward(
    model: nn.Module,
    interior_np: np.ndarray,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    pde_weight: float,
) -> float:
    if pde_weight <= 0.0 or trn.lambda_pde <= 0.0:
        return 0.0

    n_total = interior_np.shape[0]
    chunk = max(1, int(trn.train_pde_chunk_size))
    weighted_mean = 0.0

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(interior_np[s:e], device, requires_grad=True)
            lpde_chunk = weighted_pde_loss(
                model,
                xy_chunk,
                mat,
                geo,
                trn,
                create_graph=True,
                chunk_size=None,
            )
            frac = (e - s) / n_total
            (trn.lambda_pde * pde_weight * frac * lpde_chunk).backward()
            weighted_mean += float(lpde_chunk.detach().cpu()) * (e - s)
            del xy_chunk, lpde_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at PDE chunk size 1. Reduce model size/sampling.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing train PDE chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return weighted_mean / n_total


def streaming_pde_eval(
    model: nn.Module,
    interior_np: np.ndarray,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
) -> torch.Tensor:
    n_total = interior_np.shape[0]
    chunk = max(1, int(trn.val_pde_chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=device)

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(interior_np[s:e], device, requires_grad=True)
            lpde_chunk = weighted_pde_loss(
                model,
                xy_chunk,
                mat,
                geo,
                trn,
                create_graph=False,
                chunk_size=None,
            )
            total = total + lpde_chunk * (e - s)
            del xy_chunk, lpde_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at validation PDE chunk size 1.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing val PDE chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return total / n_total


def streaming_energy_backward(
    model: nn.Module,
    interior_np: np.ndarray,
    mat: MaterialParams,
    trn: TrainParams,
    device: torch.device,
    energy_weight: float,
) -> float:
    if energy_weight <= 0.0 or trn.lambda_energy <= 0.0:
        return 0.0

    n_total = interior_np.shape[0]
    chunk = max(1, int(trn.train_pde_chunk_size))
    weighted_mean = 0.0

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(interior_np[s:e], device, requires_grad=True)
            lenergy_chunk = energy_loss(model, xy_chunk, mat, trn, create_graph=True)
            frac = (e - s) / n_total
            (trn.lambda_energy * energy_weight * frac * lenergy_chunk).backward()
            weighted_mean += float(lenergy_chunk.detach().cpu()) * (e - s)
            del xy_chunk, lenergy_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at energy chunk size 1. Reduce model size/sampling.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing energy chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return weighted_mean / n_total


def streaming_energy_eval(
    model: nn.Module,
    interior_np: np.ndarray,
    mat: MaterialParams,
    trn: TrainParams,
    device: torch.device,
) -> torch.Tensor:
    n_total = interior_np.shape[0]
    chunk = max(1, int(trn.val_pde_chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=device)

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(interior_np[s:e], device, requires_grad=True)
            lenergy_chunk = energy_loss(model, xy_chunk, mat, trn, create_graph=False)
            total = total + lenergy_chunk * (e - s)
            del xy_chunk, lenergy_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at validation energy chunk size 1.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing val energy chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return total / n_total


def streaming_tip_stress_backward(
    model: nn.Module,
    tip_np: np.ndarray,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
) -> float:
    if trn.lambda_tip <= 0.0 or tip_np.shape[0] == 0:
        return 0.0

    n_total = tip_np.shape[0]
    chunk = max(1, int(trn.train_pde_chunk_size))
    weighted_mean = 0.0

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(tip_np[s:e], device, requires_grad=True)
            ltip_chunk = tip_stress_loss(
                model,
                xy_chunk,
                geo,
                trn,
                create_graph=True,
            )
            frac = (e - s) / n_total
            (trn.lambda_tip * frac * ltip_chunk).backward()
            weighted_mean += float(ltip_chunk.detach().cpu()) * (e - s)
            del xy_chunk, ltip_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at tip-stress chunk size 1.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing tip-stress chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return weighted_mean / n_total


def streaming_tip_stress_eval(
    model: nn.Module,
    tip_np: np.ndarray,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
) -> torch.Tensor:
    if tip_np.shape[0] == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    n_total = tip_np.shape[0]
    chunk = max(1, int(trn.val_pde_chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=device)

    s = 0
    while s < n_total:
        e = min(s + chunk, n_total)
        try:
            xy_chunk = to_tensor(tip_np[s:e], device, requires_grad=True)
            ltip_chunk = tip_stress_loss(
                model,
                xy_chunk,
                geo,
                trn,
                create_graph=False,
            )
            total = total + ltip_chunk * (e - s)
            del xy_chunk, ltip_chunk
            s = e
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at validation tip-stress chunk size 1.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing val tip-stress chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk

    return total / n_total


def pde_curriculum_weight(epoch: int, trn: TrainParams) -> float:
    if epoch <= trn.pretrain_epochs:
        return 0.0
    phase2_epoch = epoch - trn.pretrain_epochs
    start = min(1.0, max(0.0, trn.initial_pde_weight))
    if trn.pde_ramp_epochs <= 0:
        return 1.0
    ramp = min(1.0, phase2_epoch / max(1, trn.pde_ramp_epochs))
    return start + (1.0 - start) * ramp


def energy_curriculum_weight(epoch: int, trn: TrainParams) -> float:
    if trn.pretrain_epochs <= 0:
        return 0.0
    return 1.0 if epoch <= trn.pretrain_epochs else 0.0


# -----------------------------
# Verification diagnostics
# -----------------------------

@torch.no_grad()
def field_on_grid(model: nn.Module, geo: GeometryParams, device: torch.device, nx: int = 121, ny: int = 121):
    xs = np.linspace(geo.xmin, geo.xmax, nx, dtype=np.float32)
    ys = np.linspace(geo.ymin, geo.ymax, ny, dtype=np.float32)

    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    xy_t = to_tensor(grid, device, requires_grad=False)
    phi = phi_scalar(model, xy_t).cpu().numpy().reshape(ny, nx)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if point_in_notch_void(float(x), float(y), geo):
                phi[iy, ix] = np.nan

    return xs, ys, phi


def grad_mag(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.clone().detach().requires_grad_(True)
    p = phi_scalar(model, xy)
    g = torch.autograd.grad(
        p,
        xy,
        grad_outputs=torch.ones_like(p),
        create_graph=False,
        retain_graph=False,
    )[0]
    return safe_l2_norm(g, 1e-10)


def residual_statistics(model: nn.Module, mat: MaterialParams, geo: GeometryParams, device: torch.device, n: int = 512):
    pts = sample_points_excluding_notch(geo, n)
    xy = to_tensor(pts, device, requires_grad=True)
    r = pde_residual(model, xy, mat, grad_norm_eps=1e-10).detach().cpu().numpy()
    abs_r = np.abs(r)
    return {
        "mean_abs": float(abs_r.mean()),
        "max_abs": float(abs_r.max()),
        "rms": float(np.sqrt(np.mean(r ** 2))),
    }


def symmetry_error(model: nn.Module, geo: GeometryParams, device: torch.device, n: int = 512):
    x0, y0 = geo.tip
    pts = sample_points_excluding_notch(geo, n, ylo=y0, yhi=geo.ymax)

    pairs_a = []
    pairs_b = []
    for x, y in pts:
        ym = 2.0 * y0 - y
        if geo.ymin <= ym <= geo.ymax and (not point_in_notch_void(float(x), float(ym), geo)):
            pairs_a.append([x, y])
            pairs_b.append([x, ym])

    if len(pairs_a) == 0:
        return {"mean_abs": float("nan"), "max_abs": float("nan"), "n_pairs": 0}

    a_t = to_tensor(np.asarray(pairs_a, dtype=np.float32), device, requires_grad=False)
    b_t = to_tensor(np.asarray(pairs_b, dtype=np.float32), device, requires_grad=False)

    with torch.no_grad():
        pa = phi_scalar(model, a_t)
        pb = phi_scalar(model, b_t)
        err = torch.abs(pa - pb).cpu().numpy()

    return {
        "mean_abs": float(err.mean()),
        "max_abs": float(err.max()),
        "n_pairs": int(err.size),
    }


def tip_gradient_indicator(model: nn.Module, geo: GeometryParams, trn: TrainParams, device: torch.device):
    near_pts, far_pts = sample_tip_ratio_line_points(geo, trn, trn.tip_ratio_n_near, trn.tip_ratio_n_far)
    near_t = to_tensor(near_pts, device, requires_grad=True)
    far_t = to_tensor(far_pts, device, requires_grad=True)

    _, _, tnear = compute_stress(model, near_t, create_graph=False)
    _, _, tfar = compute_stress(model, far_t, create_graph=False)
    gnear = tnear.detach().cpu().numpy()
    gfar = tfar.detach().cpu().numpy()

    near_mean = float(gnear.mean())
    far_mean = float(gfar.mean())
    ratio = near_mean / (far_mean + 1e-8)
    return {"near_mean": near_mean, "far_mean": far_mean, "ratio": ratio}


def grid_finite_check(model: nn.Module, geo: GeometryParams, device: torch.device, nx: int = 121, ny: int = 121):
    xs, ys, phi = field_on_grid(model, geo, device, nx=nx, ny=ny)
    bad_outside = 0
    outside_total = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            inside_void = point_in_notch_void(float(x), float(y), geo)
            v = phi[iy, ix]
            if not inside_void:
                outside_total += 1
                if (not np.isfinite(v)) or np.isnan(v):
                    bad_outside += 1
    return {"outside_total": outside_total, "bad_outside": bad_outside}


def region_statistics(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    n: int = 512,
) -> Dict[str, Dict[str, float]]:
    x0, _ = geo.tip
    near_pts = sample_tip_annulus_points(geo, trn, n)
    far_xlo = max(geo.xmin, x0 - max(trn.tip_ratio_far_dmax, trn.tip_ratio_far_dmin + 1e-4))
    far_xhi = max(far_xlo + 1e-4, x0 - trn.tip_ratio_far_dmin)
    far_pts = sample_points_excluding_notch(geo, n, xlo=far_xlo, xhi=far_xhi)
    stats: Dict[str, Dict[str, float]] = {}
    for label, pts in (("near_tip", near_pts), ("far_field", far_pts)):
        xy = to_tensor(pts, device, requires_grad=True)
        _, _, tau_eq = compute_stress(model, xy, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        residual = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        tau_np = tau_eq.detach().cpu().numpy()
        res_np = np.abs(residual.detach().cpu().numpy())
        stats[label] = {
            "tau_eq_mean": float(tau_np.mean()),
            "tau_eq_max": float(tau_np.max()),
            "residual_mean_abs": float(res_np.mean()),
            "residual_max_abs": float(res_np.max()),
        }
    return stats


def boundary_diagnostics(
    model: nn.Module,
    bdata_t: Dict[str, torch.Tensor],
    bc: BCParams,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
) -> Dict[str, Dict[str, float]]:
    losses = boundary_loss_terms(model, bdata_t, bc, trn)
    diag: Dict[str, Dict[str, float]] = {}
    for label in ALL_BOUNDARY_LABELS:
        xy = bdata_t.get(label)
        diag[label] = {
            "count": int(0 if xy is None else xy.shape[0]),
            "loss": float("nan"),
        }
        if label in losses:
            diag[label]["loss"] = float(losses[label].detach().cpu())
    flux_diag = notch_face_flux_diagnostics(model, bdata_t, mat, geo, trn)
    for label, vals in flux_diag.items():
        diag[label].update(vals)
    return diag


def run_cross_verification(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    boundary_diag: Dict[str, Dict[str, float]] | None = None,
):
    rstats = residual_statistics(model, mat, geo, device)
    sstats = symmetry_error(model, geo, device)
    tipstats = tip_gradient_indicator(model, geo, trn, device)
    gstats = grid_finite_check(model, geo, device)
    region_stats = region_statistics(model, mat, geo, trn, device)

    print("Cross verification summary:")
    print(
        "  PDE residual  | "
        f"mean|r|={rstats['mean_abs']:.5e}, rms={rstats['rms']:.5e}, max|r|={rstats['max_abs']:.5e}"
    )
    print(
        "  Symmetry      | "
        f"mean|ΔΦ|={sstats['mean_abs']:.5e}, max|ΔΦ|={sstats['max_abs']:.5e} (pairs={sstats['n_pairs']})"
    )
    print(
        "  Tip stress ratio (τ_eq) | "
        f"near={tipstats['near_mean']:.5e}, far={tipstats['far_mean']:.5e}, near/far={tipstats['ratio']:.3f}"
    )
    print(
        "  Finite check  | "
        f"bad outside notch={gstats['bad_outside']} / {gstats['outside_total']}"
    )
    print(
        "  Regional stats| "
        f"near_tip τeq(mean/max)=({region_stats['near_tip']['tau_eq_mean']:.5e}, {region_stats['near_tip']['tau_eq_max']:.5e}), "
        f"far_field τeq(mean/max)=({region_stats['far_field']['tau_eq_mean']:.5e}, {region_stats['far_field']['tau_eq_max']:.5e})"
    )
    print(
        "  Regional PDE  | "
        f"near_tip mean|max|r|=({region_stats['near_tip']['residual_mean_abs']:.5e}, {region_stats['near_tip']['residual_max_abs']:.5e}), "
        f"far_field mean|max|r|=({region_stats['far_field']['residual_mean_abs']:.5e}, {region_stats['far_field']['residual_max_abs']:.5e})"
    )
    if boundary_diag is not None:
        for label in ALL_BOUNDARY_LABELS:
            info = boundary_diag[label]
            msg = f"  {BOUNDARY_DISPLAY[label]:<12}| role={boundary_roles(trn)[label]}"
            if np.isfinite(info.get("loss", float("nan"))):
                msg += f", loss={info['loss']:.5e}"
            if "mean_abs_flux_n" in info:
                msg += (
                    f", mean|q·n|={info['mean_abs_flux_n']:.5e}, "
                    f"max|q·n|={info['max_abs_flux_n']:.5e}"
                )
            print(msg)
    return {
        "residual": rstats,
        "symmetry": sstats,
        "tip_ratio": tipstats,
        "finite": gstats,
        "regions": region_stats,
        "boundary": boundary_diag,
    }


# -----------------------------
# Plotting
# -----------------------------

def reference_line_arrays(geo: GeometryParams, trn: TrainParams, n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0 = geo.tip
    x_tip = max(geo.xmin, x0 - max(1e-5, trn.reference_line_tip_offset))
    xline = np.linspace(geo.xmin, x_tip, n, dtype=np.float32)
    yline = np.full_like(xline, y0)
    return xline, yline


def field_diagnostics_on_grid(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    nx: int = 181,
    ny: int = 181,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    xs = np.linspace(geo.xmin, geo.xmax, nx, dtype=np.float32)
    ys = np.linspace(geo.ymin, geo.ymax, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    phi = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    gradmag = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    tau_eq = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    residual = np.full((grid.shape[0],), np.nan, dtype=np.float32)

    for s in range(0, grid.shape[0], batch_size):
        e = min(s + batch_size, grid.shape[0])
        batch = grid[s:e]
        mask = np.array([not point_in_notch_void(float(x), float(y), geo) for x, y in batch], dtype=bool)
        if not np.any(mask):
            continue
        batch_valid = batch[mask]
        xy = to_tensor(batch_valid, device, requires_grad=True)
        phi_batch = phi_scalar(model, xy)
        _, _, tau_batch = compute_stress(model, xy, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        res_batch = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        idx = np.where(mask)[0] + s
        phi[idx] = phi_batch.detach().cpu().numpy().astype(np.float32)
        tau_np = tau_batch.detach().cpu().numpy().astype(np.float32)
        tau_eq[idx] = tau_np
        gradmag[idx] = tau_np
        residual[idx] = res_batch.detach().cpu().numpy().astype(np.float32)

    return {
        "xs": xs,
        "ys": ys,
        "phi": phi.reshape(ny, nx),
        "grad_mag": gradmag.reshape(ny, nx),
        "tau_eq": tau_eq.reshape(ny, nx),
        "residual": residual.reshape(ny, nx),
    }


def save_run_diagnostics(
    outdir: Path,
    trn: TrainParams,
    geo: GeometryParams,
    mat: MaterialParams,
    bc: BCParams,
    collocation_counts: Dict[str, int],
    boundary_diag: Dict[str, Dict[str, float]],
    verification: Dict[str, object],
    fields: Dict[str, np.ndarray],
    reference_line: Dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "field_diagnostics.npz",
        xs=fields["xs"],
        ys=fields["ys"],
        phi=fields["phi"],
        grad_mag=fields["grad_mag"],
        tau_eq=fields["tau_eq"],
        residual=fields["residual"],
    )
    np.savez_compressed(
        outdir / "reference_line_diagnostics.npz",
        x=reference_line["x"],
        y=reference_line["y"],
        distance_to_tip=reference_line["distance_to_tip"],
        tau_eq=reference_line["tau_eq"],
        grad_mag=reference_line["grad_mag"],
    )

    ref_csv = outdir / "reference_line_tau_eq.csv"
    with ref_csv.open("w", encoding="utf-8") as fh:
        fh.write("x,y,distance_to_tip,tau_eq,grad_mag\n")
        for x, y, d, tau, grad in zip(
            reference_line["x"],
            reference_line["y"],
            reference_line["distance_to_tip"],
            reference_line["tau_eq"],
            reference_line["grad_mag"],
        ):
            fh.write(f"{x:.8f},{y:.8f},{d:.8f},{tau:.8e},{grad:.8e}\n")

    summary = {
        "boundary_roles": boundary_roles(trn),
        "g5_mode": trn.notch_face_bc_mode,
        "training": {
            "adam_epochs": trn.adam_epochs,
            "finetune_epochs": trn.finetune_epochs,
            "lbfgs_epochs": trn.lbfgs_epochs,
            "pretrain_epochs": trn.pretrain_epochs,
            "pde_ramp_epochs": trn.pde_ramp_epochs,
            "lambda_bc": trn.lambda_bc,
            "lambda_pde": trn.lambda_pde,
            "lambda_energy": trn.lambda_energy,
            "lambda_tip": trn.lambda_tip,
            "lambda_tip_ratio": trn.lambda_tip_ratio,
            "hard_bc_mode": trn.hard_bc_mode,
            "hard_bc_eps": trn.hard_bc_eps,
            "hard_bc_distance_scale": trn.hard_bc_distance_scale,
            "hard_bc_distance_power": trn.hard_bc_distance_power,
            "tip_stress_c": trn.tip_stress_c,
            "tip_ratio_target": trn.tip_ratio_target,
            "initial_pde_weight": trn.initial_pde_weight,
            "pde_loss_mode": trn.pde_loss_mode,
            "pde_residual_delta": trn.pde_residual_delta,
            "model_select_start_epoch": trn.model_select_start_epoch,
            "model_select_pde_weight_floor": trn.model_select_pde_weight_floor,
            "adaptive_sampling": trn.adaptive_sampling,
            "adaptive_start_epoch": trn.adaptive_start_epoch,
            "tip_ratio_n_near": trn.tip_ratio_n_near,
            "tip_ratio_n_far": trn.tip_ratio_n_far,
            "tip_ratio_near_dmin": trn.tip_ratio_near_dmin,
            "tip_ratio_near_dmax": trn.tip_ratio_near_dmax,
            "tip_ratio_far_dmin": trn.tip_ratio_far_dmin,
            "tip_ratio_far_dmax": trn.tip_ratio_far_dmax,
            "pointwise_nx": trn.pointwise_nx,
            "pointwise_ny": trn.pointwise_ny,
            "pointwise_boundary_each": trn.pointwise_boundary_each,
        },
        "collocation_counts": collocation_counts,
        "material": {"mu": mat.mu, "beta": mat.beta, "alpha": mat.alpha},
        "geometry": {
            "xmin": geo.xmin,
            "xmax": geo.xmax,
            "ymin": geo.ymin,
            "ymax": geo.ymax,
            "tip": list(geo.tip),
            "notch_angle_deg": geo.notch_angle_deg,
            "notch_length": geo.notch_length,
        },
        "boundary_conditions": {"sigma0": bc.sigma0, "L": bc.L},
        "boundary_diagnostics": boundary_diag,
        "verification": verification,
    }
    (outdir / "run_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_loss_history_text(
    outdir: Path,
    loss_hist: list[float],
    pde_hist: list[float],
    energy_hist: list[float],
    bc_hist: list[float],
    val_hist: list[float],
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "loss_history.csv"
    n = max(len(loss_hist), len(pde_hist), len(energy_hist), len(bc_hist), len(val_hist))

    def get(values: list[float], idx: int) -> float:
        return values[idx] if idx < len(values) else float("nan")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("epoch,total_loss,pde_loss,energy_loss,boundary_loss,validation_loss\n")
        for idx in range(n):
            fh.write(
                f"{idx + 1},"
                f"{get(loss_hist, idx):.10e},"
                f"{get(pde_hist, idx):.10e},"
                f"{get(energy_hist, idx):.10e},"
                f"{get(bc_hist, idx):.10e},"
                f"{get(val_hist, idx):.10e}\n"
            )
    return out_path


def pointwise_quantities(
    model: nn.Module,
    pts_np: np.ndarray,
    mat: MaterialParams,
    trn: TrainParams,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    xy = to_tensor(pts_np, device, requires_grad=True)
    phi = phi_scalar(model, xy)
    grad_phi = torch.autograd.grad(
        phi,
        xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]
    _, grad_norm = flux_from_grad(grad_phi, mat, trn.grad_norm_eps)
    tau_xz = grad_phi[:, 1]
    tau_yz = -grad_phi[:, 0]
    tau_eq = safe_l2_norm(torch.stack([tau_xz, tau_yz], dim=1), trn.grad_norm_eps)
    residual = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
    energy = strain_limiting_energy_density(grad_phi, mat, trn.grad_norm_eps)

    return {
        "phi": phi.detach().cpu().numpy().astype(np.float64),
        "grad_norm": grad_norm.detach().cpu().numpy().astype(np.float64),
        "tau_eq": tau_eq.detach().cpu().numpy().astype(np.float64),
        "pde_residual": residual.detach().cpu().numpy().astype(np.float64),
        "pde_loss": torch.square(residual).detach().cpu().numpy().astype(np.float64),
        "energy_density": energy.detach().cpu().numpy().astype(np.float64),
    }


def save_pointwise_loss_text(
    outdir: Path,
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    bc: BCParams,
    trn: TrainParams,
    device: torch.device,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "pointwise_loss.csv"
    xs = np.linspace(geo.xmin, geo.xmax, trn.pointwise_nx + 2, dtype=np.float32)[1:-1]
    ys = np.linspace(geo.ymin, geo.ymax, trn.pointwise_ny + 2, dtype=np.float32)[1:-1]
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    interior_mask = np.array([not point_in_notch_void(float(x), float(y), geo) for x, y in grid], dtype=bool)
    interior_pts = grid[interior_mask]
    boundary_pts = sample_boundary_points(geo, trn.pointwise_boundary_each)

    header = (
        "point_type,boundary_label,x,y,phi,target,boundary_error,"
        "pde_residual,pde_loss,energy_density,grad_norm,tau_eq,total_point_loss\n"
    )

    def fmt_optional(value: float | None) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.10e}"

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(header)

        for s in range(0, interior_pts.shape[0], trn.pointwise_batch_size):
            e = min(s + trn.pointwise_batch_size, interior_pts.shape[0])
            pts = interior_pts[s:e]
            q = pointwise_quantities(model, pts, mat, trn, device)
            total = trn.lambda_pde * q["pde_loss"] + trn.lambda_energy * q["energy_density"]
            for i, (x, y) in enumerate(pts):
                fh.write(
                    "interior,,"
                    f"{float(x):.10e},{float(y):.10e},"
                    f"{q['phi'][i]:.10e},nan,nan,"
                    f"{q['pde_residual'][i]:.10e},{q['pde_loss'][i]:.10e},"
                    f"{q['energy_density'][i]:.10e},{q['grad_norm'][i]:.10e},"
                    f"{q['tau_eq'][i]:.10e},{total[i]:.10e}\n"
                )

        for label in ALL_BOUNDARY_LABELS:
            pts = np.asarray(boundary_pts[label], dtype=np.float32)
            for s in range(0, pts.shape[0], trn.pointwise_batch_size):
                e = min(s + trn.pointwise_batch_size, pts.shape[0])
                batch = pts[s:e]
                xy = to_tensor(batch, device, requires_grad=False)
                with torch.no_grad():
                    phi = phi_scalar(model, xy).detach().cpu().numpy().astype(np.float64)
                target = dirichlet_target(label, xy, bc, trn).detach().cpu().numpy().astype(np.float64)
                boundary_error = phi - target
                boundary_loss = np.square(boundary_error)
                total = trn.lambda_bc * boundary_loss
                for i, (x, y) in enumerate(batch):
                    fh.write(
                        f"boundary,{label},"
                        f"{float(x):.10e},{float(y):.10e},"
                        f"{phi[i]:.10e},{target[i]:.10e},{boundary_error[i]:.10e},"
                        "nan,nan,nan,nan,nan,"
                        f"{fmt_optional(total[i])}\n"
                    )

    return out_path


@torch.no_grad()
def save_boundary_points_text(
    outdir: Path,
    bdata: Dict[str, np.ndarray],
    model: nn.Module,
    bc: BCParams,
    trn: TrainParams,
    device: torch.device,
    source: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "boundary_points_training.txt"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# source={source}\n")
        fh.write("boundary_label,x,y,phi_pred,phi_target,abs_error\n")
        for label in ALL_BOUNDARY_LABELS:
            pts = bdata.get(label)
            if pts is None or pts.size == 0:
                continue
            pts_np = np.asarray(pts, dtype=np.float32)
            xy = to_tensor(pts_np, device, requires_grad=False)
            pred = phi_scalar(model, xy).detach().cpu().numpy()
            tgt = dirichlet_target(label, xy, bc, trn).detach().cpu().numpy()
            for (x, y), p, t in zip(pts_np, pred, tgt):
                fh.write(f"{label},{x:.8f},{y:.8f},{p:.8e},{t:.8e},{abs(float(p - t)):.8e}\n")
    return out_path


def save_plots(
    model: nn.Module,
    mat: MaterialParams,
    bc: BCParams,
    loss_hist: list[float],
    pde_hist: list[float],
    energy_hist: list[float],
    bc_hist: list[float],
    val_hist: list[float],
    geo: GeometryParams,
    trn: TrainParams,
    outdir: Path,
    device: torch.device,
    boundary_diag: Dict[str, Dict[str, float]],
    collocation_counts: Dict[str, int],
    verification: Dict[str, object],
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    loss_csv = save_loss_history_text(outdir, loss_hist, pde_hist, energy_hist, bc_hist, val_hist)
    print(f"Loss history saved in: {loss_csv}")
    pointwise_csv = save_pointwise_loss_text(outdir, model, mat, geo, bc, trn, device)
    print(f"Pointwise loss saved in: {pointwise_csv}")

    # Loss history
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, lw=2, label="L total")
    plt.plot(pde_hist, lw=2, label="L_pde")
    if len(energy_hist) > 0:
        plt.plot(energy_hist, lw=2, label="L_energy")
    plt.plot(bc_hist, lw=2, label="L_bc")
    if len(val_hist) > 0:
        plt.plot(val_hist, lw=2, label="L_val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss_history.png", dpi=160)
    plt.close()

    fields = field_diagnostics_on_grid(model, mat, geo, trn, device, nx=181, ny=181)
    xs = fields["xs"]
    ys = fields["ys"]
    phi = fields["phi"]

    # Phi field
    plt.figure(figsize=(6, 5))
    plt.imshow(
        phi,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="auto",
        cmap="turbo",
    )
    plt.colorbar(label="Φ(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Stress function Φ(x,y)")
    plt.tight_layout()
    plt.savefig(outdir / "phi_field.png", dpi=160)
    plt.close()

    def plot_field(field: np.ndarray, title: str, label: str, filename: str, cmap: str = "turbo") -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(
            field,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar(label=label)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()

    plot_field(fields["grad_mag"], "|∇Φ| field", "|∇Φ|", "grad_phi_field.png")
    plot_field(fields["tau_eq"], "Equivalent stress field", "τ_eq", "tau_eq_field.png")
    plot_field(fields["residual"], "PDE residual field", "Residual", "pde_residual_field.png", cmap="coolwarm")

    # tau_eq approaching tip along y=y0
    x0, y0 = geo.tip
    xline, yline = reference_line_arrays(geo, trn, n=300)
    xy = to_tensor(np.stack([xline, yline], axis=1), device, requires_grad=True)
    _, _, tau_eq_line = compute_stress(model, xy, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
    gline = tau_eq_line.detach().cpu().numpy()
    dist_to_tip = x0 - xline

    plt.figure(figsize=(7, 4))
    plt.plot(dist_to_tip, gline, lw=2)
    plt.xlabel("Distance to notch tip")
    plt.ylabel("τ_eq")
    plt.title("Equivalent shear stress along reference line")
    plt.tight_layout()
    plt.savefig(outdir / "tau_eq_reference_line.png", dpi=160)
    plt.close()

    reference_line = {
        "x": xline,
        "y": yline,
        "distance_to_tip": dist_to_tip,
        "tau_eq": gline,
        "grad_mag": gline.copy(),
    }
    save_run_diagnostics(outdir, trn, geo, mat, bc, collocation_counts, boundary_diag, verification, fields, reference_line)


# -----------------------------
# Training
# -----------------------------

def get_run_outdir(root_outdir: Path, run_name: str | None = None) -> Tuple[Path, str]:
    root_outdir.mkdir(parents=True, exist_ok=True)
    if run_name is None or run_name.strip() == "":
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = root_outdir / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    (root_outdir / "latest_run.txt").write_text(run_name + "\n", encoding="utf-8")
    return outdir, run_name


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 0:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def format_bc_loss_line(bc_loss_values: Dict[str, float]) -> str:
    parts: List[str] = []
    for label in ALL_BOUNDARY_LABELS:
        val = bc_loss_values.get(label, float("nan"))
        if math.isfinite(val):
            parts.append(f"{BOUNDARY_DISPLAY[label]}={val:.2e}")
        else:
            parts.append(f"{BOUNDARY_DISPLAY[label]}=nan")
    return ", ".join(parts)


def maybe_float(tensor_val: torch.Tensor | float) -> float:
    if isinstance(tensor_val, float):
        return tensor_val
    return float(tensor_val.detach().cpu())


def all_finite(values: Dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in values.values())


def train_model(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    bc: BCParams,
    trn: TrainParams,
    outdir: Path,
    device: torch.device,
    resume: bool = False,
):
    total_epochs = trn.adam_epochs + trn.finetune_epochs + trn.lbfgs_epochs

    val_interior, val_collocation_counts = sample_interior_points_val(geo, trn)
    val_tip_interior = filter_tip_strip_points(val_interior, geo, trn)
    val_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)
    val_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in val_bdata.items()}

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0
    last_collocation_counts: Dict[str, int] = {}
    last_boundary_points: Dict[str, np.ndarray] = {}

    loss_hist: list[float] = []
    pde_hist: list[float] = []
    energy_hist: list[float] = []
    bc_hist: list[float] = []
    tip_hist: list[float] = []
    tip_ratio_hist: list[float] = []
    val_hist: list[float] = []
    val_select_hist: list[float] = []

    best_ckpt_path = outdir / "best_checkpoint.pt"
    last_ckpt_path = outdir / "last_checkpoint.pt"

    def checkpoint_payload(saved_epoch: int, reason: str) -> Dict[str, object]:
        last_state = copy.deepcopy(model.state_dict())
        return {
            "model_state": last_state,
            "last_model_state": last_state,
            "best_model_state": copy.deepcopy(best_state),
            "best_epoch": best_epoch,
            "best_val": best_val,
            "loss_total": loss_hist,
            "loss_pde": pde_hist,
            "loss_energy": energy_hist,
            "loss_bc": bc_hist,
            "loss_tip": tip_hist,
            "loss_tip_ratio": tip_ratio_hist,
            "loss_val": val_hist,
            "loss_val_select": val_select_hist,
            "completed_epochs": len(loss_hist),
            "saved_epoch": saved_epoch,
            "saved_reason": reason,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "boundary_roles": boundary_roles(trn),
            "g5_mode": trn.notch_face_bc_mode,
            "hard_bc_mode": trn.hard_bc_mode,
            "last_collocation_counts": last_collocation_counts,
            "last_boundary_points": {k: v.copy() for k, v in last_boundary_points.items()},
            "val_collocation_counts": val_collocation_counts,
        }

    def save_checkpoints(saved_epoch: int, reason: str, save_best: bool = False, verbose: bool = False) -> None:
        payload = checkpoint_payload(saved_epoch, reason)
        torch.save(payload, last_ckpt_path)
        if save_best:
            payload_best = dict(payload)
            payload_best["checkpoint_kind"] = "best"
            torch.save(payload_best, best_ckpt_path)
        save_loss_history_text(outdir, loss_hist, pde_hist, energy_hist, bc_hist, val_hist)
        if verbose:
            best_msg = " + best_checkpoint.pt" if save_best else ""
            print(
                f"[checkpoint] epoch={saved_epoch} reason={reason} -> "
                f"last_checkpoint.pt{best_msg}"
            )

    def evaluate_validation(pde_weight: float, energy_weight: float) -> Tuple[float, float]:
        model.eval()
        with torch.enable_grad():
            if pde_weight > 0.0 or trn.model_select_pde_weight_floor > 0.0:
                v_lpde = streaming_pde_eval(model, val_interior, mat, geo, trn, device)
            else:
                v_lpde = torch.zeros((), dtype=torch.float32, device=device)
            if energy_weight > 0.0:
                v_lenergy = streaming_energy_eval(model, val_interior, mat, trn, device)
            else:
                v_lenergy = torch.zeros((), dtype=torch.float32, device=device)
            if trn.lambda_tip > 0.0 and val_tip_interior.shape[0] > 0:
                v_ltip = streaming_tip_stress_eval(model, val_tip_interior, geo, trn, device)
            else:
                v_ltip = torch.zeros((), dtype=torch.float32, device=device)
            if trn.lambda_tip_ratio > 0.0:
                v_lratio, _ = tip_stress_ratio_loss(model, geo, trn, device, create_graph=False)
            else:
                v_lratio = torch.zeros((), dtype=torch.float32, device=device)
            v_lbc = boundary_loss(model, val_bdata_t, bc, trn)
            v_lg = gauge_loss(model, device)
            v_lsym = symmetry_loss(model, geo, device)
            lval = (
                trn.lambda_pde * pde_weight * v_lpde
                + trn.lambda_energy * energy_weight * v_lenergy
                + trn.lambda_tip * v_ltip
                + trn.lambda_tip_ratio * v_lratio
                + trn.lambda_bc * v_lbc
                + trn.lambda_gauge * v_lg
                + trn.lambda_sym * v_lsym
            )
            select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
            lval_select = (
                trn.lambda_pde * select_wpde * v_lpde
                + trn.lambda_energy * energy_weight * v_lenergy
                + trn.lambda_tip * v_ltip
                + trn.lambda_tip_ratio * v_lratio
                + trn.lambda_bc * v_lbc
                + trn.lambda_gauge * v_lg
                + trn.lambda_sym * v_lsym
            )
        return maybe_float(lval), maybe_float(lval_select)

    resume_source = ""
    if resume:
        if last_ckpt_path.is_file():
            resume_source = str(last_ckpt_path.name)
        elif best_ckpt_path.is_file():
            resume_source = str(best_ckpt_path.name)

    if resume_source:
        resume_path = outdir / resume_source
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        last_state_key = "last_model_state" if "last_model_state" in ckpt else "model_state"
        best_state_key = "best_model_state" if "best_model_state" in ckpt else last_state_key
        if last_state_key in ckpt:
            model.load_state_dict(ckpt[last_state_key])
            best_state = copy.deepcopy(ckpt.get(best_state_key, ckpt[last_state_key]))
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_val = float(ckpt.get("best_val", float("inf")))
            loss_hist = list(ckpt.get("loss_total", []))
            pde_hist = list(ckpt.get("loss_pde", []))
            energy_hist = list(ckpt.get("loss_energy", [0.0] * len(loss_hist)))
            bc_hist = list(ckpt.get("loss_bc", []))
            tip_hist = list(ckpt.get("loss_tip", []))
            tip_ratio_hist = list(ckpt.get("loss_tip_ratio", []))
            val_hist = list(ckpt.get("loss_val", []))
            val_select_hist = list(ckpt.get("loss_val_select", ckpt.get("loss_val", [])))
            completed_epochs = int(ckpt.get("completed_epochs", len(loss_hist)))
            last_collocation_counts = dict(ckpt.get("last_collocation_counts", {}))
            loaded_boundary_points = ckpt.get("last_boundary_points", {})
            if isinstance(loaded_boundary_points, dict):
                last_boundary_points = {
                    str(k): np.asarray(v, dtype=np.float32)
                    for k, v in loaded_boundary_points.items()
                }
            stale_epochs = 0
            print(
                f"[resume] source={resume_source} epoch={completed_epochs}/{total_epochs} "
                f"best_epoch={best_epoch} best_val={best_val:.6e}"
            )
    elif resume:
        print("[resume] Requested resume but no checkpoint found; starting from scratch.")

    t0 = time.time()
    session_epoch_start = max(1, completed_epochs + 1)

    if completed_epochs >= total_epochs:
        print(f"Checkpoint already reached target epochs ({completed_epochs}). Skipping training.")
        if best_epoch > 0:
            model.load_state_dict(best_state)
        return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
            "train_last": last_collocation_counts,
            "validation": val_collocation_counts,
        }, last_boundary_points

    def run_stage(
        stage_name: str,
        start_epoch: int,
        end_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> bool:
        nonlocal best_state, best_epoch, best_val, stale_epochs, last_collocation_counts, last_boundary_points
        if start_epoch > end_epoch:
            return False
        print(
            f"[stage] {stage_name}: epochs {start_epoch}..{end_epoch} | "
            f"lr_start={current_lr(optimizer):.3e}"
        )

        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            interior, collocation_counts = sample_interior_points(geo, trn)
            collocation_counts["adaptive"] = 0
            pde_weight = pde_curriculum_weight(epoch, trn)
            energy_weight = energy_curriculum_weight(epoch, trn)

            if trn.adaptive_sampling and pde_weight > 0.0 and epoch >= trn.adaptive_start_epoch:
                try:
                    n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                    if n_adapt > 0:
                        adapt_pts = adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
                        if adapt_pts.size > 0:
                            interior = np.vstack([interior, adapt_pts]).astype(np.float32)
                            collocation_counts["adaptive"] = int(adapt_pts.shape[0])
                            collocation_counts["total"] = int(interior.shape[0])
                except RuntimeError as exc:
                    print(f"[adaptive sampling] RuntimeError; skipping adaptive points this epoch. {exc}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            bdata = sample_boundary_points(geo, trn.n_boundary_each)
            bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}

            optimizer.zero_grad(set_to_none=True)
            bc_terms = boundary_loss_terms(model, bdata_t, bc, trn)
            lbc = torch.stack(list(bc_terms.values())).mean()
            lg = gauge_loss(model, device)
            lsym = symmetry_loss(model, geo, device)
            base_loss = trn.lambda_bc * lbc + trn.lambda_gauge * lg + trn.lambda_sym * lsym
            base_loss.backward()

            ltip_f = 0.0
            lratio_f = 0.0
            lenergy_f = streaming_energy_backward(model, interior, mat, trn, device, energy_weight)
            ratio_f = float("nan")
            if trn.lambda_tip > 0.0:
                tip_interior = sample_tip_strip_points(geo, trn, trn.n_interior_tip_strip)
                ltip_f = streaming_tip_stress_backward(model, tip_interior, geo, trn, device)
            if trn.lambda_tip_ratio > 0.0:
                lratio_t, ratio_t = tip_stress_ratio_loss(model, geo, trn, device, create_graph=True)
                (trn.lambda_tip_ratio * lratio_t).backward()
                lratio_f = maybe_float(lratio_t)
                ratio_f = maybe_float(ratio_t)

            lpde_f = streaming_pde_backward(model, interior, mat, geo, trn, device, pde_weight)
            lbc_f = maybe_float(lbc)
            lg_f = maybe_float(lg)
            lsym_f = maybe_float(lsym)
            ltot_f = (
                trn.lambda_bc * lbc_f
                + trn.lambda_gauge * lg_f
                + trn.lambda_sym * lsym_f
                + trn.lambda_energy * energy_weight * lenergy_f
                + trn.lambda_tip * ltip_f
                + trn.lambda_tip_ratio * lratio_f
                + trn.lambda_pde * pde_weight * lpde_f
            )
            loss_parts = {
                "total": ltot_f,
                "pde": lpde_f,
                "energy": lenergy_f,
                "bc": lbc_f,
                "gauge": lg_f,
                "sym": lsym_f,
                "tip": ltip_f,
                "tip_ratio": lratio_f,
            }
            if not all_finite(loss_parts):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} before optimizer step; losses={loss_parts}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_before_step_epoch_{epoch}", save_best=False, verbose=True)
                return True

            if trn.max_grad_norm > 0.0:
                grad_norm = maybe_float(torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm))
            else:
                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += maybe_float(torch.sum(p.grad.detach() ** 2))
                grad_norm = math.sqrt(grad_sq)
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} non-finite gradient norm={grad_norm}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_grad_epoch_{epoch}", save_best=False, verbose=True)
                return True

            optimizer.step()
            scheduler.step()
            lr_now = current_lr(optimizer)

            do_validate = (epoch == start_epoch) or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
            if do_validate:
                lval_f, lval_select_f = evaluate_validation(pde_weight, energy_weight)
            else:
                lval_f = val_hist[-1] if len(val_hist) > 0 else float("nan")
                lval_select_f = val_select_hist[-1] if len(val_select_hist) > 0 else float("nan")

            loss_hist.append(ltot_f)
            pde_hist.append(lpde_f)
            energy_hist.append(lenergy_f)
            bc_hist.append(lbc_f)
            tip_hist.append(ltip_f)
            tip_ratio_hist.append(lratio_f)
            val_hist.append(lval_f)
            val_select_hist.append(lval_select_f)

            new_best = False
            if do_validate:
                if (not math.isfinite(best_val)) or (best_epoch == 0):
                    best_val = lval_select_f
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                    new_best = True
                elif epoch >= trn.model_select_start_epoch and lval_select_f < best_val - trn.min_improve:
                    best_val = lval_select_f
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                    new_best = True
                elif epoch >= trn.model_select_start_epoch:
                    stale_epochs += 1

            periodic_ckpt = trn.checkpoint_every > 0 and (epoch % trn.checkpoint_every == 0)
            if new_best:
                save_checkpoints(epoch, reason="new_best", save_best=True, verbose=True)
            elif periodic_ckpt:
                save_checkpoints(epoch, reason="periodic", save_best=False, verbose=True)

            should_log = (epoch == start_epoch) or (trn.print_every > 0 and epoch % trn.print_every == 0)
            if should_log:
                elapsed = time.time() - t0
                epochs_done_this_session = max(1, epoch - session_epoch_start + 1)
                sec_per_epoch = elapsed / epochs_done_this_session
                eta_s = sec_per_epoch * max(0, total_epochs - epoch)
                best_disp = best_val if math.isfinite(best_val) else float("nan")
                ckpt_flag = "best+last" if new_best else ("last" if periodic_ckpt else "-")
                val_tag = "val" if do_validate else "val(skip)"
                print(
                    f"Epoch {epoch:5d}/{total_epochs} | L={ltot_f:.5e} | Lpde={lpde_f:.5e} | "
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lg={lg_f:.5e} | Lsym={lsym_f:.5e} | "
                    f"Ltip={ltip_f:.5e} | LtipRatio={lratio_f:.5e} (ratio={ratio_f:.3f}) | "
                    f"Lval={lval_f:.5e} ({val_tag}) | lr={lr_now:.3e} | grad={grad_norm:.3e} | "
                    f"wpde={pde_weight:.3f} | wE={energy_weight:.3f} | Nint={collocation_counts['total']} "
                    f"(tip_strip={collocation_counts['tip_strip']}, tip_annulus={collocation_counts['tip_annulus']}, "
                    f"adapt={collocation_counts['adaptive']}) | best={best_disp:.5e}@{best_epoch} | "
                    f"new_best={'yes' if new_best else 'no'} | ckpt={ckpt_flag} | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
                bc_loss_values = {k: maybe_float(v) for k, v in bc_terms.items()}
                print(f"  BC(train): {format_bc_loss_line(bc_loss_values)}")

            detailed = (epoch == start_epoch) or (trn.detailed_diag_every > 0 and epoch % trn.detailed_diag_every == 0)
            if detailed:
                model.eval()
                with torch.enable_grad():
                    rstats = residual_statistics(model, mat, geo, device, n=trn.diagnostics_samples)
                    tstats = tip_gradient_indicator(model, geo, trn, device)
                    region = region_statistics(
                        model,
                        mat,
                        geo,
                        trn,
                        device,
                        n=max(64, trn.diagnostics_samples // 2),
                    )
                    bdiag = boundary_diagnostics(model, val_bdata_t, bc, mat, geo, trn)

                print(
                    "  Diag(PDE): "
                    f"mean|r|={rstats['mean_abs']:.4e}, rms={rstats['rms']:.4e}, max|r|={rstats['max_abs']:.4e}"
                )
                print(
                    "  Diag(Tip): "
                    f"near={tstats['near_mean']:.4e}, far={tstats['far_mean']:.4e}, near/far={tstats['ratio']:.3f} | "
                    f"near_tip mean|max|r|=({region['near_tip']['residual_mean_abs']:.4e}, "
                    f"{region['near_tip']['residual_max_abs']:.4e})"
                )
                bdiag_losses: Dict[str, float] = {}
                for label in ALL_BOUNDARY_LABELS:
                    info = bdiag.get(label, {})
                    bdiag_losses[label] = float(info.get("loss", float("nan")))
                print(f"  Diag(BC,val): {format_bc_loss_line(bdiag_losses)}")
                g5a_flux = bdiag["G5a"].get("mean_abs_flux_n", float("nan"))
                g5b_flux = bdiag["G5b"].get("mean_abs_flux_n", float("nan"))
                print(f"  Diag(Flux): Γ5a mean|q·n|={g5a_flux:.4e}, Γ5b mean|q·n|={g5b_flux:.4e}")

            if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
                print(f"[early-stop] Triggered at epoch {epoch}; best_epoch={best_epoch}")
                return True

        return False

    def run_lbfgs_stage(start_epoch: int, end_epoch: int) -> bool:
        nonlocal best_state, best_epoch, best_val, stale_epochs, last_collocation_counts, last_boundary_points
        if start_epoch > end_epoch:
            return False

        print(
            f"[stage] lbfgs: epochs {start_epoch}..{end_epoch} | "
            f"lr={trn.lbfgs_lr:.3e} history={trn.lbfgs_history_size}"
        )
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=trn.lbfgs_lr,
            max_iter=max(1, trn.lbfgs_max_iter),
            history_size=max(1, trn.lbfgs_history_size),
            line_search_fn="strong_wolfe",
        )

        counts_override = {
            "uniform": trn.lbfgs_n_uniform,
            "refine": trn.lbfgs_n_refine,
            "tip_strip": trn.lbfgs_n_tip_strip,
            "tip_annulus": trn.lbfgs_n_tip_annulus,
        }

        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            interior, collocation_counts = sample_interior_points(geo, trn, counts_override=counts_override)
            collocation_counts["adaptive"] = 0
            pde_weight = 1.0
            energy_weight = 0.0
            bdata = sample_boundary_points(geo, trn.lbfgs_n_boundary_each)
            bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            interior_t = to_tensor(interior, device, requires_grad=True)
            tip_interior = sample_tip_strip_points(geo, trn, max(1, trn.lbfgs_n_tip_strip))
            tip_t = to_tensor(tip_interior, device, requires_grad=True)
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}

            closure_vals: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                lpde = weighted_pde_loss(
                    model,
                    interior_t,
                    mat,
                    geo,
                    trn,
                    create_graph=True,
                    chunk_size=trn.train_pde_chunk_size,
                )
                lbc = boundary_loss(model, bdata_t, bc, trn)
                lg = gauge_loss(model, device)
                lsym = symmetry_loss(model, geo, device)
                ltip = (
                    tip_stress_loss(model, tip_t, geo, trn, create_graph=True)
                    if trn.lambda_tip > 0.0
                    else torch.zeros((), dtype=torch.float32, device=device)
                )
                if trn.lambda_tip_ratio > 0.0:
                    lratio, ratio = tip_stress_ratio_loss(model, geo, trn, device, create_graph=True)
                else:
                    lratio = torch.zeros((), dtype=torch.float32, device=device)
                    ratio = torch.tensor(float("nan"), dtype=torch.float32, device=device)

                loss = (
                    trn.lambda_pde * pde_weight * lpde
                    + trn.lambda_bc * lbc
                    + trn.lambda_gauge * lg
                    + trn.lambda_sym * lsym
                    + trn.lambda_tip * ltip
                    + trn.lambda_tip_ratio * lratio
                )
                loss.backward()
                if trn.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm)
                closure_vals.update(
                    {
                        "total": maybe_float(loss),
                        "pde": maybe_float(lpde),
                        "bc": maybe_float(lbc),
                        "gauge": maybe_float(lg),
                        "sym": maybe_float(lsym),
                        "tip": maybe_float(ltip),
                        "tip_ratio": maybe_float(lratio),
                        "ratio": maybe_float(ratio),
                    }
                )
                return loss

            optimizer.step(closure)

            lpde_f = closure_vals.get("pde", float("nan"))
            lenergy_f = 0.0
            lbc_f = closure_vals.get("bc", float("nan"))
            lg_f = closure_vals.get("gauge", float("nan"))
            lsym_f = closure_vals.get("sym", float("nan"))
            ltip_f = closure_vals.get("tip", 0.0)
            lratio_f = closure_vals.get("tip_ratio", 0.0)
            ratio_f = closure_vals.get("ratio", float("nan"))
            ltot_f = closure_vals.get("total", float("nan"))

            grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq += maybe_float(torch.sum(p.grad.detach() ** 2))
            grad_norm = math.sqrt(max(0.0, grad_sq))

            do_validate = (epoch == start_epoch) or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
            if do_validate:
                lval_f, lval_select_f = evaluate_validation(pde_weight, energy_weight)
            else:
                lval_f = val_hist[-1] if len(val_hist) > 0 else float("nan")
                lval_select_f = val_select_hist[-1] if len(val_select_hist) > 0 else float("nan")

            loss_hist.append(ltot_f)
            pde_hist.append(lpde_f)
            energy_hist.append(lenergy_f)
            bc_hist.append(lbc_f)
            tip_hist.append(ltip_f)
            tip_ratio_hist.append(lratio_f)
            val_hist.append(lval_f)
            val_select_hist.append(lval_select_f)

            new_best = False
            if do_validate:
                if lval_select_f < best_val - trn.min_improve:
                    best_val = lval_select_f
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                    new_best = True
                else:
                    stale_epochs += 1

            periodic_ckpt = trn.checkpoint_every > 0 and (epoch % trn.checkpoint_every == 0)
            if new_best:
                save_checkpoints(epoch, reason="new_best_lbfgs", save_best=True, verbose=True)
            elif periodic_ckpt:
                save_checkpoints(epoch, reason="periodic_lbfgs", save_best=False, verbose=True)

            should_log = (epoch == start_epoch) or (trn.print_every > 0 and epoch % trn.print_every == 0)
            if should_log:
                elapsed = time.time() - t0
                epochs_done_this_session = max(1, epoch - session_epoch_start + 1)
                sec_per_epoch = elapsed / epochs_done_this_session
                eta_s = sec_per_epoch * max(0, total_epochs - epoch)
                best_disp = best_val if math.isfinite(best_val) else float("nan")
                ckpt_flag = "best+last" if new_best else ("last" if periodic_ckpt else "-")
                val_tag = "val" if do_validate else "val(skip)"
                print(
                    f"Epoch {epoch:5d}/{total_epochs} | L={ltot_f:.5e} | Lpde={lpde_f:.5e} | "
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lg={lg_f:.5e} | Lsym={lsym_f:.5e} | "
                    f"Ltip={ltip_f:.5e} | LtipRatio={lratio_f:.5e} (ratio={ratio_f:.3f}) | "
                    f"Lval={lval_f:.5e} ({val_tag}) | lr={trn.lbfgs_lr:.3e} | grad={grad_norm:.3e} | "
                    f"wpde={pde_weight:.3f} | wE={energy_weight:.3f} | Nint={collocation_counts['total']} "
                    f"(tip_strip={collocation_counts['tip_strip']}, tip_annulus={collocation_counts['tip_annulus']}, "
                    f"adapt={collocation_counts['adaptive']}) | best={best_disp:.5e}@{best_epoch} | "
                    f"new_best={'yes' if new_best else 'no'} | ckpt={ckpt_flag} | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
                bc_terms = boundary_loss_terms(model, bdata_t, bc, trn)
                bc_loss_values = {k: maybe_float(v) for k, v in bc_terms.items()}
                print(f"  BC(train): {format_bc_loss_line(bc_loss_values)}")

            if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
                print(f"[early-stop] Triggered during L-BFGS at epoch {epoch}; best_epoch={best_epoch}")
                return True

        return False

    stopped_early = False

    adam_start = max(1, completed_epochs + 1)
    adam_end = min(trn.adam_epochs, total_epochs)
    if adam_start <= adam_end:
        opt_adam = torch.optim.Adam(model.parameters(), lr=trn.learning_rate)
        sch_adam = torch.optim.lr_scheduler.ExponentialLR(opt_adam, gamma=trn.lr_gamma_adam)
        stopped_early = run_stage("adam", adam_start, adam_end, opt_adam, sch_adam)
    else:
        print(f"[stage] adam skipped (start={adam_start}, end={adam_end}).")

    finetune_start = max(completed_epochs + 1, trn.adam_epochs + 1)
    finetune_end = min(trn.adam_epochs + trn.finetune_epochs, total_epochs)
    if (not stopped_early) and finetune_start <= finetune_end:
        opt_fine = torch.optim.Adam(model.parameters(), lr=trn.finetune_lr)
        sch_fine = torch.optim.lr_scheduler.ExponentialLR(opt_fine, gamma=trn.lr_gamma_finetune)
        stopped_early = run_stage("finetune", finetune_start, finetune_end, opt_fine, sch_fine)
    elif not stopped_early:
        print(f"[stage] finetune skipped (start={finetune_start}, end={finetune_end}).")

    lbfgs_start = max(completed_epochs + 1, trn.adam_epochs + trn.finetune_epochs + 1)
    if (not stopped_early) and trn.lbfgs_epochs > 0 and lbfgs_start <= total_epochs:
        stopped_early = run_lbfgs_stage(lbfgs_start, total_epochs)
    elif not stopped_early and trn.lbfgs_epochs > 0:
        print(f"[stage] lbfgs skipped (start={lbfgs_start}, end={total_epochs}).")

    if best_epoch <= 0:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(loss_hist)
        best_val = val_select_hist[-1] if len(val_select_hist) > 0 else float("nan")

    model.load_state_dict(best_state)
    final_epoch = len(loss_hist)
    save_checkpoints(final_epoch, reason="training_complete", save_best=True, verbose=True)

    elapsed_total = time.time() - t0
    print("Training summary:")
    print(f"  Completed epochs this run: {max(0, final_epoch - completed_epochs)}")
    print(f"  Total tracked epochs: {final_epoch} / {total_epochs}")
    print(f"  Best validation epoch: {best_epoch}")
    print(f"  Best validation score: {best_val:.6e}")
    print(f"  Final train loss: {loss_hist[-1]:.6e}" if len(loss_hist) > 0 else "  Final train loss: n/a")
    print(f"  Final val loss: {val_hist[-1]:.6e}" if len(val_hist) > 0 else "  Final val loss: n/a")
    print(f"  Early stopping used: {'yes' if stopped_early else 'no'}")
    print(f"  Runtime (this invocation): {elapsed_total/60:.2f} min")
    print(f"  Checkpoints: best={best_ckpt_path.name}, last={last_ckpt_path.name}")

    return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
        "train_last": last_collocation_counts,
        "validation": val_collocation_counts,
    }, last_boundary_points


# -----------------------------
# Main
# -----------------------------

def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def main():
    default_pretrain_epochs = env_int("KAN_PINN_PRETRAIN_EPOCHS", 3000)
    default_pde_ramp_epochs = env_int("KAN_PINN_PDE_RAMP_EPOCHS", 9000)
    default_model_select_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)
    default_adaptive_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)

    trn = TrainParams(
        adam_epochs=env_int("KAN_PINN_ADAM_EPOCHS", env_int("KAN_PINN_EPOCHS", 10000)),
        finetune_epochs=env_int("KAN_PINN_FINETUNE_EPOCHS", 5000),
        pretrain_epochs=default_pretrain_epochs,
        pde_ramp_epochs=default_pde_ramp_epochs,
        n_interior_uniform=env_int("KAN_PINN_NU", 256),
        n_interior_refine=env_int("KAN_PINN_NR", 256),
        n_interior_tip_strip=env_int("KAN_PINN_NTIP", 1536),
        n_interior_tip_annulus=env_int("KAN_PINN_NANNULUS", 768),
        n_boundary_each=env_int("KAN_PINN_NB", 128),
        val_n_interior_uniform=env_int("KAN_PINN_VAL_NU", 256),
        val_n_interior_refine=env_int("KAN_PINN_VAL_NR", 256),
        val_n_interior_tip_strip=env_int("KAN_PINN_VAL_NTIP", 2048),
        val_n_interior_tip_annulus=env_int("KAN_PINN_VAL_NANNULUS", 1024),
        val_n_boundary_each=env_int("KAN_PINN_VAL_NB", 128),
        lambda_bc=env_float("KAN_PINN_LAMBDA_BC", 10.0),
        lambda_sym=env_float("KAN_PINN_LAMBDA_SYM", 0.5),
        lambda_pde=env_float("KAN_PINN_LAMBDA_PDE", 0.1),
        lambda_energy=env_float("KAN_PINN_LAMBDA_ENERGY", 1.0),
        lambda_tip=env_float("KAN_PINN_LAMBDA_TIP", 0.0),
        lambda_tip_ratio=env_float("KAN_PINN_LAMBDA_TIP_RATIO", 0.0),
        learning_rate=env_float("KAN_PINN_LR", 5e-5),
        finetune_lr=env_float("KAN_PINN_FINETUNE_LR", 1e-5),
        print_every=env_int("KAN_PINN_PRINT_EVERY", 10),
        validation_every=env_int("KAN_PINN_VAL_EVERY", 10),
        checkpoint_every=env_int("KAN_PINN_CHECKPOINT_EVERY", 50),
        detailed_diag_every=env_int("KAN_PINN_DETAILED_DIAG_EVERY", 100),
        early_stop_patience=env_int("KAN_PINN_PATIENCE", 99999),
        min_improve=env_float("KAN_PINN_MIN_IMPROVE", 1e-5),
        max_grad_norm=env_float("KAN_PINN_MAX_GRAD_NORM", 0.25),
        diagnostics_samples=env_int("KAN_PINN_DIAGNOSTIC_SAMPLES", 512),
        pointwise_nx=env_int("KAN_PINN_POINTWISE_NX", 181),
        pointwise_ny=env_int("KAN_PINN_POINTWISE_NY", 181),
        pointwise_boundary_each=env_int("KAN_PINN_POINTWISE_BOUNDARY_EACH", 256),
        pointwise_batch_size=env_int("KAN_PINN_POINTWISE_BATCH", 512),
        model_select_start_epoch=env_int("KAN_PINN_MODEL_SELECT_START_EPOCH", default_model_select_start),
        model_select_pde_weight_floor=env_float("KAN_PINN_MODEL_SELECT_PDE_FLOOR", 0.25),
        train_pde_chunk_size=env_int("KAN_PINN_TRAIN_PDE_CHUNK", 256),
        val_pde_chunk_size=env_int("KAN_PINN_VAL_PDE_CHUNK", 256),
        tip_weight_eps=env_float("KAN_PINN_TIP_WEIGHT_EPS", 2e-3),
        tip_weight_clip=env_float("KAN_PINN_TIP_WEIGHT_CLIP", 25.0),
        grad_norm_eps=env_float("KAN_PINN_GRAD_NORM_EPS", 1e-10),
        initial_pde_weight=env_float("KAN_PINN_INITIAL_PDE_WEIGHT", 1e-6),
        pde_loss_mode=os.getenv("KAN_PINN_PDE_LOSS_MODE", "pseudo_huber").strip(),
        pde_residual_delta=env_float("KAN_PINN_PDE_RESIDUAL_DELTA", 25.0),
        notch_face_bc_mode=os.getenv("KAN_PINN_G5_MODE", "dirichlet_zero").strip(),
        hard_bc_mode=os.getenv("KAN_PINN_HARD_BC_MODE", "distance_all").strip(),
        hard_bc_eps=env_float("KAN_PINN_HARD_BC_EPS", 1e-5),
        hard_bc_distance_scale=env_float("KAN_PINN_HARD_BC_DISTANCE_SCALE", 0.25),
        hard_bc_distance_power=env_float("KAN_PINN_HARD_BC_DISTANCE_POWER", 2.0),
        use_tip_enhanced_sampling=env_bool("KAN_PINN_USE_TIP_ENHANCED_SAMPLING", True),
        tip_strip_half_height=env_float("KAN_PINN_TIP_STRIP_HH", 0.02),
        tip_strip_length=env_float("KAN_PINN_TIP_STRIP_LEN", 0.12),
        tip_annulus_rmin=env_float("KAN_PINN_TIP_ANNULUS_RMIN", 2e-3),
        tip_annulus_rmax=env_float("KAN_PINN_TIP_ANNULUS_RMAX", 0.12),
        tip_annulus_bias_power=env_float("KAN_PINN_TIP_ANNULUS_BIAS_POWER", 2.0),
        tip_stress_c=env_float("KAN_PINN_TIP_STRESS_C", 0.25),
        tip_stress_eps=env_float("KAN_PINN_TIP_STRESS_EPS", 1e-5),
        tip_ratio_target=env_float("KAN_PINN_TIP_RATIO_TARGET", 1.2),
        tip_strip_bias_power=env_float("KAN_PINN_TIP_STRIP_BIAS_POWER", 2.5),
        tip_loss_r_weight_power=env_float("KAN_PINN_TIP_R_WEIGHT_POWER", 0.5),
        lbfgs_epochs=env_int("KAN_PINN_LBFGS_EPOCHS", 0),
        lbfgs_lr=env_float("KAN_PINN_LBFGS_LR", 0.8),
        lbfgs_history_size=env_int("KAN_PINN_LBFGS_HISTORY", 25),
        lbfgs_max_iter=env_int("KAN_PINN_LBFGS_MAX_ITER", 1),
        lbfgs_n_uniform=env_int("KAN_PINN_LBFGS_NU", 128),
        lbfgs_n_refine=env_int("KAN_PINN_LBFGS_NR", 128),
        lbfgs_n_tip_strip=env_int("KAN_PINN_LBFGS_NTIP", 384),
        lbfgs_n_tip_annulus=env_int("KAN_PINN_LBFGS_NANNULUS", 192),
        lbfgs_n_boundary_each=env_int("KAN_PINN_LBFGS_NB", 96),
        adaptive_sampling=env_bool("KAN_PINN_ADAPTIVE_SAMPLING", False),
        adaptive_candidates=env_int("KAN_PINN_ADAPTIVE_CANDIDATES", 4096),
        adaptive_topk=env_int("KAN_PINN_ADAPTIVE_TOPK", 512),
        adaptive_start_epoch=env_int("KAN_PINN_ADAPTIVE_START_EPOCH", default_adaptive_start),
        seed=env_int("KAN_PINN_SEED", 42),
        tip_weight_power=env_float("KAN_PINN_TIP_WEIGHT_POWER", 1.0),
        reference_line_tip_offset=env_float("KAN_PINN_REFERENCE_LINE_TIP_OFFSET", 2e-3),
        tip_ratio_n_near=env_int("KAN_PINN_TIP_RATIO_N_NEAR", 128),
        tip_ratio_n_far=env_int("KAN_PINN_TIP_RATIO_N_FAR", 128),
        tip_ratio_near_dmin=env_float("KAN_PINN_TIP_RATIO_NEAR_DMIN", 8e-3),
        tip_ratio_near_dmax=env_float("KAN_PINN_TIP_RATIO_NEAR_DMAX", 5e-2),
        tip_ratio_far_dmin=env_float("KAN_PINN_TIP_RATIO_FAR_DMIN", 0.18),
        tip_ratio_far_dmax=env_float("KAN_PINN_TIP_RATIO_FAR_DMAX", 0.30),
        hidden=env_int("KAN_PINN_HIDDEN", 96),
        n_basis=env_int("KAN_PINN_N_BASIS", 48),
    )

    mat = MaterialParams(
        mu=env_float("KAN_PINN_MU", 1.0),
        beta=env_float("KAN_PINN_BETA", 1.0),
        alpha=env_float("KAN_PINN_ALPHA", 0.2),
    )

    geo = GeometryParams(
        xmin=env_float("KAN_PINN_XMIN", 0.0),
        xmax=env_float("KAN_PINN_XMAX", 1.0),
        ymin=env_float("KAN_PINN_YMIN", 0.0),
        ymax=env_float("KAN_PINN_YMAX", 1.0),
        tip=(env_float("KAN_PINN_TIP_X", 0.5), env_float("KAN_PINN_TIP_Y", 0.5)),
        notch_angle_deg=env_float("KAN_PINN_NOTCH_ANGLE_DEG", 20.0),
        notch_length=env_float("KAN_PINN_NOTCH_LENGTH", 0.50),
        refine_half_width=env_float("KAN_PINN_REFINE_HALF_WIDTH", 0.10),
    )

    bc = BCParams(
        sigma0=env_float("KAN_PINN_SIGMA0", 1.0),
        L=env_float("KAN_PINN_L", 1.0),
    )

    trn.notch_face_bc_mode = canonical_g5_mode(trn.notch_face_bc_mode)
    trn.hard_bc_mode = canonical_hard_bc_mode(trn.hard_bc_mode)
    trn.pde_loss_mode = canonical_pde_loss_mode(trn.pde_loss_mode)
    validate_configuration(mat, geo, trn)

    run_name = os.getenv("KAN_PINN_RUN_NAME", "").strip()
    resume_training = os.getenv("KAN_PINN_RESUME", "0").strip().lower() in ("1", "true", "yes", "y")

    random.seed(trn.seed)
    np.random.seed(trn.seed)
    torch.manual_seed(trn.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpl_dir = Path("/tmp/mplconfig_kan_odes")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    print("Starting training (Eq. 40 interior + Dirichlet BCs on Γ1-Γ5, with Γ5=0).")
    print(f"Device: {device}")
    print(f"Γ5 treatment: {boundary_roles(trn)['G5a']}")
    print(f"Hard BC ansatz: {canonical_hard_bc_mode(trn.hard_bc_mode)}")
    print(
        f"PDE curriculum: lambda={trn.lambda_pde:g}, start_weight={trn.initial_pde_weight:g}, "
        f"ramp={trn.pde_ramp_epochs}, loss={trn.pde_loss_mode}"
    )
    print(
        f"Training phases: energy_pretrain={trn.pretrain_epochs}, "
        f"adam={trn.adam_epochs}, finetune={trn.finetune_epochs}, lbfgs={trn.lbfgs_epochs}"
    )

    model = KANPINN(hidden=trn.hidden, n_basis=trn.n_basis).to(device)
    model.configure_boundary_ansatz(geo, bc, trn)

    root_outdir = Path(__file__).resolve().parent / "results_strainlimiting_python"
    outdir, selected_run = get_run_outdir(root_outdir, run_name if run_name else None)
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")

    model, best_epoch, best_val, lhist, lpde_hist, lenergy_hist, lbc_hist, val_hist, collocation_counts, train_boundary_points = train_model(
        model, mat, geo, bc, trn, outdir, device, resume=resume_training
    )

    final_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)
    final_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in final_bdata.items()}
    boundary_points_source = "last_training_epoch" if len(train_boundary_points) > 0 else "post_training_validation_sample"
    boundary_points_to_save = train_boundary_points if len(train_boundary_points) > 0 else final_bdata
    boundary_points_txt = save_boundary_points_text(
        outdir,
        boundary_points_to_save,
        model,
        bc,
        trn,
        device,
        source=boundary_points_source,
    )
    print(f"Boundary datapoints saved in: {boundary_points_txt}")
    final_boundary_diag = boundary_diagnostics(model, final_bdata_t, bc, mat, geo, trn)
    verification = run_cross_verification(model, mat, geo, trn, device, boundary_diag=final_boundary_diag)
    save_plots(
        model,
        mat,
        bc,
        lhist,
        lpde_hist,
        lenergy_hist,
        lbc_hist,
        val_hist,
        geo,
        trn,
        outdir,
        device,
        final_boundary_diag,
        collocation_counts,
        verification,
    )

    print(f"Training complete. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
