#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
KAN-PINN (PyTorch) for the strain-limiting PDE (Equation 40):

div( ∇Φ / [ 2μ (1 + β |∇Φ|^α)^(1/α) ] ) = 0

Features implemented per task:
- Exact PDE residual via autograd (no finite differences)
- Notched geometry sampling (rectangle minus V-notch void)
- Dirichlet BCs on Γ1-Γ4, natural on notch faces Γ5a/Γ5b
- KAN network (Gaussian-basis Kolmogorov-Arnold layers), not an MLP
- Weighted PDE residual near tip: w(x)=1/(dist_to_tip+eps)
- Adam + LR schedule + grad clipping + early stopping + validation
- Outputs: loss plot, Φ field heatmap, |∇Φ| line plot
- Diagnostics: PDE residual stats, symmetry, near/far gradient ratio, finite check

Run example:
  python StrainLimiting_KAN_PINN.py

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
from datetime import datetime
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
    adam_epochs: int = 8000
    finetune_epochs: int = 8000
    pretrain_epochs: int = 1000
    pde_ramp_epochs: int = 3500

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
    lambda_pde: float = 1.0
    lambda_tip: float = 0.02
    lambda_tip_ratio: float = 1.0

    tip_stress_c: float = 0.25
    tip_stress_eps: float = 1e-5
    tip_ratio_target: float = 1.2
    tip_strip_bias_power: float = 2.5
    tip_loss_r_weight_power: float = 0.5

    learning_rate: float = 3e-4
    finetune_lr: float = 5e-5

    print_every: int = 50
    validation_every: int = 10
    checkpoint_every: int = 50
    early_stop_patience: int = 99999
    min_improve: float = 1e-5
    max_grad_norm: float = 1.0

    # Best-model selection (physics-aware)
    model_select_start_epoch: int = 2750
    model_select_pde_weight_floor: float = 0.25

    # Singular weighting w=1/(dist_to_tip+eps)
    tip_weight_eps: float = 2e-3
    tip_weight_clip: float = 25.0
    grad_norm_eps: float = 1e-10
    initial_pde_weight: float = 5e-3
    notch_face_bc_mode: str = "natural"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.k1(x))
        h = torch.tanh(self.k2(h))
        h = torch.tanh(self.k3(h))
        out = self.k4(h)
        return out


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


def dirichlet_boundary_labels(trn: TrainParams) -> Tuple[str, ...]:
    mode = trn.notch_face_bc_mode.strip().lower()
    if mode == "dirichlet_zero":
        return ALL_BOUNDARY_LABELS
    if mode in ("natural", "exclude"):
        return OUTER_BOUNDARY_LABELS
    raise ValueError(
        f"Unsupported KAN_PINN_G5_MODE='{trn.notch_face_bc_mode}'. "
        "Use 'natural', 'exclude', or 'dirichlet_zero'."
    )


def boundary_roles(trn: TrainParams) -> Dict[str, str]:
    mode = trn.notch_face_bc_mode.strip().lower()
    roles = {label: "Dirichlet" for label in OUTER_BOUNDARY_LABELS}
    if mode == "dirichlet_zero":
        roles["G5a"] = "Dirichlet-zero (legacy)"
        roles["G5b"] = "Dirichlet-zero (legacy)"
    elif mode == "natural":
        roles["G5a"] = "Natural / traction-free"
        roles["G5b"] = "Natural / traction-free"
    elif mode == "exclude":
        roles["G5a"] = "Excluded from Dirichlet loss"
        roles["G5b"] = "Excluded from Dirichlet loss"
    else:
        raise ValueError(
            f"Unsupported KAN_PINN_G5_MODE='{trn.notch_face_bc_mode}'. "
            "Use 'natural', 'exclude', or 'dirichlet_zero'."
        )
    return roles


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
    x = xy[:, 0]
    if label == "G1":
        return torch.full_like(x, bc.sigma0 * bc.L)
    if label == "G2":
        return torch.zeros_like(x)
    if label == "G3":
        return -bc.sigma0 * (x - bc.L)
    if label == "G4":
        return bc.sigma0 * (bc.L - x)
    if label in NOTCH_FACE_LABELS and trn.notch_face_bc_mode.strip().lower() == "dirichlet_zero":
        return torch.zeros_like(x)
    raise ValueError(f"Dirichlet target requested for non-Dirichlet boundary '{label}'")


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
        return torch.mean((w * res) ** 2)

    total = torch.zeros((), dtype=torch.float32, device=interior_xy.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xy_chunk = interior_xy[s:e]
        res = pde_residual(model, xy_chunk, mat, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
        w = tip_residual_weights(xy_chunk, geo, trn)
        chunk_loss = torch.mean((w * res) ** 2)
        total = total + chunk_loss * (e - s)

    return total / n


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
            "pretrain_epochs": trn.pretrain_epochs,
            "pde_ramp_epochs": trn.pde_ramp_epochs,
            "lambda_bc": trn.lambda_bc,
            "lambda_pde": trn.lambda_pde,
            "lambda_tip": trn.lambda_tip,
            "lambda_tip_ratio": trn.lambda_tip_ratio,
            "tip_stress_c": trn.tip_stress_c,
            "tip_ratio_target": trn.tip_ratio_target,
            "initial_pde_weight": trn.initial_pde_weight,
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


def save_plots(
    model: nn.Module,
    mat: MaterialParams,
    bc: BCParams,
    loss_hist: list[float],
    pde_hist: list[float],
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

    # Loss history
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, lw=2, label="L total")
    plt.plot(pde_hist, lw=2, label="L_pde")
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
    total_epochs = trn.adam_epochs + trn.finetune_epochs

    val_interior, val_collocation_counts = sample_interior_points_val(geo, trn)
    val_tip_interior = filter_tip_strip_points(val_interior, geo, trn)
    val_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)

    val_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in val_bdata.items()}
    last_collocation_counts: Dict[str, int] = {}

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0

    loss_hist: list[float] = []
    pde_hist: list[float] = []
    bc_hist: list[float] = []
    tip_hist: list[float] = []
    tip_ratio_hist: list[float] = []
    val_hist: list[float] = []
    val_select_hist: list[float] = []

    ckpt_path = outdir / "best_checkpoint.pt"

    def save_checkpoint() -> None:
        last_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "model_state": last_state,
                "best_model_state": best_state,
                "last_model_state": last_state,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "loss_total": loss_hist,
                "loss_pde": pde_hist,
                "loss_bc": bc_hist,
                "loss_tip": tip_hist,
                "loss_tip_ratio": tip_ratio_hist,
                "loss_val": val_hist,
                "loss_val_select": val_select_hist,
                "completed_epochs": len(loss_hist),
                "boundary_roles": boundary_roles(trn),
                "g5_mode": trn.notch_face_bc_mode,
                "last_collocation_counts": last_collocation_counts,
                "val_collocation_counts": val_collocation_counts,
            },
            ckpt_path,
        )

    if resume and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        last_state_key = "last_model_state" if "last_model_state" in ckpt else "model_state"
        best_state_key = "best_model_state" if "best_model_state" in ckpt else last_state_key
        if last_state_key in ckpt:
            model.load_state_dict(ckpt[last_state_key])
            best_state = copy.deepcopy(ckpt[best_state_key]) if best_state_key in ckpt else copy.deepcopy(model.state_dict())
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_val = float(ckpt.get("best_val", float("inf")))
            loss_hist = list(ckpt.get("loss_total", []))
            pde_hist = list(ckpt.get("loss_pde", []))
            bc_hist = list(ckpt.get("loss_bc", []))
            tip_hist = list(ckpt.get("loss_tip", []))
            tip_ratio_hist = list(ckpt.get("loss_tip_ratio", []))
            val_hist = list(ckpt.get("loss_val", []))
            val_select_hist = list(ckpt.get("loss_val_select", ckpt.get("loss_val", [])))
            completed_epochs = int(ckpt.get("completed_epochs", len(loss_hist)))
            stale_epochs = 0
            print(
                f"Resuming from checkpoint: epoch {completed_epochs}/{total_epochs} | "
                f"best_epoch={best_epoch} best_val={best_val:.6f} (resume=last_state)"
            )

    t0 = time.time()

    if completed_epochs >= total_epochs:
        print(f"Checkpoint already reached target epochs ({completed_epochs}). Skipping training.")
        return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist, {
            "train_last": last_collocation_counts,
            "validation": val_collocation_counts,
        }

    # Stage 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=trn.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trn.lr_gamma_adam)

    adam_start = completed_epochs + 1
    if adam_start < 1:
        adam_start = 1

    for epoch in range(adam_start, trn.adam_epochs + 1):
        model.train()
        interior, collocation_counts = sample_interior_points(geo, trn)
        collocation_counts["adaptive"] = 0
        pde_weight = pde_curriculum_weight(epoch, trn)
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
                print(f"[adaptive sampling] RuntimeError encountered; skipping adaptive points this epoch. {exc}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bdata = sample_boundary_points(geo, trn.n_boundary_each)
        last_collocation_counts = dict(collocation_counts)

        bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}

        optimizer.zero_grad(set_to_none=True)
        bc_terms = boundary_loss_terms(model, bdata_t, bc, trn)
        lbc = torch.stack(list(bc_terms.values())).mean()
        lg = gauge_loss(model, device)
        lsym = symmetry_loss(model, geo, device)
        base_loss = trn.lambda_bc * lbc + trn.lambda_gauge * lg + trn.lambda_sym * lsym
        base_loss.backward()

        ltip_f = 0.0
        lratio_f = 0.0
        ratio_f = 0.0
        if trn.lambda_tip > 0.0:
            tip_interior = sample_tip_strip_points(geo, trn, trn.n_interior_tip_strip)
            ltip_f = streaming_tip_stress_backward(model, tip_interior, geo, trn, device)
        if trn.lambda_tip_ratio > 0.0:
            lratio_t, ratio_t = tip_stress_ratio_loss(model, geo, trn, device, create_graph=True)
            (trn.lambda_tip_ratio * lratio_t).backward()
            lratio_f = float(lratio_t.detach().cpu())
            ratio_f = float(ratio_t.detach().cpu())
        lpde_f = streaming_pde_backward(model, interior, mat, geo, trn, device, pde_weight)
        ltot_f = float(base_loss.detach().cpu()) + trn.lambda_tip * ltip_f + trn.lambda_tip_ratio * lratio_f + trn.lambda_pde * pde_weight * lpde_f

        if trn.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm)
        optimizer.step()
        scheduler.step()

        do_validate = (epoch == 1) or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
        if do_validate:
            model.eval()
            with torch.enable_grad():
                if pde_weight > 0.0 or trn.model_select_pde_weight_floor > 0.0:
                    v_lpde = streaming_pde_eval(model, val_interior, mat, geo, trn, device)
                else:
                    v_lpde = torch.zeros((), dtype=torch.float32, device=device)
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
                    + trn.lambda_tip * v_ltip
                    + trn.lambda_tip_ratio * v_lratio
                    + trn.lambda_bc * v_lbc
                    + trn.lambda_gauge * v_lg
                    + trn.lambda_sym * v_lsym
                )
                select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
                lval_select = (
                    trn.lambda_pde * select_wpde * v_lpde
                    + trn.lambda_tip * v_ltip
                    + trn.lambda_tip_ratio * v_lratio
                    + trn.lambda_bc * v_lbc
                    + trn.lambda_gauge * v_lg
                    + trn.lambda_sym * v_lsym
                )
            lval_f = float(lval.detach().cpu())
            lval_select_f = float(lval_select.detach().cpu())
        else:
            lval_f = val_hist[-1] if len(val_hist) > 0 else float("nan")
            lval_select_f = val_select_hist[-1] if len(val_select_hist) > 0 else float("nan")

        lbc_f = float(lbc.detach().cpu())

        loss_hist.append(ltot_f)
        pde_hist.append(lpde_f)
        bc_hist.append(lbc_f)
        tip_hist.append(ltip_f)
        tip_ratio_hist.append(lratio_f)
        val_hist.append(lval_f)
        val_select_hist.append(lval_select_f)

        if do_validate:
            if (not math.isfinite(best_val)) or (best_epoch == 0):
                best_val = lval_select_f
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint()
            elif epoch >= trn.model_select_start_epoch:
                if lval_select_f < best_val - trn.min_improve:
                    best_val = lval_select_f
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                    save_checkpoint()
                else:
                    stale_epochs += 1

        if trn.checkpoint_every > 0 and (epoch % trn.checkpoint_every == 0):
            save_checkpoint()

        if epoch == 1 or (epoch % trn.print_every == 0):
            elapsed = time.time() - t0
            sec_per_ep = elapsed / epoch
            eta = sec_per_ep * (total_epochs - epoch)
            val_tag = "val" if do_validate else "val(skip)"
            print(
                f"Epoch {epoch:5d}/{total_epochs} | "
                f"L={ltot_f:.5e} | Lpde={lpde_f:.5e} | Lbc={lbc_f:.5e} | "
                f"Lval={lval_f:.5e} | Lval(sel)={lval_select_f:.5e} ({val_tag}) | "
                f"Nint={collocation_counts['total']} (tip_strip={collocation_counts['tip_strip']}, "
                f"tip_annulus={collocation_counts['tip_annulus']}, adapt={collocation_counts['adaptive']}) | "
                f"wpde={pde_weight:.3f} | {sec_per_ep:.2f}s/ep | ETA {eta/60:.1f} min"
            )

        if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}, best epoch = {best_epoch}")
            model.load_state_dict(best_state)
            return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist, {
                "train_last": last_collocation_counts,
                "validation": val_collocation_counts,
            }

    # Stage 2: Fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=trn.finetune_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trn.lr_gamma_finetune)
    finetune_start = max(completed_epochs + 1, trn.adam_epochs + 1)
    if finetune_start <= total_epochs:
        print(f"Starting fine-tune stage with lower LR = {trn.finetune_lr}")

    for epoch in range(finetune_start, total_epochs + 1):
        model.train()

        interior, collocation_counts = sample_interior_points(geo, trn)
        collocation_counts["adaptive"] = 0
        pde_weight = pde_curriculum_weight(epoch, trn)
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
                print(f"[adaptive sampling] RuntimeError encountered; skipping adaptive points this epoch. {exc}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bdata = sample_boundary_points(geo, trn.n_boundary_each)
        last_collocation_counts = dict(collocation_counts)
        bdata_t = {kk: to_tensor(vv, device, requires_grad=False) for kk, vv in bdata.items()}

        optimizer.zero_grad(set_to_none=True)
        bc_terms = boundary_loss_terms(model, bdata_t, bc, trn)
        lbc = torch.stack(list(bc_terms.values())).mean()
        lg = gauge_loss(model, device)
        lsym = symmetry_loss(model, geo, device)
        base_loss = trn.lambda_bc * lbc + trn.lambda_gauge * lg + trn.lambda_sym * lsym
        base_loss.backward()

        ltip_f = 0.0
        lratio_f = 0.0
        ratio_f = 0.0
        if trn.lambda_tip > 0.0:
            tip_interior = sample_tip_strip_points(geo, trn, trn.n_interior_tip_strip)
            ltip_f = streaming_tip_stress_backward(model, tip_interior, geo, trn, device)
        if trn.lambda_tip_ratio > 0.0:
            lratio_t, ratio_t = tip_stress_ratio_loss(model, geo, trn, device, create_graph=True)
            (trn.lambda_tip_ratio * lratio_t).backward()
            lratio_f = float(lratio_t.detach().cpu())
            ratio_f = float(ratio_t.detach().cpu())
        lpde_f = streaming_pde_backward(model, interior, mat, geo, trn, device, pde_weight)
        ltot_f = float(base_loss.detach().cpu()) + trn.lambda_tip * ltip_f + trn.lambda_tip_ratio * lratio_f + trn.lambda_pde * pde_weight * lpde_f

        if trn.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm)
        optimizer.step()
        scheduler.step()

        do_validate = (epoch == 1) or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
        if do_validate:
            model.eval()
            with torch.enable_grad():
                if pde_weight > 0.0 or trn.model_select_pde_weight_floor > 0.0:
                    v_lpde = streaming_pde_eval(model, val_interior, mat, geo, trn, device)
                else:
                    v_lpde = torch.zeros((), dtype=torch.float32, device=device)
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
                    + trn.lambda_tip * v_ltip
                    + trn.lambda_tip_ratio * v_lratio
                    + trn.lambda_bc * v_lbc
                    + trn.lambda_gauge * v_lg
                    + trn.lambda_sym * v_lsym
                )
                select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
                lval_select = (
                    trn.lambda_pde * select_wpde * v_lpde
                    + trn.lambda_tip * v_ltip
                    + trn.lambda_tip_ratio * v_lratio
                    + trn.lambda_bc * v_lbc
                    + trn.lambda_gauge * v_lg
                    + trn.lambda_sym * v_lsym
                )
            lval_f = float(lval.detach().cpu())
            lval_select_f = float(lval_select.detach().cpu())
        else:
            lval_f = val_hist[-1] if len(val_hist) > 0 else float("nan")
            lval_select_f = val_select_hist[-1] if len(val_select_hist) > 0 else float("nan")

        lbc_f = float(lbc.detach().cpu())

        loss_hist.append(ltot_f)
        pde_hist.append(lpde_f)
        bc_hist.append(lbc_f)
        tip_hist.append(ltip_f)
        tip_ratio_hist.append(lratio_f)
        val_hist.append(lval_f)
        val_select_hist.append(lval_select_f)

        if do_validate:
            if (not math.isfinite(best_val)) or (best_epoch == 0):
                best_val = lval_select_f
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint()
            elif epoch >= trn.model_select_start_epoch:
                if lval_select_f < best_val - trn.min_improve:
                    best_val = lval_select_f
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                    save_checkpoint()
                else:
                    stale_epochs += 1

        if trn.checkpoint_every > 0 and (epoch % trn.checkpoint_every == 0):
            save_checkpoint()

        if epoch == 1 or (epoch % trn.print_every == 0):
            elapsed = time.time() - t0
            sec_per_ep = elapsed / epoch
            eta = sec_per_ep * (total_epochs - epoch)
            val_tag = "val" if do_validate else "val(skip)"
            print(
                f"Epoch {epoch:5d}/{total_epochs} | "
                f"L={ltot_f:.5e} | Lpde={lpde_f:.5e} | Lbc={lbc_f:.5e} | "
                f"Lval={lval_f:.5e} | Lval(sel)={lval_select_f:.5e} ({val_tag}) | "
                f"Nint={collocation_counts['total']} (tip_strip={collocation_counts['tip_strip']}, "
                f"tip_annulus={collocation_counts['tip_annulus']}, adapt={collocation_counts['adaptive']}) | "
                f"wpde={pde_weight:.3f} | {sec_per_ep:.2f}s/ep | ETA {eta/60:.1f} min"
            )

        if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}, best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Best validation epoch: {best_epoch} | Best validation loss: {best_val:.6f}")

    save_checkpoint()

    return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist, {
        "train_last": last_collocation_counts,
        "validation": val_collocation_counts,
    }


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
    default_pretrain_epochs = env_int("KAN_PINN_PRETRAIN_EPOCHS", 1000)
    default_pde_ramp_epochs = env_int("KAN_PINN_PDE_RAMP_EPOCHS", 3500)
    default_model_select_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)
    default_adaptive_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)

    trn = TrainParams(
        adam_epochs=env_int("KAN_PINN_ADAM_EPOCHS", env_int("KAN_PINN_EPOCHS", 8000)),
        finetune_epochs=env_int("KAN_PINN_FINETUNE_EPOCHS", 8000),
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
        lambda_pde=env_float("KAN_PINN_LAMBDA_PDE", 1.0),
        lambda_tip=env_float("KAN_PINN_LAMBDA_TIP", 0.02),
        lambda_tip_ratio=env_float("KAN_PINN_LAMBDA_TIP_RATIO", 1.0),
        learning_rate=env_float("KAN_PINN_LR", 3e-4),
        finetune_lr=env_float("KAN_PINN_FINETUNE_LR", 5e-5),
        print_every=env_int("KAN_PINN_PRINT_EVERY", 50),
        validation_every=env_int("KAN_PINN_VAL_EVERY", 10),
        checkpoint_every=env_int("KAN_PINN_CHECKPOINT_EVERY", 50),
        early_stop_patience=env_int("KAN_PINN_PATIENCE", 99999),
        min_improve=env_float("KAN_PINN_MIN_IMPROVE", 1e-5),
        max_grad_norm=env_float("KAN_PINN_MAX_GRAD_NORM", 1.0),
        model_select_start_epoch=env_int("KAN_PINN_MODEL_SELECT_START_EPOCH", default_model_select_start),
        model_select_pde_weight_floor=env_float("KAN_PINN_MODEL_SELECT_PDE_FLOOR", 0.25),
        train_pde_chunk_size=env_int("KAN_PINN_TRAIN_PDE_CHUNK", 256),
        val_pde_chunk_size=env_int("KAN_PINN_VAL_PDE_CHUNK", 256),
        tip_weight_eps=env_float("KAN_PINN_TIP_WEIGHT_EPS", 2e-3),
        tip_weight_clip=env_float("KAN_PINN_TIP_WEIGHT_CLIP", 25.0),
        grad_norm_eps=env_float("KAN_PINN_GRAD_NORM_EPS", 1e-10),
        initial_pde_weight=env_float("KAN_PINN_INITIAL_PDE_WEIGHT", 5e-3),
        notch_face_bc_mode=os.getenv("KAN_PINN_G5_MODE", "natural").strip(),
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

    run_name = os.getenv("KAN_PINN_RUN_NAME", "").strip()
    resume_training = os.getenv("KAN_PINN_RESUME", "0").strip().lower() in ("1", "true", "yes", "y")

    random.seed(trn.seed)
    np.random.seed(trn.seed)
    torch.manual_seed(trn.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting training (Eq. 40 interior + Dirichlet Table-3 BCs on Γ1-Γ4).")
    print(f"Device: {device}")
    print(f"Γ5 treatment: {boundary_roles(trn)['G5a']}")

    model = KANPINN(hidden=trn.hidden, n_basis=trn.n_basis).to(device)

    root_outdir = Path(__file__).resolve().parent / "results_strainlimiting_python"
    outdir, selected_run = get_run_outdir(root_outdir, run_name if run_name else None)
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")

    model, best_epoch, best_val, lhist, lpde_hist, lbc_hist, val_hist, collocation_counts = train_model(
        model, mat, geo, bc, trn, outdir, device, resume=resume_training
    )

    final_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)
    final_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in final_bdata.items()}
    final_boundary_diag = boundary_diagnostics(model, final_bdata_t, bc, mat, geo, trn)
    verification = run_cross_verification(model, mat, geo, trn, device, boundary_diag=final_boundary_diag)
    save_plots(model, mat, bc, lhist, lpde_hist, lbc_hist, val_hist, geo, trn, outdir, device, final_boundary_diag, collocation_counts, verification)

    print(f"Training complete. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
