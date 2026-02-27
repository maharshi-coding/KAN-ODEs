#!/usr/bin/env python3
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
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class MaterialParams:
    mu: float = 1.0
    beta: float = 5.0
    alpha: float = 2.0


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
    adam_epochs: int = 4000
    finetune_epochs: int = 6000
    pretrain_epochs: int = 1000
    pde_ramp_epochs: int = 2000

    n_interior_uniform: int = 64
    n_interior_refine: int = 128
    n_interior_tip_strip: int = 5000
    n_boundary_each: int = 48

    val_n_interior_uniform: int = 192
    val_n_interior_refine: int = 192
    val_n_interior_tip_strip: int = 8192
    val_n_boundary_each: int = 96

    lambda_bc: float = 300.0
    lambda_gauge: float = 1e-3
    lambda_sym: float = 10.0
    lambda_pde: float = 10.0
    lambda_tip: float = 1.0

    tip_stress_c: float = 1.0
    tip_stress_eps: float = 1e-5

    learning_rate: float = 1e-4
    finetune_lr: float = 1e-4

    print_every: int = 50
    validation_every: int = 10
    checkpoint_every: int = 25
    early_stop_patience: int = 300
    min_improve: float = 1e-4
    max_grad_norm: float = 1.0

    # Best-model selection (physics-aware)
    model_select_start_epoch: int = 1001
    model_select_pde_weight_floor: float = 1.0

    # Singular weighting w=1/(dist_to_tip+eps)
    tip_weight_eps: float = 2e-3

    # Sampling around tip strip
    tip_strip_half_height: float = 0.02
    tip_strip_length: float = 0.12

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
    adaptive_start_epoch: int = 200

    # Reproducibility
    seed: int = 42

    # Model shape
    hidden: int = 96
    n_basis: int = 48


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


def sample_interior_points(geo: GeometryParams, trn: TrainParams) -> np.ndarray:
    uniform_pts = sample_points_excluding_notch(geo, trn.n_interior_uniform)

    x0, y0 = geo.tip
    hr = geo.refine_half_width
    refine_pts = sample_points_excluding_notch(
        geo,
        trn.n_interior_refine,
        xlo=max(geo.xmin, x0 - hr),
        xhi=min(geo.xmax, x0 + hr),
        ylo=max(geo.ymin, y0 - hr),
        yhi=min(geo.ymax, y0 + hr),
    )

    tip_pts = sample_tip_strip_points(geo, trn, trn.n_interior_tip_strip)

    return np.vstack([uniform_pts, refine_pts, tip_pts]).astype(np.float32)


def sample_interior_points_val(geo: GeometryParams, trn: TrainParams) -> np.ndarray:
    original = (
        trn.n_interior_uniform,
        trn.n_interior_refine,
        trn.n_interior_tip_strip,
    )
    trn.n_interior_uniform = trn.val_n_interior_uniform
    trn.n_interior_refine = trn.val_n_interior_refine
    trn.n_interior_tip_strip = trn.val_n_interior_tip_strip
    pts = sample_interior_points(geo, trn)
    trn.n_interior_uniform, trn.n_interior_refine, trn.n_interior_tip_strip = original
    return pts


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
    while k < n:
        x = xlo + (xhi - xlo) * random.random()
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


def phi_scalar(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    # xy: [N,2], returns [N]
    return model(xy).squeeze(-1)


def compute_stress(
    model: nn.Module,
    xy: torch.Tensor,
    create_graph: bool = True,
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
    tau_eq = torch.sqrt(tau_xz.pow(2) + tau_yz.pow(2) + 1e-12)
    return tau_xz, tau_yz, tau_eq


def pde_residual(
    model: nn.Module,
    xy: torch.Tensor,
    mat: MaterialParams,
    create_graph: bool = True,
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

    gnorm = torch.sqrt((grad_phi ** 2).sum(dim=1) + 1e-12)
    denom = 2.0 * mat.mu * torch.pow(1.0 + mat.beta * torch.pow(gnorm, mat.alpha), 1.0 / mat.alpha)
    q = grad_phi / denom.unsqueeze(1)

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


def dirichlet_target(label: str, xy: torch.Tensor, bc: BCParams) -> torch.Tensor:
    x = xy[:, 0]
    if label == "G1":
        return torch.full_like(x, bc.sigma0 * bc.L)
    if label == "G2":
        return torch.zeros_like(x)
    if label == "G3":
        return -bc.sigma0 * (x - bc.L)
    if label == "G4":
        return bc.sigma0 * (bc.L - x)
    return torch.zeros_like(x)


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
        res = pde_residual(model, interior_xy, mat, create_graph=create_graph)
        x0, y0 = geo.tip
        dist = torch.sqrt((interior_xy[:, 0] - x0) ** 2 + (interior_xy[:, 1] - y0) ** 2 + 1e-12)
        w = 1.0 / (dist**1.5 + trn.tip_weight_eps)
        return torch.mean(torch.log1p((w * res) ** 2))

    x0, y0 = geo.tip
    total = torch.zeros((), dtype=torch.float32, device=interior_xy.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xy_chunk = interior_xy[s:e]
        res = pde_residual(model, xy_chunk, mat, create_graph=create_graph)
        dist = torch.sqrt((xy_chunk[:, 0] - x0) ** 2 + (xy_chunk[:, 1] - y0) ** 2 + 1e-12)
        w = 1.0 / (dist**1.5 + trn.tip_weight_eps)
        chunk_loss = torch.mean(torch.log1p((w * res) ** 2))
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

    _, _, tau_eq = compute_stress(model, tip_xy, create_graph=create_graph)
    x0, y0 = geo.tip
    r = torch.sqrt((tip_xy[:, 0] - x0) ** 2 + (tip_xy[:, 1] - y0) ** 2 + 1e-12)
    singular_scaled = tau_eq * torch.sqrt(r + trn.tip_stress_eps)
    return torch.mean((singular_scaled - trn.tip_stress_c) ** 2)


def boundary_loss(model: nn.Module, bdata_t: Dict[str, torch.Tensor], bc: BCParams) -> torch.Tensor:
    losses = []
    for label in ("G1", "G2", "G3", "G4"):
        xy = bdata_t[label]
        pred = phi_scalar(model, xy)
        tgt = dirichlet_target(label, xy, bc)
        losses.append(torch.mean((pred - tgt) ** 2))
    return torch.stack(losses).mean()


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
    lbc = boundary_loss(model, bdata_t, bc)
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
    return min(1.0, phase2_epoch / max(1, trn.pde_ramp_epochs))


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
    return torch.sqrt((g ** 2).sum(dim=1) + 1e-12)


def residual_statistics(model: nn.Module, mat: MaterialParams, geo: GeometryParams, device: torch.device, n: int = 512):
    pts = sample_points_excluding_notch(geo, n)
    xy = to_tensor(pts, device, requires_grad=True)
    r = pde_residual(model, xy, mat).detach().cpu().numpy()
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


def tip_gradient_indicator(model: nn.Module, geo: GeometryParams, device: torch.device):
    x0, y0 = geo.tip

    xnear = np.linspace(max(geo.xmin, x0 - 0.06), x0 - 0.005, 80, dtype=np.float32)
    xfar = np.linspace(geo.xmin, max(geo.xmin, x0 - 0.20), 80, dtype=np.float32)
    ynear = np.full_like(xnear, y0)
    yfar = np.full_like(xfar, y0)

    near_t = to_tensor(np.stack([xnear, ynear], axis=1), device, requires_grad=True)
    far_t = to_tensor(np.stack([xfar, yfar], axis=1), device, requires_grad=True)

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


def run_cross_verification(model: nn.Module, mat: MaterialParams, geo: GeometryParams, trn: TrainParams, device: torch.device):
    rstats = residual_statistics(model, mat, geo, device)
    sstats = symmetry_error(model, geo, device)
    tipstats = tip_gradient_indicator(model, geo, device)
    gstats = grid_finite_check(model, geo, device)

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


# -----------------------------
# Plotting
# -----------------------------

def save_plots(
    model: nn.Module,
    loss_hist: list[float],
    pde_hist: list[float],
    bc_hist: list[float],
    val_hist: list[float],
    geo: GeometryParams,
    outdir: Path,
    device: torch.device,
):
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

    # Phi field
    xs, ys, phi = field_on_grid(model, geo, device, nx=181, ny=181)
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

    # tau_eq approaching tip along y=y0
    x0, y0 = geo.tip
    xline = np.linspace(geo.xmin, x0, 300, dtype=np.float32)
    yline = np.full_like(xline, y0)
    xy = to_tensor(np.stack([xline, yline], axis=1), device, requires_grad=True)
    _, _, tau_eq_line = compute_stress(model, xy, create_graph=False)
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

    val_interior = sample_interior_points_val(geo, trn)
    val_tip_interior = filter_tip_strip_points(val_interior, geo, trn)
    val_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)

    val_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in val_bdata.items()}

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0

    loss_hist: list[float] = []
    pde_hist: list[float] = []
    bc_hist: list[float] = []
    tip_hist: list[float] = []
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
                "loss_val": val_hist,
                "loss_val_select": val_select_hist,
                "completed_epochs": len(loss_hist),
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
        return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist

    # Stage 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=trn.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trn.lr_gamma_adam)

    adam_start = completed_epochs + 1
    if adam_start < 1:
        adam_start = 1

    for epoch in range(adam_start, trn.adam_epochs + 1):
        model.train()
        interior = sample_interior_points(geo, trn)
        if trn.adaptive_sampling and epoch >= trn.adaptive_start_epoch:
            try:
                n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                if n_adapt > 0:
                    adapt_pts = adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
                    if adapt_pts.size > 0:
                        interior = np.vstack([interior, adapt_pts]).astype(np.float32)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg:
                    raise
                print("[adaptive sampling] OOM encountered; skipping adaptive points this epoch.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bdata = sample_boundary_points(geo, trn.n_boundary_each)

        bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
        pde_weight = pde_curriculum_weight(epoch, trn)

        optimizer.zero_grad(set_to_none=True)
        lbc = boundary_loss(model, bdata_t, bc)
        lg = gauge_loss(model, device)
        lsym = symmetry_loss(model, geo, device)
        base_loss = trn.lambda_bc * lbc + trn.lambda_gauge * lg + trn.lambda_sym * lsym
        base_loss.backward()

        tip_interior = filter_tip_strip_points(interior, geo, trn)
        ltip_f = streaming_tip_stress_backward(model, tip_interior, geo, trn, device)
        lpde_f = streaming_pde_backward(model, interior, mat, geo, trn, device, pde_weight)
        ltot_f = float(base_loss.detach().cpu()) + trn.lambda_tip * ltip_f + trn.lambda_pde * pde_weight * lpde_f

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
                v_ltip = streaming_tip_stress_eval(model, val_tip_interior, geo, trn, device)
                v_lbc = boundary_loss(model, val_bdata_t, bc)
                v_lg = gauge_loss(model, device)
                v_lsym = symmetry_loss(model, geo, device)
                lval = trn.lambda_pde * pde_weight * v_lpde + trn.lambda_tip * v_ltip + trn.lambda_bc * v_lbc + trn.lambda_gauge * v_lg + trn.lambda_sym * v_lsym
                select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
                lval_select = trn.lambda_pde * select_wpde * v_lpde + trn.lambda_tip * v_ltip + trn.lambda_bc * v_lbc + trn.lambda_gauge * v_lg + trn.lambda_sym * v_lsym
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
        val_hist.append(lval_f)
        val_select_hist.append(lval_select_f)

        if do_validate:
            if epoch >= trn.model_select_start_epoch:
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
                f"L={ltot_f:.5e} | Lpde={lpde_f:.5e} | Ltip={ltip_f:.5e} | Lbc={lbc_f:.5e} | "
                f"Lval={lval_f:.5e} | Lval(sel)={lval_select_f:.5e} ({val_tag}) | "
                f"wpde={pde_weight:.3f} | {sec_per_ep:.2f}s/ep | ETA {eta/60:.1f} min"
            )

        if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}, best epoch = {best_epoch}")
            model.load_state_dict(best_state)
            return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist

    # Stage 2: Fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=trn.finetune_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trn.lr_gamma_finetune)
    finetune_start = max(completed_epochs + 1, trn.adam_epochs + 1)
    if finetune_start <= total_epochs:
        print(f"Starting fine-tune stage with lower LR = {trn.finetune_lr}")

    for epoch in range(finetune_start, total_epochs + 1):
        model.train()

        interior = sample_interior_points(geo, trn)
        if trn.adaptive_sampling and epoch >= trn.adaptive_start_epoch:
            try:
                n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                if n_adapt > 0:
                    adapt_pts = adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
                    if adapt_pts.size > 0:
                        interior = np.vstack([interior, adapt_pts]).astype(np.float32)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg:
                    raise
                print("[adaptive sampling] OOM encountered; skipping adaptive points this epoch.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bdata = sample_boundary_points(geo, trn.n_boundary_each)
        bdata_t = {kk: to_tensor(vv, device, requires_grad=False) for kk, vv in bdata.items()}
        pde_weight = pde_curriculum_weight(epoch, trn)

        optimizer.zero_grad(set_to_none=True)
        lbc = boundary_loss(model, bdata_t, bc)
        lg = gauge_loss(model, device)
        lsym = symmetry_loss(model, geo, device)
        base_loss = trn.lambda_bc * lbc + trn.lambda_gauge * lg + trn.lambda_sym * lsym
        base_loss.backward()

        tip_interior = filter_tip_strip_points(interior, geo, trn)
        ltip_f = streaming_tip_stress_backward(model, tip_interior, geo, trn, device)
        lpde_f = streaming_pde_backward(model, interior, mat, geo, trn, device, pde_weight)
        ltot_f = float(base_loss.detach().cpu()) + trn.lambda_tip * ltip_f + trn.lambda_pde * pde_weight * lpde_f

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
                v_ltip = streaming_tip_stress_eval(model, val_tip_interior, geo, trn, device)
                v_lbc = boundary_loss(model, val_bdata_t, bc)
                v_lg = gauge_loss(model, device)
                v_lsym = symmetry_loss(model, geo, device)
                lval = trn.lambda_pde * pde_weight * v_lpde + trn.lambda_tip * v_ltip + trn.lambda_bc * v_lbc + trn.lambda_gauge * v_lg + trn.lambda_sym * v_lsym
                select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
                lval_select = trn.lambda_pde * select_wpde * v_lpde + trn.lambda_tip * v_ltip + trn.lambda_bc * v_lbc + trn.lambda_gauge * v_lg + trn.lambda_sym * v_lsym
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
        val_hist.append(lval_f)
        val_select_hist.append(lval_select_f)

        if do_validate:
            if epoch >= trn.model_select_start_epoch:
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
                f"L={ltot_f:.5e} | Lpde={lpde_f:.5e} | Ltip={ltip_f:.5e} | Lbc={lbc_f:.5e} | "
                f"Lval={lval_f:.5e} | Lval(sel)={lval_select_f:.5e} ({val_tag}) | "
                f"wpde={pde_weight:.3f} | {sec_per_ep:.2f}s/ep | ETA {eta/60:.1f} min"
            )

        if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}, best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Best validation epoch: {best_epoch} | Best validation loss: {best_val:.6f}")

    save_checkpoint()

    return model, best_epoch, best_val, loss_hist, pde_hist, bc_hist, val_hist


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
    trn = TrainParams(
        adam_epochs=env_int("KAN_PINN_ADAM_EPOCHS", env_int("KAN_PINN_EPOCHS", 4000)),
        finetune_epochs=env_int("KAN_PINN_FINETUNE_EPOCHS", 6000),
        pretrain_epochs=env_int("KAN_PINN_PRETRAIN_EPOCHS", 1000),
        pde_ramp_epochs=env_int("KAN_PINN_PDE_RAMP_EPOCHS", 2000),
        n_interior_uniform=env_int("KAN_PINN_NU", 64),
        n_interior_refine=env_int("KAN_PINN_NR", 128),
        n_interior_tip_strip=env_int("KAN_PINN_NTIP", 5000),
        n_boundary_each=env_int("KAN_PINN_NB", 48),
        val_n_interior_uniform=env_int("KAN_PINN_VAL_NU", 192),
        val_n_interior_refine=env_int("KAN_PINN_VAL_NR", 192),
        val_n_interior_tip_strip=env_int("KAN_PINN_VAL_NTIP", 8192),
        val_n_boundary_each=env_int("KAN_PINN_VAL_NB", 96),
        lambda_bc=env_float("KAN_PINN_LAMBDA_BC", 300.0),
        lambda_sym=env_float("KAN_PINN_LAMBDA_SYM", 10.0),
        lambda_pde=env_float("KAN_PINN_LAMBDA_PDE", 10.0),
        lambda_tip=env_float("KAN_PINN_LAMBDA_TIP", 1.0),
        learning_rate=env_float("KAN_PINN_LR", 1e-4),
        finetune_lr=env_float("KAN_PINN_FINETUNE_LR", 1e-4),
        print_every=env_int("KAN_PINN_PRINT_EVERY", 50),
        validation_every=env_int("KAN_PINN_VAL_EVERY", 10),
        checkpoint_every=env_int("KAN_PINN_CHECKPOINT_EVERY", 25),
        early_stop_patience=env_int("KAN_PINN_PATIENCE", 300),
        min_improve=env_float("KAN_PINN_MIN_IMPROVE", 1e-4),
        max_grad_norm=env_float("KAN_PINN_MAX_GRAD_NORM", 1.0),
        model_select_start_epoch=env_int("KAN_PINN_MODEL_SELECT_START_EPOCH", env_int("KAN_PINN_PRETRAIN_EPOCHS", 1000) + 1),
        model_select_pde_weight_floor=env_float("KAN_PINN_MODEL_SELECT_PDE_FLOOR", 1.0),
        train_pde_chunk_size=env_int("KAN_PINN_TRAIN_PDE_CHUNK", 256),
        val_pde_chunk_size=env_int("KAN_PINN_VAL_PDE_CHUNK", 256),
        tip_strip_half_height=env_float("KAN_PINN_TIP_STRIP_HH", 0.02),
        tip_strip_length=env_float("KAN_PINN_TIP_STRIP_LEN", 0.12),
        tip_stress_c=env_float("KAN_PINN_TIP_STRESS_C", 1.0),
        tip_stress_eps=env_float("KAN_PINN_TIP_STRESS_EPS", 1e-5),
        adaptive_sampling=env_bool("KAN_PINN_ADAPTIVE_SAMPLING", False),
        adaptive_candidates=env_int("KAN_PINN_ADAPTIVE_CANDIDATES", 4096),
        adaptive_topk=env_int("KAN_PINN_ADAPTIVE_TOPK", 512),
        adaptive_start_epoch=env_int("KAN_PINN_ADAPTIVE_START_EPOCH", 200),
        seed=env_int("KAN_PINN_SEED", 42),
    )

    mat = MaterialParams(
        mu=env_float("KAN_PINN_MU", 1.0),
        beta=env_float("KAN_PINN_BETA", 5.0),
        alpha=env_float("KAN_PINN_ALPHA", 2.0),
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

    print("Starting training (Eq. 40 interior + Dirichlet Table-3 BCs on Γ1-Γ4; natural on Γ5a/Γ5b).")
    print(f"Device: {device}")

    model = KANPINN(hidden=trn.hidden, n_basis=trn.n_basis).to(device)

    root_outdir = Path(__file__).resolve().parent / "results_strainlimiting_python"
    outdir, selected_run = get_run_outdir(root_outdir, run_name if run_name else None)
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")

    model, best_epoch, best_val, lhist, lpde_hist, lbc_hist, val_hist = train_model(
        model, mat, geo, bc, trn, outdir, device, resume=resume_training
    )

    run_cross_verification(model, mat, geo, trn, device)
    save_plots(model, lhist, lpde_hist, lbc_hist, val_hist, geo, outdir, device)

    print(f"Training complete. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
