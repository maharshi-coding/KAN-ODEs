#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Standalone no-crack full-square KAN-PINN for the strain-limiting PDE:

    -div( grad(Phi) / (2*mu*(1 + beta*|grad(Phi)|^alpha)^(1/alpha)) ) = 0

on Omega = [0, 1] x [0, 1].

Only the four outer Dirichlet boundaries are used:
    Gamma1: x = 0, Phi = sigma0 * L
    Gamma2: x = 1, Phi = 0
    Gamma3: y = 0, Phi = -sigma0 * (x - L)
    Gamma4: y = 1, Phi = sigma0 * (L - x)

This script intentionally contains no crack, no notch, no Gamma5 boundary, no
notch void exclusion, no tip-focused sampling, and no crack-tip residual
weighting. It keeps the Gaussian-basis KAN model, PyTorch autograd PDE
residual, Adam training, optional L-BFGS polishing, validation, plots,
checkpoints, and saved diagnostics.

Quick smoke test:
    KAN_PINN_RUN_NAME=quick_no_crack KAN_PINN_EPOCHS=2 KAN_PINN_FINETUNE_EPOCHS=0 \
    KAN_PINN_LBFGS_EPOCHS=0 KAN_PINN_NU=16 KAN_PINN_NB=8 KAN_PINN_VAL_NU=16 \
    KAN_PINN_VAL_NB=8 KAN_PINN_HIDDEN=8 KAN_PINN_N_BASIS=8 \
    python StrainLimiting_NoCrack_KAN_PINN.py
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
    "G1": "Gamma1",
    "G2": "Gamma2",
    "G3": "Gamma3",
    "G4": "Gamma4",
}
BOUNDARY_LABELS = ("G1", "G2", "G3", "G4")


# -----------------------------
# Configuration
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


@dataclass
class BCParams:
    sigma0: float = 1.0
    L: float = 1.0


@dataclass
class TrainParams:
    adam_epochs: int = 10000
    finetune_epochs: int = 5000
    pretrain_epochs: int = 1500
    pde_ramp_epochs: int = 5000

    n_interior_uniform: int = 1536
    n_boundary_each: int = 128
    val_n_interior_uniform: int = 1536
    val_n_boundary_each: int = 128

    lambda_bc: float = 10.0
    lambda_pde: float = 0.25
    lambda_energy: float = 1.0

    learning_rate: float = 1e-4
    finetune_lr: float = 2e-5

    print_every: int = 10
    validation_every: int = 10
    checkpoint_every: int = 50
    detailed_diag_every: int = 100
    early_stop_patience: int = 99999
    min_improve: float = 1e-5
    max_grad_norm: float = 1.0
    diagnostics_samples: int = 512

    model_select_start_epoch: int = 2750
    model_select_pde_weight_floor: float = 0.25

    grad_norm_eps: float = 1e-10
    initial_pde_weight: float = 1e-6
    pde_loss_mode: str = "pseudo_huber"
    pde_residual_delta: float = 25.0
    pde_mse_blend: float = 0.02

    hard_bc_mode: str = "distance_outer"
    hard_bc_eps: float = 1e-5
    hard_bc_distance_scale: float = 0.08
    hard_bc_distance_power: float = 2.0

    lr_gamma_adam: float = 0.9998
    lr_gamma_finetune: float = 0.9999

    lbfgs_epochs: int = 250
    lbfgs_lr: float = 0.5
    lbfgs_history_size: int = 25
    lbfgs_max_iter: int = 1
    lbfgs_n_uniform: int = 256
    lbfgs_n_boundary_each: int = 96

    train_pde_chunk_size: int = 256
    val_pde_chunk_size: int = 256

    adaptive_sampling: bool = True
    adaptive_candidates: int = 4096
    adaptive_topk: int = 768
    adaptive_start_epoch: int = 2750
    adaptive_refresh_every: int = 25

    pointwise_nx: int = 181
    pointwise_ny: int = 181
    pointwise_boundary_each: int = 256
    pointwise_batch_size: int = 512

    seed: int = 42
    hidden: int = 96
    n_basis: int = 48


# -----------------------------
# KAN model
# -----------------------------

class KANLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_basis: int,
        scale: float = 0.1,
        center_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis

        self.coeff = nn.Parameter(scale * torch.randn(out_dim, in_dim, n_basis))
        self.lin = nn.Parameter(scale * torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        centers = torch.linspace(float(center_range[0]), float(center_range[1]), n_basis)
        self.centers = nn.Parameter(centers)
        self.logwidth = nn.Parameter(torch.full((n_basis,), math.log(0.15)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        widths = torch.exp(self.logwidth) + 1e-5
        lin_part = x @ self.lin.t()
        z = (x.unsqueeze(-1) - self.centers.view(1, 1, -1)) / widths.view(1, 1, -1)
        basis = torch.exp(-(z ** 2))
        basis_part = torch.einsum("nib,oib->no", basis, self.coeff)
        return lin_part + basis_part + self.bias.view(1, -1)


class KANPINN(nn.Module):
    def __init__(self, hidden: int = 96, n_basis: int = 48):
        super().__init__()
        self.k1 = KANLayer(2, hidden, n_basis, center_range=(0.0, 1.0))
        self.k2 = KANLayer(hidden, hidden, n_basis, center_range=(-1.0, 1.0))
        self.k3 = KANLayer(hidden, hidden, n_basis, center_range=(-1.0, 1.0))
        self.k4 = KANLayer(hidden, 1, n_basis, center_range=(-1.0, 1.0))
        self.hard_bc_mode = "none"
        self.hard_bc_eps = 1e-12
        self.hard_bc_distance_scale = 0.08
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
        return self.k4(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_forward(x).squeeze(-1)
        if self.hard_bc_mode == "none":
            return raw.unsqueeze(-1)
        if self.geo is None or self.bc is None:
            raise RuntimeError("Hard boundary ansatz requested before configure_boundary_ansatz(...).")
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
# Validation and sampling
# -----------------------------

def canonical_hard_bc_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases = {
        "distance_outer": "distance_outer",
        "distance": "distance_outer",
        "outer": "distance_outer",
        "hard": "distance_outer",
        "none": "none",
        "off": "none",
        "soft": "none",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported KAN_PINN_HARD_BC_MODE={mode!r}; allowed: {allowed}")
    return aliases[normalized]


def canonical_pde_loss_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"mse", "l2"}:
        return "mse"
    if normalized in {"pseudo_huber", "huber", "robust"}:
        return "pseudo_huber"
    raise ValueError(f"Unsupported KAN_PINN_PDE_LOSS_MODE={mode!r}; use mse or pseudo_huber.")


def validate_configuration(mat: MaterialParams, geo: GeometryParams, trn: TrainParams) -> None:
    if mat.mu <= 0.0:
        raise ValueError("mu must be positive.")
    if mat.beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if mat.alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if not (geo.xmin < geo.xmax and geo.ymin < geo.ymax):
        raise ValueError("Invalid square bounds.")
    if trn.n_interior_uniform <= 0 or trn.val_n_interior_uniform <= 0:
        raise ValueError("Interior sample counts must be positive.")
    if trn.n_boundary_each <= 0 or trn.val_n_boundary_each <= 0:
        raise ValueError("Boundary sample counts must be positive.")
    canonical_hard_bc_mode(trn.hard_bc_mode)
    canonical_pde_loss_mode(trn.pde_loss_mode)


def sample_interior_points(geo: GeometryParams, n: int) -> np.ndarray:
    x = geo.xmin + (geo.xmax - geo.xmin) * np.random.rand(n).astype(np.float32)
    y = geo.ymin + (geo.ymax - geo.ymin) * np.random.rand(n).astype(np.float32)
    return np.stack([x, y], axis=1).astype(np.float32)


def adaptive_residual_points(
    model: nn.Module,
    geo: GeometryParams,
    mat: MaterialParams,
    trn: TrainParams,
    device: torch.device,
    n_select: int,
) -> np.ndarray:
    n_candidates = max(n_select, int(trn.adaptive_candidates))
    candidates = sample_interior_points(geo, n_candidates)
    scores: List[np.ndarray] = []
    model.eval()
    for s in range(0, candidates.shape[0], trn.val_pde_chunk_size):
        e = min(s + trn.val_pde_chunk_size, candidates.shape[0])
        xy = to_tensor(candidates[s:e], device, requires_grad=True)
        res = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        scores.append(torch.abs(res).detach().cpu().numpy())
        del xy, res
    score_np = np.concatenate(scores, axis=0)
    topk = min(n_select, candidates.shape[0])
    idx = np.argpartition(score_np, -topk)[-topk:]
    return candidates[idx].astype(np.float32)


def sample_boundary_points(geo: GeometryParams, n_each: int) -> Dict[str, np.ndarray]:
    y1 = geo.ymin + (geo.ymax - geo.ymin) * np.random.rand(n_each).astype(np.float32)
    g1 = np.stack([np.full(n_each, geo.xmin, dtype=np.float32), y1], axis=1)

    y2 = geo.ymin + (geo.ymax - geo.ymin) * np.random.rand(n_each).astype(np.float32)
    g2 = np.stack([np.full(n_each, geo.xmax, dtype=np.float32), y2], axis=1)

    x3 = geo.xmin + (geo.xmax - geo.xmin) * np.random.rand(n_each).astype(np.float32)
    g3 = np.stack([x3, np.full(n_each, geo.ymin, dtype=np.float32)], axis=1)

    x4 = geo.xmin + (geo.xmax - geo.xmin) * np.random.rand(n_each).astype(np.float32)
    g4 = np.stack([x4, np.full(n_each, geo.ymax, dtype=np.float32)], axis=1)

    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


# -----------------------------
# PDE and losses
# -----------------------------

def to_tensor(x: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(x, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def safe_l2_norm(vec: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(vec ** 2, dim=1) + eps)


def phi_scalar(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    return model(xy).squeeze(-1)


def flux_from_grad(grad_phi: torch.Tensor, mat: MaterialParams, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    gnorm = safe_l2_norm(grad_phi, eps)
    denom = 2.0 * mat.mu * torch.pow(1.0 + mat.beta * torch.pow(gnorm, mat.alpha), 1.0 / mat.alpha)
    flux = grad_phi / denom.unsqueeze(1)
    return flux, gnorm


def boundary_distances(xy: torch.Tensor, geo: GeometryParams) -> Dict[str, torch.Tensor]:
    x = xy[:, 0]
    y = xy[:, 1]
    return {
        "G1": torch.clamp(x - geo.xmin, min=0.0),
        "G2": torch.clamp(geo.xmax - x, min=0.0),
        "G3": torch.clamp(y - geo.ymin, min=0.0),
        "G4": torch.clamp(geo.ymax - y, min=0.0),
    }


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
    raise ValueError(f"Unknown boundary label: {label}")


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
    mode = canonical_hard_bc_mode(mode)
    if mode == "none":
        return raw_phi

    distances = boundary_distances(xy, geo)
    d_stack = torch.stack([distances[label] for label in BOUNDARY_LABELS], dim=1)
    d_pos = torch.clamp(d_stack, min=0.0)

    weights = 1.0 / torch.pow(d_pos + eps, distance_power)
    targets = torch.stack([dirichlet_target_values(label, xy, bc) for label in BOUNDARY_LABELS], dim=1)
    extension = torch.sum(weights * targets, dim=1) / torch.sum(weights, dim=1).clamp_min(eps)

    inv_nearest = torch.sum(1.0 / (d_pos + eps), dim=1)
    nearest = 1.0 / inv_nearest.clamp_min(eps)
    vanish = nearest / (nearest + distance_scale)
    soft_phi = extension + vanish * raw_phi

    on_boundary = d_pos <= eps
    boundary_count = torch.sum(on_boundary.to(dtype=xy.dtype), dim=1)
    boundary_target = torch.sum(on_boundary.to(dtype=xy.dtype) * targets, dim=1) / boundary_count.clamp_min(1.0)
    return torch.where(boundary_count > 0.0, boundary_target, soft_phi)


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
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    phi = phi_scalar(model, xy)
    grad_phi = torch.autograd.grad(
        phi,
        xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]
    flux, _ = flux_from_grad(grad_phi, mat, grad_norm_eps)

    def partial_derivative(values: torch.Tensor, coord: int, retain_graph: bool) -> torch.Tensor:
        if not values.requires_grad:
            return torch.zeros_like(xy[:, coord])
        grad = torch.autograd.grad(
            values,
            xy,
            grad_outputs=torch.ones_like(values),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros_like(xy[:, coord])
        return grad[:, coord]

    dqx_dx = partial_derivative(flux[:, 0], 0, retain_graph=True)
    dqy_dy = partial_derivative(flux[:, 1], 1, retain_graph=create_graph)
    return -(dqx_dx + dqy_dy)


def pde_residual_objective(residual: torch.Tensor, trn: TrainParams) -> torch.Tensor:
    mode = canonical_pde_loss_mode(trn.pde_loss_mode)
    if mode == "mse":
        return residual ** 2
    delta = torch.as_tensor(trn.pde_residual_delta, dtype=residual.dtype, device=residual.device)
    scaled = residual / delta
    objective = 2.0 * delta * delta * (torch.sqrt(1.0 + scaled * scaled) - 1.0)
    if trn.pde_mse_blend > 0.0:
        objective = objective + trn.pde_mse_blend * residual ** 2
    return objective


def pde_loss(
    model: nn.Module,
    interior_xy: torch.Tensor,
    mat: MaterialParams,
    trn: TrainParams,
    create_graph: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    n = interior_xy.shape[0]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        res = pde_residual(model, interior_xy, mat, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
        return torch.mean(pde_residual_objective(res, trn))

    total = torch.zeros((), dtype=torch.float32, device=interior_xy.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        xy_chunk = interior_xy[s:e]
        res = pde_residual(model, xy_chunk, mat, create_graph=create_graph, grad_norm_eps=trn.grad_norm_eps)
        total = total + torch.mean(pde_residual_objective(res, trn)) * (e - s)
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
    return torch.mean(strain_limiting_energy_density(grad_phi, mat, trn.grad_norm_eps))


def boundary_loss_terms(model: nn.Module, bdata_t: Dict[str, torch.Tensor], bc: BCParams) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    for label in BOUNDARY_LABELS:
        xy = bdata_t[label]
        pred = phi_scalar(model, xy)
        tgt = dirichlet_target_values(label, xy, bc)
        losses[label] = torch.mean((pred - tgt) ** 2)
    return losses


def boundary_loss(model: nn.Module, bdata_t: Dict[str, torch.Tensor], bc: BCParams) -> torch.Tensor:
    losses = boundary_loss_terms(model, bdata_t, bc)
    return torch.stack(list(losses.values())).mean()


def streaming_pde_backward(
    model: nn.Module,
    interior_np: np.ndarray,
    mat: MaterialParams,
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
            lpde = pde_loss(model, xy_chunk, mat, trn, create_graph=True, chunk_size=None)
            frac = (e - s) / n_total
            (trn.lambda_pde * pde_weight * frac * lpde).backward()
            weighted_mean += float(lpde.detach().cpu()) * (e - s)
            del xy_chunk, lpde
            s = e
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at PDE chunk size 1.")
            new_chunk = max(1, chunk // 2)
            print(f"[OOM fallback] Reducing train PDE chunk size: {chunk} -> {new_chunk}")
            chunk = new_chunk
    return weighted_mean / n_total


def streaming_pde_eval(
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
            lpde = pde_loss(model, xy_chunk, mat, trn, create_graph=False, chunk_size=None)
            total = total + lpde * (e - s)
            del xy_chunk, lpde
            s = e
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
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
            lenergy = energy_loss(model, xy_chunk, mat, trn, create_graph=True)
            frac = (e - s) / n_total
            (trn.lambda_energy * energy_weight * frac * lenergy).backward()
            weighted_mean += float(lenergy.detach().cpu()) * (e - s)
            del xy_chunk, lenergy
            s = e
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise RuntimeError("CUDA OOM even at energy chunk size 1.")
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
        xy_chunk = to_tensor(interior_np[s:e], device, requires_grad=True)
        lenergy = energy_loss(model, xy_chunk, mat, trn, create_graph=False)
        total = total + lenergy * (e - s)
        del xy_chunk, lenergy
        s = e
    return total / n_total


def pde_curriculum_weight(epoch: int, trn: TrainParams) -> float:
    if epoch <= trn.pretrain_epochs:
        return 0.0
    phase_epoch = epoch - trn.pretrain_epochs
    start = min(1.0, max(0.0, trn.initial_pde_weight))
    if trn.pde_ramp_epochs <= 0:
        return 1.0
    ramp = min(1.0, phase_epoch / max(1, trn.pde_ramp_epochs))
    return start + (1.0 - start) * ramp


def energy_curriculum_weight(epoch: int, trn: TrainParams) -> float:
    if trn.pretrain_epochs <= 0:
        return 0.0
    return 1.0 if epoch <= trn.pretrain_epochs else 0.0


# -----------------------------
# Diagnostics and artifacts
# -----------------------------

def residual_statistics(model: nn.Module, mat: MaterialParams, geo: GeometryParams, trn: TrainParams, device: torch.device):
    pts = sample_interior_points(geo, trn.diagnostics_samples)
    xy = to_tensor(pts, device, requires_grad=True)
    res = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps).detach().cpu().numpy()
    abs_res = np.abs(res)
    return {
        "mean_abs": float(abs_res.mean()),
        "max_abs": float(abs_res.max()),
        "rms": float(np.sqrt(np.mean(res ** 2))),
    }


def grid_finite_check(model: nn.Module, geo: GeometryParams, device: torch.device, nx: int = 121, ny: int = 121):
    xs = np.linspace(geo.xmin, geo.xmax, nx, dtype=np.float32)
    ys = np.linspace(geo.ymin, geo.ymax, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        phi = phi_scalar(model, to_tensor(grid, device, requires_grad=False)).detach().cpu().numpy()
    bad = int(np.sum(~np.isfinite(phi)))
    return {"total": int(phi.size), "bad": bad}


def boundary_diagnostics(model: nn.Module, bdata_t: Dict[str, torch.Tensor], bc: BCParams) -> Dict[str, Dict[str, float]]:
    losses = boundary_loss_terms(model, bdata_t, bc)
    diag: Dict[str, Dict[str, float]] = {}
    for label in BOUNDARY_LABELS:
        xy = bdata_t[label]
        with torch.no_grad():
            pred = phi_scalar(model, xy)
            target = dirichlet_target_values(label, xy, bc)
            err = torch.abs(pred - target)
        diag[label] = {
            "count": int(xy.shape[0]),
            "loss": float(losses[label].detach().cpu()),
            "mean_abs_error": float(torch.mean(err).detach().cpu()),
            "max_abs_error": float(torch.max(err).detach().cpu()),
        }
    return diag


def field_diagnostics_on_grid(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    device: torch.device,
    nx: int,
    ny: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    xs = np.linspace(geo.xmin, geo.xmax, nx, dtype=np.float32)
    ys = np.linspace(geo.ymin, geo.ymax, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    phi = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    grad_mag = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    tau_eq = np.full((grid.shape[0],), np.nan, dtype=np.float32)
    residual = np.full((grid.shape[0],), np.nan, dtype=np.float32)

    for s in range(0, grid.shape[0], batch_size):
        e = min(s + batch_size, grid.shape[0])
        xy = to_tensor(grid[s:e], device, requires_grad=True)
        phi_batch = phi_scalar(model, xy)
        _, _, tau_batch = compute_stress(model, xy, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        res_batch = pde_residual(model, xy, mat, create_graph=False, grad_norm_eps=trn.grad_norm_eps)
        phi[s:e] = phi_batch.detach().cpu().numpy().astype(np.float32)
        tau_np = tau_batch.detach().cpu().numpy().astype(np.float32)
        grad_mag[s:e] = tau_np
        tau_eq[s:e] = tau_np
        residual[s:e] = res_batch.detach().cpu().numpy().astype(np.float32)

    return {
        "xs": xs,
        "ys": ys,
        "phi": phi.reshape(ny, nx),
        "grad_mag": grad_mag.reshape(ny, nx),
        "tau_eq": tau_eq.reshape(ny, nx),
        "residual": residual.reshape(ny, nx),
    }


def run_cross_verification(
    model: nn.Module,
    mat: MaterialParams,
    geo: GeometryParams,
    trn: TrainParams,
    bc: BCParams,
    device: torch.device,
    bdata_t: Dict[str, torch.Tensor],
):
    rstats = residual_statistics(model, mat, geo, trn, device)
    finite = grid_finite_check(model, geo, device)
    bdiag = boundary_diagnostics(model, bdata_t, bc)
    print("Cross verification summary:")
    print(
        "  PDE residual | "
        f"mean|r|={rstats['mean_abs']:.5e}, rms={rstats['rms']:.5e}, max|r|={rstats['max_abs']:.5e}"
    )
    print(f"  Finite grid  | bad={finite['bad']} / {finite['total']}")
    for label in BOUNDARY_LABELS:
        info = bdiag[label]
        print(
            f"  {BOUNDARY_DISPLAY[label]:<8} | loss={info['loss']:.5e}, "
            f"mean|err|={info['mean_abs_error']:.5e}, max|err|={info['max_abs_error']:.5e}"
        )
    return {"residual": rstats, "finite": finite, "boundary": bdiag}


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
    out_path = outdir / "pointwise_loss.csv"
    xs = np.linspace(geo.xmin, geo.xmax, trn.pointwise_nx + 2, dtype=np.float32)[1:-1]
    ys = np.linspace(geo.ymin, geo.ymax, trn.pointwise_ny + 2, dtype=np.float32)[1:-1]
    xx, yy = np.meshgrid(xs, ys)
    interior_pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    boundary_pts = sample_boundary_points(geo, trn.pointwise_boundary_each)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "point_type,boundary_label,x,y,phi,target,boundary_error,"
            "pde_residual,pde_loss,energy_density,grad_norm,tau_eq,total_point_loss\n"
        )
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
        for label in BOUNDARY_LABELS:
            pts = boundary_pts[label]
            for s in range(0, pts.shape[0], trn.pointwise_batch_size):
                e = min(s + trn.pointwise_batch_size, pts.shape[0])
                batch = pts[s:e]
                xy = to_tensor(batch, device, requires_grad=False)
                with torch.no_grad():
                    phi = phi_scalar(model, xy).detach().cpu().numpy().astype(np.float64)
                target = dirichlet_target_values(label, xy, bc).detach().cpu().numpy().astype(np.float64)
                boundary_error = phi - target
                total = trn.lambda_bc * np.square(boundary_error)
                for i, (x, y) in enumerate(batch):
                    fh.write(
                        f"boundary,{label},"
                        f"{float(x):.10e},{float(y):.10e},"
                        f"{phi[i]:.10e},{target[i]:.10e},{boundary_error[i]:.10e},"
                        f"nan,nan,nan,nan,nan,{total[i]:.10e}\n"
                    )
    return out_path


@torch.no_grad()
def save_boundary_points_text(
    outdir: Path,
    bdata: Dict[str, np.ndarray],
    model: nn.Module,
    bc: BCParams,
    device: torch.device,
    source: str,
) -> Path:
    out_path = outdir / "boundary_points_training.txt"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# source={source}\n")
        fh.write("boundary_label,x,y,phi_pred,phi_target,abs_error\n")
        for label in BOUNDARY_LABELS:
            pts = np.asarray(bdata[label], dtype=np.float32)
            xy = to_tensor(pts, device, requires_grad=False)
            pred = phi_scalar(model, xy).detach().cpu().numpy()
            target = dirichlet_target_values(label, xy, bc).detach().cpu().numpy()
            for (x, y), p, t in zip(pts, pred, target):
                fh.write(f"{label},{x:.8f},{y:.8f},{p:.8e},{t:.8e},{abs(float(p - t)):.8e}\n")
    return out_path


def save_run_diagnostics(
    outdir: Path,
    trn: TrainParams,
    geo: GeometryParams,
    mat: MaterialParams,
    bc: BCParams,
    collocation_counts: Dict[str, Dict[str, int]],
    boundary_diag: Dict[str, Dict[str, float]],
    verification: Dict[str, object],
    fields: Dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        outdir / "field_diagnostics.npz",
        xs=fields["xs"],
        ys=fields["ys"],
        phi=fields["phi"],
        grad_mag=fields["grad_mag"],
        tau_eq=fields["tau_eq"],
        residual=fields["residual"],
    )
    summary = {
        "problem": "no-crack full-square strain-limiting KAN-PINN",
        "domain": "full square, no internal boundary",
        "boundary_labels": list(BOUNDARY_LABELS),
        "training": {
            "adam_epochs": trn.adam_epochs,
            "finetune_epochs": trn.finetune_epochs,
            "lbfgs_epochs": trn.lbfgs_epochs,
            "pretrain_epochs": trn.pretrain_epochs,
            "pde_ramp_epochs": trn.pde_ramp_epochs,
            "lambda_bc": trn.lambda_bc,
            "lambda_pde": trn.lambda_pde,
            "lambda_energy": trn.lambda_energy,
            "hard_bc_mode": trn.hard_bc_mode,
            "pde_loss_mode": trn.pde_loss_mode,
            "adaptive_sampling": trn.adaptive_sampling,
        },
        "collocation_counts": collocation_counts,
        "material": {"mu": mat.mu, "beta": mat.beta, "alpha": mat.alpha},
        "geometry": {"xmin": geo.xmin, "xmax": geo.xmax, "ymin": geo.ymin, "ymax": geo.ymax},
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
    energy_hist: list[float],
    bc_hist: list[float],
    val_hist: list[float],
    geo: GeometryParams,
    trn: TrainParams,
    outdir: Path,
    device: torch.device,
    boundary_diag: Dict[str, Dict[str, float]],
    collocation_counts: Dict[str, Dict[str, int]],
    verification: Dict[str, object],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_csv = save_loss_history_text(outdir, loss_hist, pde_hist, energy_hist, bc_hist, val_hist)
    print(f"Loss history saved in: {loss_csv}")
    pointwise_csv = save_pointwise_loss_text(outdir, model, mat, geo, bc, trn, device)
    print(f"Pointwise loss saved in: {pointwise_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, lw=2, label="L_total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L_total")
    plt.title("L_total")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss_history.png", dpi=160)
    plt.close()

    fields = field_diagnostics_on_grid(
        model,
        mat,
        geo,
        trn,
        device,
        nx=trn.pointwise_nx,
        ny=trn.pointwise_ny,
        batch_size=trn.pointwise_batch_size,
    )
    xs = fields["xs"]
    ys = fields["ys"]

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

    plot_field(fields["phi"], "Phi(x,y) on full square", "Phi", "phi_field.png")
    plot_field(fields["grad_mag"], "|grad Phi| field", "|grad Phi|", "grad_phi_field.png")
    plot_field(fields["tau_eq"], "Equivalent stress magnitude field", "tau_eq", "tau_eq_field.png")
    plot_field(fields["residual"], "PDE residual field", "Residual", "pde_residual_field.png", cmap="coolwarm")
    save_run_diagnostics(outdir, trn, geo, mat, bc, collocation_counts, boundary_diag, verification, fields)


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


def maybe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, float):
        return value
    return float(value.detach().cpu())


def all_finite(values: Dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in values.values())


def format_bc_loss_line(losses: Dict[str, float]) -> str:
    return ", ".join(f"{BOUNDARY_DISPLAY[label]}={losses.get(label, float('nan')):.2e}" for label in BOUNDARY_LABELS)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
    val_interior = sample_interior_points(geo, trn.val_n_interior_uniform)
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
            "loss_val": val_hist,
            "loss_val_select": val_select_hist,
            "completed_epochs": len(loss_hist),
            "saved_epoch": saved_epoch,
            "saved_reason": reason,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "hard_bc_mode": trn.hard_bc_mode,
            "last_collocation_counts": last_collocation_counts,
            "last_boundary_points": {k: v.copy() for k, v in last_boundary_points.items()},
            "val_collocation_counts": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
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
            suffix = " + best_checkpoint.pt" if save_best else ""
            print(f"[checkpoint] epoch={saved_epoch} reason={reason} -> last_checkpoint.pt{suffix}")

    def evaluate_validation(pde_weight: float, energy_weight: float) -> Tuple[float, float]:
        model.eval()
        with torch.enable_grad():
            if pde_weight > 0.0 or trn.model_select_pde_weight_floor > 0.0:
                v_lpde = streaming_pde_eval(model, val_interior, mat, trn, device)
            else:
                v_lpde = torch.zeros((), dtype=torch.float32, device=device)
            if energy_weight > 0.0:
                v_lenergy = streaming_energy_eval(model, val_interior, mat, trn, device)
            else:
                v_lenergy = torch.zeros((), dtype=torch.float32, device=device)
            v_lbc = boundary_loss(model, val_bdata_t, bc)
            lval = trn.lambda_pde * pde_weight * v_lpde + trn.lambda_energy * energy_weight * v_lenergy + trn.lambda_bc * v_lbc
            select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
            lval_select = trn.lambda_pde * select_wpde * v_lpde + trn.lambda_energy * energy_weight * v_lenergy + trn.lambda_bc * v_lbc
        return maybe_float(lval), maybe_float(lval_select)

    resume_source = ""
    if resume:
        if last_ckpt_path.is_file():
            resume_source = str(last_ckpt_path.name)
        elif best_ckpt_path.is_file():
            resume_source = str(best_ckpt_path.name)

    if resume_source:
        ckpt = load_checkpoint(outdir / resume_source, device)
        last_state_key = "last_model_state" if "last_model_state" in ckpt else "model_state"
        best_state_key = "best_model_state" if "best_model_state" in ckpt else last_state_key
        model.load_state_dict(ckpt[last_state_key])
        best_state = copy.deepcopy(ckpt.get(best_state_key, ckpt[last_state_key]))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        loss_hist = list(ckpt.get("loss_total", []))
        pde_hist = list(ckpt.get("loss_pde", []))
        energy_hist = list(ckpt.get("loss_energy", [0.0] * len(loss_hist)))
        bc_hist = list(ckpt.get("loss_bc", []))
        val_hist = list(ckpt.get("loss_val", []))
        val_select_hist = list(ckpt.get("loss_val_select", ckpt.get("loss_val", [])))
        completed_epochs = int(ckpt.get("completed_epochs", len(loss_hist)))
        last_collocation_counts = dict(ckpt.get("last_collocation_counts", {}))
        loaded_boundary_points = ckpt.get("last_boundary_points", {})
        if isinstance(loaded_boundary_points, dict):
            last_boundary_points = {str(k): np.asarray(v, dtype=np.float32) for k, v in loaded_boundary_points.items()}
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
            "validation": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
        }, last_boundary_points

    def run_stage(
        stage_name: str,
        start_epoch: int,
        end_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> bool:
        nonlocal best_state, best_epoch, best_val, stale_epochs, last_collocation_counts, last_boundary_points
        if start_epoch > end_epoch:
            return False
        print(f"[stage] {stage_name}: epochs {start_epoch}..{end_epoch} | lr_start={current_lr(optimizer):.3e}")
        adaptive_cache = np.empty((0, 2), dtype=np.float32)
        adaptive_cache_epoch = -10**9

        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            interior = sample_interior_points(geo, trn.n_interior_uniform)
            collocation_counts = {"uniform": int(interior.shape[0]), "adaptive": 0, "total": int(interior.shape[0])}
            pde_weight = pde_curriculum_weight(epoch, trn)
            energy_weight = energy_curriculum_weight(epoch, trn)

            if trn.adaptive_sampling and pde_weight > 0.0 and epoch >= trn.adaptive_start_epoch:
                try:
                    n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                    refresh = adaptive_cache.shape[0] == 0 or epoch - adaptive_cache_epoch >= max(1, trn.adaptive_refresh_every)
                    if n_adapt > 0 and refresh:
                        adaptive_cache = adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
                        adaptive_cache_epoch = epoch
                    if adaptive_cache.size > 0:
                        interior = np.vstack([interior, adaptive_cache]).astype(np.float32)
                        collocation_counts["adaptive"] = int(adaptive_cache.shape[0])
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
            bc_terms = boundary_loss_terms(model, bdata_t, bc)
            lbc = torch.stack(list(bc_terms.values())).mean()
            base_loss = trn.lambda_bc * lbc
            if base_loss.requires_grad:
                base_loss.backward()

            lenergy_f = streaming_energy_backward(model, interior, mat, trn, device, energy_weight)
            lpde_f = streaming_pde_backward(model, interior, mat, trn, device, pde_weight)
            lbc_f = maybe_float(lbc)
            ltot_f = trn.lambda_bc * lbc_f + trn.lambda_energy * energy_weight * lenergy_f + trn.lambda_pde * pde_weight * lpde_f
            loss_parts = {"total": ltot_f, "pde": lpde_f, "energy": lenergy_f, "bc": lbc_f}
            if not all_finite(loss_parts):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} before optimizer step; losses={loss_parts}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_before_step_epoch_{epoch}", verbose=True)
                return True

            if trn.max_grad_norm > 0.0:
                grad_norm = maybe_float(torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm))
            else:
                grad_sq = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_sq += maybe_float(torch.sum(param.grad.detach() ** 2))
                grad_norm = math.sqrt(max(0.0, grad_sq))
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} non-finite gradient norm={grad_norm}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_grad_epoch_{epoch}", verbose=True)
                return True

            optimizer.step()
            scheduler.step()
            lr_now = current_lr(optimizer)

            do_validate = epoch == start_epoch or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
            if do_validate:
                lval_f, lval_select_f = evaluate_validation(pde_weight, energy_weight)
            else:
                lval_f = val_hist[-1] if val_hist else float("nan")
                lval_select_f = val_select_hist[-1] if val_select_hist else float("nan")

            loss_hist.append(ltot_f)
            pde_hist.append(lpde_f)
            energy_hist.append(lenergy_f)
            bc_hist.append(lbc_f)
            val_hist.append(lval_f)
            val_select_hist.append(lval_select_f)

            new_best = False
            if do_validate:
                if (not math.isfinite(best_val)) or best_epoch == 0:
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

            periodic_ckpt = trn.checkpoint_every > 0 and epoch % trn.checkpoint_every == 0
            if new_best:
                save_checkpoints(epoch, reason="new_best", save_best=True, verbose=True)
            elif periodic_ckpt:
                save_checkpoints(epoch, reason="periodic", verbose=True)

            should_log = epoch == start_epoch or (trn.print_every > 0 and epoch % trn.print_every == 0)
            if should_log:
                elapsed = time.time() - t0
                epochs_done_this_session = max(1, epoch - session_epoch_start + 1)
                sec_per_epoch = elapsed / epochs_done_this_session
                eta_s = sec_per_epoch * max(0, total_epochs - epoch)
                ckpt_flag = "best+last" if new_best else ("last" if periodic_ckpt else "-")
                val_tag = "val" if do_validate else "val(skip)"
                best_disp = best_val if math.isfinite(best_val) else float("nan")
                print(
                    f"Epoch {epoch:5d}/{total_epochs} | L={ltot_f:.5e} | Lpde={lpde_f:.5e} | "
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lval={lval_f:.5e} ({val_tag}) | "
                    f"lr={lr_now:.3e} | grad={grad_norm:.3e} | wpde={pde_weight:.3f} | "
                    f"wE={energy_weight:.3f} | Nint={collocation_counts['total']} "
                    f"(adaptive={collocation_counts['adaptive']}) | best={best_disp:.5e}@{best_epoch} | "
                    f"new_best={'yes' if new_best else 'no'} | ckpt={ckpt_flag} | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
                bc_loss_values = {k: maybe_float(v) for k, v in bc_terms.items()}
                print(f"  BC(train): {format_bc_loss_line(bc_loss_values)}")

            detailed = epoch == start_epoch or (trn.detailed_diag_every > 0 and epoch % trn.detailed_diag_every == 0)
            if detailed:
                model.eval()
                with torch.enable_grad():
                    rstats = residual_statistics(model, mat, geo, trn, device)
                    bdiag = boundary_diagnostics(model, val_bdata_t, bc)
                print(
                    "  Diag(PDE): "
                    f"mean|r|={rstats['mean_abs']:.4e}, rms={rstats['rms']:.4e}, max|r|={rstats['max_abs']:.4e}"
                )
                bdiag_losses = {label: float(bdiag[label]["loss"]) for label in BOUNDARY_LABELS}
                print(f"  Diag(BC,val): {format_bc_loss_line(bdiag_losses)}")

            if do_validate and trn.early_stop_patience > 0 and stale_epochs >= trn.early_stop_patience:
                print(f"[early-stop] Triggered at epoch {epoch}; best_epoch={best_epoch}")
                return True
        return False

    def run_lbfgs_stage(start_epoch: int, end_epoch: int) -> bool:
        nonlocal best_state, best_epoch, best_val, stale_epochs, last_collocation_counts, last_boundary_points
        if start_epoch > end_epoch:
            return False
        print(f"[stage] lbfgs: epochs {start_epoch}..{end_epoch} | lr={trn.lbfgs_lr:.3e}")
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=trn.lbfgs_lr,
            max_iter=max(1, trn.lbfgs_max_iter),
            history_size=max(1, trn.lbfgs_history_size),
            line_search_fn="strong_wolfe",
        )
        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            interior = sample_interior_points(geo, trn.lbfgs_n_uniform)
            bdata = sample_boundary_points(geo, trn.lbfgs_n_boundary_each)
            bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            interior_t = to_tensor(interior, device, requires_grad=True)
            collocation_counts = {"uniform": int(interior.shape[0]), "adaptive": 0, "total": int(interior.shape[0])}
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}
            closure_vals: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                lpde = pde_loss(model, interior_t, mat, trn, create_graph=True, chunk_size=trn.train_pde_chunk_size)
                lbc = boundary_loss(model, bdata_t, bc)
                loss = trn.lambda_pde * lpde + trn.lambda_bc * lbc
                loss.backward()
                if trn.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm)
                closure_vals.update({"total": maybe_float(loss), "pde": maybe_float(lpde), "bc": maybe_float(lbc)})
                return loss

            optimizer.step(closure)
            lpde_f = closure_vals.get("pde", float("nan"))
            lenergy_f = 0.0
            lbc_f = closure_vals.get("bc", float("nan"))
            ltot_f = closure_vals.get("total", float("nan"))

            grad_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_sq += maybe_float(torch.sum(param.grad.detach() ** 2))
            grad_norm = math.sqrt(max(0.0, grad_sq))

            do_validate = epoch == start_epoch or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
            if do_validate:
                lval_f, lval_select_f = evaluate_validation(1.0, 0.0)
            else:
                lval_f = val_hist[-1] if val_hist else float("nan")
                lval_select_f = val_select_hist[-1] if val_select_hist else float("nan")

            loss_hist.append(ltot_f)
            pde_hist.append(lpde_f)
            energy_hist.append(lenergy_f)
            bc_hist.append(lbc_f)
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

            periodic_ckpt = trn.checkpoint_every > 0 and epoch % trn.checkpoint_every == 0
            if new_best:
                save_checkpoints(epoch, reason="new_best_lbfgs", save_best=True, verbose=True)
            elif periodic_ckpt:
                save_checkpoints(epoch, reason="periodic_lbfgs", verbose=True)

            should_log = epoch == start_epoch or (trn.print_every > 0 and epoch % trn.print_every == 0)
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
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lval={lval_f:.5e} ({val_tag}) | "
                    f"lr={trn.lbfgs_lr:.3e} | grad={grad_norm:.3e} | wpde=1.000 | wE=0.000 | "
                    f"Nint={collocation_counts['total']} | best={best_disp:.5e}@{best_epoch} | "
                    f"new_best={'yes' if new_best else 'no'} | ckpt={ckpt_flag} | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
                bc_loss_values = {k: maybe_float(v) for k, v in boundary_loss_terms(model, bdata_t, bc).items()}
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
        best_val = val_select_hist[-1] if val_select_hist else float("nan")

    model.load_state_dict(best_state)
    final_epoch = len(loss_hist)
    save_checkpoints(final_epoch, reason="training_complete", save_best=True, verbose=True)

    elapsed_total = time.time() - t0
    print("Training summary:")
    print(f"  Completed epochs this run: {max(0, final_epoch - completed_epochs)}")
    print(f"  Total tracked epochs: {final_epoch} / {total_epochs}")
    print(f"  Best validation epoch: {best_epoch}")
    print(f"  Best validation score: {best_val:.6e}")
    print(f"  Final train loss: {loss_hist[-1]:.6e}" if loss_hist else "  Final train loss: n/a")
    print(f"  Final val loss: {val_hist[-1]:.6e}" if val_hist else "  Final val loss: n/a")
    print(f"  Early stopping used: {'yes' if stopped_early else 'no'}")
    print(f"  Runtime (this invocation): {elapsed_total/60:.2f} min")
    print(f"  Checkpoints: best={best_ckpt_path.name}, last={last_ckpt_path.name}")

    return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
        "train_last": last_collocation_counts,
        "validation": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
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


def main() -> None:
    default_pretrain_epochs = env_int("KAN_PINN_PRETRAIN_EPOCHS", 1500)
    default_pde_ramp_epochs = env_int("KAN_PINN_PDE_RAMP_EPOCHS", 5000)
    default_model_select_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)
    default_adaptive_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)

    trn = TrainParams(
        adam_epochs=env_int("KAN_PINN_ADAM_EPOCHS", env_int("KAN_PINN_EPOCHS", 10000)),
        finetune_epochs=env_int("KAN_PINN_FINETUNE_EPOCHS", 5000),
        pretrain_epochs=default_pretrain_epochs,
        pde_ramp_epochs=default_pde_ramp_epochs,
        n_interior_uniform=env_int("KAN_PINN_NU", 1536),
        n_boundary_each=env_int("KAN_PINN_NB", 128),
        val_n_interior_uniform=env_int("KAN_PINN_VAL_NU", 1536),
        val_n_boundary_each=env_int("KAN_PINN_VAL_NB", 128),
        lambda_bc=env_float("KAN_PINN_LAMBDA_BC", 10.0),
        lambda_pde=env_float("KAN_PINN_LAMBDA_PDE", 0.25),
        lambda_energy=env_float("KAN_PINN_LAMBDA_ENERGY", 1.0),
        learning_rate=env_float("KAN_PINN_LR", 1e-4),
        finetune_lr=env_float("KAN_PINN_FINETUNE_LR", 2e-5),
        print_every=env_int("KAN_PINN_PRINT_EVERY", 10),
        validation_every=env_int("KAN_PINN_VAL_EVERY", 10),
        checkpoint_every=env_int("KAN_PINN_CHECKPOINT_EVERY", 50),
        detailed_diag_every=env_int("KAN_PINN_DETAILED_DIAG_EVERY", 100),
        early_stop_patience=env_int("KAN_PINN_PATIENCE", 99999),
        min_improve=env_float("KAN_PINN_MIN_IMPROVE", 1e-5),
        max_grad_norm=env_float("KAN_PINN_MAX_GRAD_NORM", 1.0),
        diagnostics_samples=env_int("KAN_PINN_DIAGNOSTIC_SAMPLES", 512),
        model_select_start_epoch=env_int("KAN_PINN_MODEL_SELECT_START_EPOCH", default_model_select_start),
        model_select_pde_weight_floor=env_float("KAN_PINN_MODEL_SELECT_PDE_FLOOR", 0.25),
        grad_norm_eps=env_float("KAN_PINN_GRAD_NORM_EPS", 1e-10),
        initial_pde_weight=env_float("KAN_PINN_INITIAL_PDE_WEIGHT", 1e-6),
        pde_loss_mode=os.getenv("KAN_PINN_PDE_LOSS_MODE", "pseudo_huber").strip(),
        pde_residual_delta=env_float("KAN_PINN_PDE_RESIDUAL_DELTA", 25.0),
        pde_mse_blend=env_float("KAN_PINN_PDE_MSE_BLEND", 0.02),
        hard_bc_mode=os.getenv("KAN_PINN_HARD_BC_MODE", "distance_outer").strip(),
        hard_bc_eps=env_float("KAN_PINN_HARD_BC_EPS", 1e-5),
        hard_bc_distance_scale=env_float("KAN_PINN_HARD_BC_DISTANCE_SCALE", 0.08),
        hard_bc_distance_power=env_float("KAN_PINN_HARD_BC_DISTANCE_POWER", 2.0),
        lbfgs_epochs=env_int("KAN_PINN_LBFGS_EPOCHS", 250),
        lbfgs_lr=env_float("KAN_PINN_LBFGS_LR", 0.5),
        lbfgs_history_size=env_int("KAN_PINN_LBFGS_HISTORY", 25),
        lbfgs_max_iter=env_int("KAN_PINN_LBFGS_MAX_ITER", 1),
        lbfgs_n_uniform=env_int("KAN_PINN_LBFGS_NU", 256),
        lbfgs_n_boundary_each=env_int("KAN_PINN_LBFGS_NB", 96),
        train_pde_chunk_size=env_int("KAN_PINN_TRAIN_PDE_CHUNK", 256),
        val_pde_chunk_size=env_int("KAN_PINN_VAL_PDE_CHUNK", 256),
        adaptive_sampling=env_bool("KAN_PINN_ADAPTIVE_SAMPLING", True),
        adaptive_candidates=env_int("KAN_PINN_ADAPTIVE_CANDIDATES", 4096),
        adaptive_topk=env_int("KAN_PINN_ADAPTIVE_TOPK", 768),
        adaptive_start_epoch=env_int("KAN_PINN_ADAPTIVE_START_EPOCH", default_adaptive_start),
        adaptive_refresh_every=env_int("KAN_PINN_ADAPTIVE_REFRESH_EVERY", 25),
        pointwise_nx=env_int("KAN_PINN_POINTWISE_NX", 181),
        pointwise_ny=env_int("KAN_PINN_POINTWISE_NY", 181),
        pointwise_boundary_each=env_int("KAN_PINN_POINTWISE_BOUNDARY_EACH", 256),
        pointwise_batch_size=env_int("KAN_PINN_POINTWISE_BATCH", 512),
        seed=env_int("KAN_PINN_SEED", 42),
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
    )
    bc = BCParams(
        sigma0=env_float("KAN_PINN_SIGMA0", 1.0),
        L=env_float("KAN_PINN_L", 1.0),
    )

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

    print("Starting no-crack full-square training (Eq. 40 + Gamma1-Gamma4 Dirichlet BCs).")
    print(f"Device: {device}")
    print(f"Domain: [{geo.xmin:g},{geo.xmax:g}] x [{geo.ymin:g},{geo.ymax:g}]")
    print("Internal boundaries: none")
    print(f"Hard BC ansatz: {trn.hard_bc_mode}")
    print(
        f"PDE curriculum: lambda={trn.lambda_pde:g}, start_weight={trn.initial_pde_weight:g}, "
        f"ramp={trn.pde_ramp_epochs}, loss={trn.pde_loss_mode}, mse_blend={trn.pde_mse_blend:g}"
    )
    print(
        f"Sampling: NU={trn.n_interior_uniform}, NB={trn.n_boundary_each}, "
        f"adaptive={'on' if trn.adaptive_sampling else 'off'}"
    )
    print(
        f"Training phases: energy_pretrain={trn.pretrain_epochs}, "
        f"adam={trn.adam_epochs}, finetune={trn.finetune_epochs}, lbfgs={trn.lbfgs_epochs}"
    )

    model = KANPINN(hidden=trn.hidden, n_basis=trn.n_basis).to(device)
    model.configure_boundary_ansatz(geo, bc, trn)

    root_outdir = Path(__file__).resolve().parent / "results_strainlimiting_no_crack_python"
    outdir, selected_run = get_run_outdir(root_outdir, run_name if run_name else None)
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")

    model, best_epoch, best_val, lhist, lpde_hist, lenergy_hist, lbc_hist, val_hist, collocation_counts, train_boundary_points = train_model(
        model, mat, geo, bc, trn, outdir, device, resume=resume_training
    )
    print(f"Best model selected from epoch {best_epoch} with validation score {best_val:.6e}.")

    final_bdata = sample_boundary_points(geo, trn.val_n_boundary_each)
    final_bdata_t = {k: to_tensor(v, device, requires_grad=False) for k, v in final_bdata.items()}
    boundary_points_source = "last_training_epoch" if train_boundary_points else "post_training_validation_sample"
    boundary_points_to_save = train_boundary_points if train_boundary_points else final_bdata
    boundary_points_txt = save_boundary_points_text(
        outdir,
        boundary_points_to_save,
        model,
        bc,
        device,
        source=boundary_points_source,
    )
    print(f"Boundary datapoints saved in: {boundary_points_txt}")

    verification = run_cross_verification(model, mat, geo, trn, bc, device, final_bdata_t)
    final_boundary_diag = verification["boundary"]
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
