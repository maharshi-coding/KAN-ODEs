#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Experimental no-crack full-square adaptive-grid RBF KAN-PINN for:

    -div( grad(Phi) / (2*mu*(1 + beta*|grad(Phi)|^alpha)^(1/alpha)) ) = 0

on Omega = [0, 1] x [0, 1].

This script is a controlled comparison target for the Chebyshev polynomial
baseline. It follows the adaptive-grid PIKAN idea in a PyTorch-friendly way:

    * grid-dependent one-dimensional Gaussian/RBF basis functions,
    * periodic data-adaptive center updates for every KAN layer,
    * residual-based adaptive collocation point enrichment,
    * the same no-crack hard outer-boundary ansatz and PDE residual,
    * exact-solution diagnostics for Phi(x, y) = sigma0 * (L - x) when L=xmax.

Quick smoke test:
    KAN_PINN_RUN_NAME=adaptive_grid_smoke KAN_PINN_EPOCHS=2 \
    KAN_PINN_FINETUNE_EPOCHS=0 KAN_PINN_PRETRAIN_EPOCHS=0 \
    KAN_PINN_PDE_RAMP_EPOCHS=1 KAN_PINN_LBFGS_EPOCHS=0 \
    KAN_PINN_NU=16 KAN_PINN_NB=8 KAN_PINN_VAL_NU=16 KAN_PINN_VAL_NB=8 \
    KAN_PINN_HIDDEN=8 KAN_PINN_N_BASIS=8 KAN_PINN_ADAPTIVE_CANDIDATES=64 \
    KAN_PINN_ADAPTIVE_TOPK=8 KAN_PINN_POINTWISE_NX=31 KAN_PINN_POINTWISE_NY=31 \
    python3 StrainLimiting_NoCrack_AdaptiveGridKAN_PINN.py
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "StrainLimiting_NoCrack_PolyKAN_PINN.py"
SPEC = importlib.util.spec_from_file_location("_no_crack_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load shared no-crack helpers from {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


BOUNDARY_LABELS = base.BOUNDARY_LABELS
BOUNDARY_DISPLAY = base.BOUNDARY_DISPLAY
MaterialParams = base.MaterialParams
GeometryParams = base.GeometryParams
BCParams = base.BCParams


@dataclass
class AdaptiveGridTrainParams(base.TrainParams):
    n_basis: int = 32
    grid_adaptation: bool = True
    grid_adapt_every: int = 50
    grid_warmup_epochs: int = 100
    grid_mix_uniform: float = 0.05
    grid_width_scale: float = 1.5
    grid_min_width: float = 1e-3


class AdaptiveGridRBFKANLayer(nn.Module):
    """
    KAN layer with grid-dependent Gaussian/RBF basis functions.

    The centers are non-trainable grid state. The coefficients and linear path
    are trainable. Periodic calls to adapt_grid(...) move the centers toward the
    empirical distribution of each layer's current inputs.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_basis: int,
        center_range: Tuple[float, float],
        scale: float = 0.05,
        width_scale: float = 1.5,
        min_width: float = 1e-3,
    ):
        super().__init__()
        if n_basis < 2:
            raise ValueError("KAN_PINN_N_BASIS must be >= 2 for adaptive-grid RBF layers.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.center_range = center_range
        self.width_scale = width_scale
        self.min_width = min_width

        self.coeff = nn.Parameter(scale * torch.randn(out_dim, in_dim, n_basis))
        self.lin = nn.Parameter(scale * torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        lo, hi = center_range
        centers = torch.linspace(float(lo), float(hi), n_basis).repeat(in_dim, 1)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", self._widths_from_centers(centers))

    def _widths_from_centers(self, centers: torch.Tensor) -> torch.Tensor:
        if centers.shape[1] == 1:
            return torch.full_like(centers, self.min_width)
        left = centers[:, 1:] - centers[:, :-1]
        first = left[:, :1]
        last = left[:, -1:]
        local = torch.cat([first, 0.5 * (left[:, :-1] + left[:, 1:]), last], dim=1)
        return torch.clamp(local.abs() * self.width_scale, min=self.min_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        widths = torch.clamp(self.widths, min=self.min_width)
        z = (x.unsqueeze(-1) - self.centers.unsqueeze(0)) / widths.unsqueeze(0)
        basis = torch.exp(-(z ** 2))
        basis_part = torch.einsum("nib,oib->no", basis, self.coeff)
        lin_part = x @ self.lin.t()
        return lin_part + basis_part + self.bias.view(1, -1)

    @torch.no_grad()
    def adapt_grid(self, samples: torch.Tensor, mix_uniform: float) -> None:
        if samples.numel() == 0:
            return
        mix = float(np.clip(mix_uniform, 0.0, 1.0))
        samples = samples.detach()
        new_centers = self.centers.clone()
        default_lo, default_hi = self.center_range

        for dim in range(self.in_dim):
            vals = samples[:, dim]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() < 2:
                continue

            lo = torch.min(vals)
            hi = torch.max(vals)
            if torch.isclose(lo, hi):
                lo = torch.as_tensor(default_lo, dtype=vals.dtype, device=vals.device)
                hi = torch.as_tensor(default_hi, dtype=vals.dtype, device=vals.device)

            uniform = torch.linspace(lo.item(), hi.item(), self.n_basis, dtype=vals.dtype, device=vals.device)
            sorted_vals = torch.sort(vals).values
            q_idx = torch.linspace(0, sorted_vals.numel() - 1, self.n_basis, device=vals.device).round().long()
            adaptive = sorted_vals[q_idx]
            centers = mix * uniform + (1.0 - mix) * adaptive
            new_centers[dim] = torch.sort(centers).values

        self.centers.copy_(new_centers)
        self.widths.copy_(self._widths_from_centers(new_centers))

    def grid_summary(self) -> Dict[str, object]:
        centers = self.centers.detach().cpu().numpy()
        widths = self.widths.detach().cpu().numpy()
        return {
            "n_basis": int(self.n_basis),
            "center_min": float(np.min(centers)),
            "center_max": float(np.max(centers)),
            "width_min": float(np.min(widths)),
            "width_max": float(np.max(widths)),
        }


class AdaptiveGridRBFKANPINN(nn.Module):
    def __init__(
        self,
        hidden: int = 96,
        n_basis: int = 32,
        width_scale: float = 1.5,
        min_width: float = 1e-3,
    ):
        super().__init__()
        self.k1 = AdaptiveGridRBFKANLayer(2, hidden, n_basis, (0.0, 1.0), width_scale=width_scale, min_width=min_width)
        self.k2 = AdaptiveGridRBFKANLayer(hidden, hidden, n_basis, (-1.0, 1.0), width_scale=width_scale, min_width=min_width)
        self.k3 = AdaptiveGridRBFKANLayer(hidden, hidden, n_basis, (-1.0, 1.0), width_scale=width_scale, min_width=min_width)
        self.k4 = AdaptiveGridRBFKANLayer(hidden, 1, n_basis, (-1.0, 1.0), width_scale=width_scale, min_width=min_width)
        self.hard_bc_mode = "none"
        self.hard_bc_eps = 1e-12
        self.hard_bc_distance_scale = 0.08
        self.hard_bc_distance_power = 2.0
        self.geo: GeometryParams | None = None
        self.bc: BCParams | None = None
        self.grid_updates_done = 0

    def configure_boundary_ansatz(self, geo: GeometryParams, bc: BCParams, trn: AdaptiveGridTrainParams) -> None:
        self.geo = geo
        self.bc = bc
        self.hard_bc_mode = base.canonical_hard_bc_mode(trn.hard_bc_mode)
        self.hard_bc_eps = trn.hard_bc_eps
        self.hard_bc_distance_scale = trn.hard_bc_distance_scale
        self.hard_bc_distance_power = trn.hard_bc_distance_power

    def raw_forward(self, xy: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.k1(xy))
        h = torch.tanh(self.k2(h))
        h = torch.tanh(self.k3(h))
        return self.k4(h)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        raw = self.raw_forward(xy).squeeze(-1)
        if self.hard_bc_mode == "none":
            return raw.unsqueeze(-1)
        if self.geo is None or self.bc is None:
            raise RuntimeError("Hard boundary ansatz requested before configure_boundary_ansatz(...).")
        phi = base.hard_boundary_ansatz(
            raw,
            xy,
            self.geo,
            self.bc,
            self.hard_bc_mode,
            self.hard_bc_eps,
            self.hard_bc_distance_scale,
            self.hard_bc_distance_power,
        )
        return phi.unsqueeze(-1)

    @torch.no_grad()
    def adapt_grids(self, xy: torch.Tensor, mix_uniform: float) -> None:
        xy = xy.detach()
        self.k1.adapt_grid(xy, mix_uniform)
        h1 = torch.tanh(self.k1(xy))
        self.k2.adapt_grid(h1, mix_uniform)
        h2 = torch.tanh(self.k2(h1))
        self.k3.adapt_grid(h2, mix_uniform)
        h3 = torch.tanh(self.k3(h2))
        self.k4.adapt_grid(h3, mix_uniform)
        self.grid_updates_done += 1

    def grid_summary(self) -> Dict[str, object]:
        return {
            "updates_done": int(self.grid_updates_done),
            "layers": {
                "k1": self.k1.grid_summary(),
                "k2": self.k2.grid_summary(),
                "k3": self.k3.grid_summary(),
                "k4": self.k4.grid_summary(),
            },
        }


def validate_adaptive_configuration(mat: MaterialParams, geo: GeometryParams, trn: AdaptiveGridTrainParams) -> None:
    base.validate_configuration(mat, geo, trn)
    if trn.n_basis < 2:
        raise ValueError("KAN_PINN_N_BASIS must be >= 2.")
    if trn.grid_adapt_every <= 0:
        raise ValueError("KAN_PINN_GRID_ADAPT_EVERY must be positive.")
    if not (0.0 <= trn.grid_mix_uniform <= 1.0):
        raise ValueError("KAN_PINN_GRID_MIX_UNIFORM must be in [0, 1].")
    if trn.grid_width_scale <= 0.0 or trn.grid_min_width <= 0.0:
        raise ValueError("Grid width scale and minimum width must be positive.")


def exact_solution_diagnostics(
    model: nn.Module,
    geo: GeometryParams,
    bc: BCParams,
    device: torch.device,
    nx: int,
    ny: int,
) -> Dict[str, object]:
    applicable = math.isclose(bc.L, geo.xmax, rel_tol=0.0, abs_tol=1e-12)
    xs = np.linspace(geo.xmin, geo.xmax, nx, dtype=np.float32)
    ys = np.linspace(geo.ymin, geo.ymax, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        pred = model(base.to_tensor(pts, device, requires_grad=False)).squeeze(-1).detach().cpu().numpy()
    exact = bc.sigma0 * (bc.L - pts[:, 0])
    err = pred - exact
    denom = float(np.linalg.norm(exact))
    rel_l2 = float(np.linalg.norm(err) / denom) if denom > 0 else float("nan")
    return {
        "applicable": applicable,
        "formula": "Phi_exact(x,y) = sigma0 * (L - x); exact for this square setup when L == xmax",
        "relative_l2": rel_l2,
        "mean_abs": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def save_run_diagnostics(
    outdir: Path,
    trn: AdaptiveGridTrainParams,
    geo: GeometryParams,
    mat: MaterialParams,
    bc: BCParams,
    model: AdaptiveGridRBFKANPINN,
    collocation_counts: Dict[str, Dict[str, int]],
    boundary_diag: Dict[str, Dict[str, float]],
    verification: Dict[str, object],
    fields: Dict[str, np.ndarray],
    exact_diag: Dict[str, object],
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
        "problem": "no-crack full-square strain-limiting adaptive-grid RBF KAN-PINN",
        "basis_type": "grid-dependent Gaussian/RBF",
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
            "grid_adaptation": trn.grid_adaptation,
            "grid_adapt_every": trn.grid_adapt_every,
            "grid_warmup_epochs": trn.grid_warmup_epochs,
            "grid_mix_uniform": trn.grid_mix_uniform,
        },
        "basis": {
            "n_basis": trn.n_basis,
            "grid_width_scale": trn.grid_width_scale,
            "grid_min_width": trn.grid_min_width,
            "grid_summary": model.grid_summary(),
        },
        "collocation_counts": collocation_counts,
        "material": {"mu": mat.mu, "beta": mat.beta, "alpha": mat.alpha},
        "geometry": {"xmin": geo.xmin, "xmax": geo.xmax, "ymin": geo.ymin, "ymax": geo.ymax},
        "boundary_conditions": {"sigma0": bc.sigma0, "L": bc.L},
        "boundary_diagnostics": boundary_diag,
        "verification": verification,
        "exact_solution_diagnostics": exact_diag,
    }
    (outdir / "run_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_plots(
    model: AdaptiveGridRBFKANPINN,
    mat: MaterialParams,
    bc: BCParams,
    loss_hist: list[float],
    pde_hist: list[float],
    energy_hist: list[float],
    bc_hist: list[float],
    val_hist: list[float],
    geo: GeometryParams,
    trn: AdaptiveGridTrainParams,
    outdir: Path,
    device: torch.device,
    boundary_diag: Dict[str, Dict[str, float]],
    collocation_counts: Dict[str, Dict[str, int]],
    verification: Dict[str, object],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_csv = base.save_loss_history_text(outdir, loss_hist, pde_hist, energy_hist, bc_hist, val_hist)
    print(f"Loss history saved in: {loss_csv}")
    pointwise_csv = base.save_pointwise_loss_text(outdir, model, mat, geo, bc, trn, device)
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

    fields = base.field_diagnostics_on_grid(model, mat, geo, trn, device)
    xs = fields["xs"]
    ys = fields["ys"]

    def plot_field(field: np.ndarray, title: str, label: str, filename: str, cmap: str = "turbo") -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(field, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="auto", cmap=cmap)
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
    exact_diag = exact_solution_diagnostics(model, geo, bc, device, trn.pointwise_nx, trn.pointwise_ny)
    save_run_diagnostics(outdir, trn, geo, mat, bc, model, collocation_counts, boundary_diag, verification, fields, exact_diag)


def train_model(
    model: AdaptiveGridRBFKANPINN,
    mat: MaterialParams,
    geo: GeometryParams,
    bc: BCParams,
    trn: AdaptiveGridTrainParams,
    outdir: Path,
    device: torch.device,
    resume: bool = False,
):
    total_epochs = trn.adam_epochs + trn.finetune_epochs + trn.lbfgs_epochs
    val_interior = base.sample_interior_points(geo, trn.val_n_interior_uniform)
    val_bdata = base.sample_boundary_points(geo, trn.val_n_boundary_each)
    val_bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in val_bdata.items()}

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
            "basis_type": "adaptive_grid_rbf",
            "n_basis": trn.n_basis,
            "hard_bc_mode": trn.hard_bc_mode,
            "grid_summary": model.grid_summary(),
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
        base.save_loss_history_text(outdir, loss_hist, pde_hist, energy_hist, bc_hist, val_hist)
        if verbose:
            suffix = " + best_checkpoint.pt" if save_best else ""
            print(f"[checkpoint] epoch={saved_epoch} reason={reason} -> last_checkpoint.pt{suffix}")

    def evaluate_validation(pde_weight: float, energy_weight: float) -> Tuple[float, float]:
        model.eval()
        with torch.enable_grad():
            if pde_weight > 0.0 or trn.model_select_pde_weight_floor > 0.0:
                v_lpde = base.streaming_pde_eval(model, val_interior, mat, trn, device)
            else:
                v_lpde = torch.zeros((), dtype=torch.float32, device=device)
            if energy_weight > 0.0:
                v_lenergy = base.streaming_energy_eval(model, val_interior, mat, trn, device)
            else:
                v_lenergy = torch.zeros((), dtype=torch.float32, device=device)
            v_lbc = base.boundary_loss(model, val_bdata_t, bc)
            lval = trn.lambda_pde * pde_weight * v_lpde + trn.lambda_energy * energy_weight * v_lenergy + trn.lambda_bc * v_lbc
            select_wpde = max(pde_weight, trn.model_select_pde_weight_floor)
            lval_select = trn.lambda_pde * select_wpde * v_lpde + trn.lambda_energy * energy_weight * v_lenergy + trn.lambda_bc * v_lbc
        return base.maybe_float(lval), base.maybe_float(lval_select)

    if resume and last_ckpt_path.is_file():
        ckpt = base.load_checkpoint(last_ckpt_path, device)
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
        print(f"[resume] source={last_ckpt_path.name} epoch={completed_epochs}/{total_epochs} best_epoch={best_epoch} best_val={best_val:.6e}")
    elif resume:
        print("[resume] Requested resume but no last_checkpoint.pt found; starting from scratch.")

    if completed_epochs >= total_epochs:
        print(f"Checkpoint already reached target epochs ({completed_epochs}). Skipping training.")
        if best_epoch > 0:
            model.load_state_dict(best_state)
        return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
            "train_last": last_collocation_counts,
            "validation": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
        }, last_boundary_points

    t0 = time.time()
    session_epoch_start = max(1, completed_epochs + 1)

    def maybe_adapt_grid(epoch: int, interior: np.ndarray) -> bool:
        if not trn.grid_adaptation:
            return False
        if epoch < trn.grid_warmup_epochs:
            return False
        if epoch % max(1, trn.grid_adapt_every) != 0:
            return False
        model.eval()
        xy = base.to_tensor(interior, device, requires_grad=False)
        model.adapt_grids(xy, trn.grid_mix_uniform)
        del xy
        return True

    def run_adam_stage(stage_name: str, start_epoch: int, end_epoch: int, optimizer, scheduler) -> bool:
        nonlocal best_state, best_epoch, best_val, stale_epochs, last_collocation_counts, last_boundary_points
        if start_epoch > end_epoch:
            return False
        print(f"[stage] {stage_name}: epochs {start_epoch}..{end_epoch} | lr_start={base.current_lr(optimizer):.3e}")
        adaptive_cache = np.empty((0, 2), dtype=np.float32)
        adaptive_cache_epoch = -10**9

        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            interior = base.sample_interior_points(geo, trn.n_interior_uniform)
            collocation_counts = {"uniform": int(interior.shape[0]), "adaptive": 0, "total": int(interior.shape[0])}
            pde_weight = base.pde_curriculum_weight(epoch, trn)
            energy_weight = base.energy_curriculum_weight(epoch, trn)

            if trn.adaptive_sampling and pde_weight > 0.0 and epoch >= trn.adaptive_start_epoch:
                try:
                    n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                    refresh = adaptive_cache.shape[0] == 0 or epoch - adaptive_cache_epoch >= max(1, trn.adaptive_refresh_every)
                    if n_adapt > 0 and refresh:
                        adaptive_cache = base.adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
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

            grid_updated = maybe_adapt_grid(epoch, interior)

            bdata = base.sample_boundary_points(geo, trn.n_boundary_each)
            bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}

            model.train()
            optimizer.zero_grad(set_to_none=True)
            bc_terms = base.boundary_loss_terms(model, bdata_t, bc)
            lbc = torch.stack(list(bc_terms.values())).mean()
            base_loss = trn.lambda_bc * lbc
            if base_loss.requires_grad:
                base_loss.backward()
            lenergy_f = base.streaming_energy_backward(model, interior, mat, trn, device, energy_weight)
            lpde_f = base.streaming_pde_backward(model, interior, mat, trn, device, pde_weight)
            lbc_f = base.maybe_float(lbc)
            ltot_f = trn.lambda_bc * lbc_f + trn.lambda_energy * energy_weight * lenergy_f + trn.lambda_pde * pde_weight * lpde_f

            loss_parts = {"total": ltot_f, "pde": lpde_f, "energy": lenergy_f, "bc": lbc_f}
            if not base.all_finite(loss_parts):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch}; losses={loss_parts}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_epoch_{epoch}", verbose=True)
                return True

            if trn.max_grad_norm > 0.0:
                grad_norm = base.maybe_float(torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm))
            else:
                grad_norm = math.sqrt(sum(base.maybe_float(torch.sum(p.grad.detach() ** 2)) for p in model.parameters() if p.grad is not None))
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} non-finite gradient norm={grad_norm}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_grad_epoch_{epoch}", verbose=True)
                return True

            optimizer.step()
            scheduler.step()
            lr_now = base.current_lr(optimizer)

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
                sec_per_epoch = elapsed / max(1, epoch - session_epoch_start + 1)
                eta_s = sec_per_epoch * max(0, total_epochs - epoch)
                best_disp = best_val if math.isfinite(best_val) else float("nan")
                ckpt_flag = "best+last" if new_best else ("last" if periodic_ckpt else "-")
                val_tag = "val" if do_validate else "val(skip)"
                print(
                    f"Epoch {epoch:5d}/{total_epochs} | L={ltot_f:.5e} | Lpde={lpde_f:.5e} | "
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lval={lval_f:.5e} ({val_tag}) | "
                    f"lr={lr_now:.3e} | grad={grad_norm:.3e} | wpde={pde_weight:.3f} | wE={energy_weight:.3f} | "
                    f"Nint={collocation_counts['total']} (adaptive={collocation_counts['adaptive']}) | "
                    f"grid={'updated' if grid_updated else '-'}#{model.grid_updates_done} | "
                    f"best={best_disp:.5e}@{best_epoch} | new_best={'yes' if new_best else 'no'} | "
                    f"ckpt={ckpt_flag} | elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
                print(f"  BC(train): {base.format_bc_loss_line({k: base.maybe_float(v) for k, v in bc_terms.items()})}")

            detailed = epoch == start_epoch or (trn.detailed_diag_every > 0 and epoch % trn.detailed_diag_every == 0)
            if detailed:
                model.eval()
                with torch.enable_grad():
                    rstats = base.residual_statistics(model, mat, geo, trn, device)
                    bdiag = base.boundary_diagnostics(model, val_bdata_t, bc)
                print(
                    "  Diag(PDE): "
                    f"mean|r|={rstats['mean_abs']:.4e}, rms={rstats['rms']:.4e}, max|r|={rstats['max_abs']:.4e}"
                )
                print(f"  Diag(BC,val): {base.format_bc_loss_line({label: float(bdiag[label]['loss']) for label in BOUNDARY_LABELS})}")

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
            interior = base.sample_interior_points(geo, trn.lbfgs_n_uniform)
            if trn.grid_adaptation and epoch % max(1, trn.grid_adapt_every) == 0:
                maybe_adapt_grid(epoch, interior)
            bdata = base.sample_boundary_points(geo, trn.lbfgs_n_boundary_each)
            bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            interior_t = base.to_tensor(interior, device, requires_grad=True)
            collocation_counts = {"uniform": int(interior.shape[0]), "adaptive": 0, "total": int(interior.shape[0])}
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}
            closure_vals: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                lpde = base.pde_loss(model, interior_t, mat, trn, create_graph=True, chunk_size=trn.train_pde_chunk_size)
                lbc = base.boundary_loss(model, bdata_t, bc)
                loss = trn.lambda_pde * lpde + trn.lambda_bc * lbc
                loss.backward()
                if trn.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm)
                closure_vals.update({"total": base.maybe_float(loss), "pde": base.maybe_float(lpde), "bc": base.maybe_float(lbc)})
                return loss

            optimizer.step(closure)
            lpde_f = closure_vals.get("pde", float("nan"))
            lenergy_f = 0.0
            lbc_f = closure_vals.get("bc", float("nan"))
            ltot_f = closure_vals.get("total", float("nan"))
            grad_norm = math.sqrt(sum(base.maybe_float(torch.sum(p.grad.detach() ** 2)) for p in model.parameters() if p.grad is not None))

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
                eta_s = (elapsed / max(1, epoch - session_epoch_start + 1)) * max(0, total_epochs - epoch)
                best_disp = best_val if math.isfinite(best_val) else float("nan")
                print(
                    f"Epoch {epoch:5d}/{total_epochs} | L={ltot_f:.5e} | Lpde={lpde_f:.5e} | "
                    f"Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | Lval={lval_f:.5e} | "
                    f"lr={trn.lbfgs_lr:.3e} | grad={grad_norm:.3e} | Nint={collocation_counts['total']} | "
                    f"best={best_disp:.5e}@{best_epoch} | new_best={'yes' if new_best else 'no'} | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
                )
        return False

    stopped_early = False
    adam_start = max(1, completed_epochs + 1)
    adam_end = min(trn.adam_epochs, total_epochs)
    if adam_start <= adam_end:
        opt = torch.optim.Adam(model.parameters(), lr=trn.learning_rate)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=trn.lr_gamma_adam)
        stopped_early = run_adam_stage("adam", adam_start, adam_end, opt, sch)
    else:
        print(f"[stage] adam skipped (start={adam_start}, end={adam_end}).")

    finetune_start = max(completed_epochs + 1, trn.adam_epochs + 1)
    finetune_end = min(trn.adam_epochs + trn.finetune_epochs, total_epochs)
    if (not stopped_early) and finetune_start <= finetune_end:
        opt = torch.optim.Adam(model.parameters(), lr=trn.finetune_lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=trn.lr_gamma_finetune)
        stopped_early = run_adam_stage("finetune", finetune_start, finetune_end, opt, sch)
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
    print(f"  Grid updates: {model.grid_updates_done}")
    print(f"  Early stopping used: {'yes' if stopped_early else 'no'}")
    print(f"  Runtime (this invocation): {elapsed_total/60:.2f} min")
    print(f"  Checkpoints: best={best_ckpt_path.name}, last={last_ckpt_path.name}")

    return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
        "train_last": last_collocation_counts,
        "validation": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
    }, last_boundary_points


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

    trn = AdaptiveGridTrainParams(
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
        lr_gamma_adam=env_float("KAN_PINN_LR_GAMMA_ADAM", 0.9998),
        lr_gamma_finetune=env_float("KAN_PINN_LR_GAMMA_FINETUNE", 0.9999),
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
        poly_order=1,
        n_basis=env_int("KAN_PINN_N_BASIS", env_int("KAN_PINN_BASIS", 32)),
        grid_adaptation=env_bool("KAN_PINN_GRID_ADAPTATION", True),
        grid_adapt_every=env_int("KAN_PINN_GRID_ADAPT_EVERY", 50),
        grid_warmup_epochs=env_int("KAN_PINN_GRID_WARMUP_EPOCHS", 100),
        grid_mix_uniform=env_float("KAN_PINN_GRID_MIX_UNIFORM", 0.05),
        grid_width_scale=env_float("KAN_PINN_GRID_WIDTH_SCALE", 1.5),
        grid_min_width=env_float("KAN_PINN_GRID_MIN_WIDTH", 1e-3),
    )
    trn.hard_bc_mode = base.canonical_hard_bc_mode(trn.hard_bc_mode)
    trn.pde_loss_mode = base.canonical_pde_loss_mode(trn.pde_loss_mode)

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
    bc = BCParams(sigma0=env_float("KAN_PINN_SIGMA0", 1.0), L=env_float("KAN_PINN_L", 1.0))
    validate_adaptive_configuration(mat, geo, trn)

    run_name = os.getenv("KAN_PINN_RUN_NAME", "").strip()
    resume_training = os.getenv("KAN_PINN_RESUME", "0").strip().lower() in ("1", "true", "yes", "y")

    random.seed(trn.seed)
    np.random.seed(trn.seed)
    torch.manual_seed(trn.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpl_dir = Path("/tmp/mplconfig_kan_odes")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    print("Starting no-crack full-square training with adaptive-grid RBF KAN.")
    print(f"Device: {device}")
    print(f"Domain: [{geo.xmin:g},{geo.xmax:g}] x [{geo.ymin:g},{geo.ymax:g}]")
    print("Internal boundaries: none")
    print(f"Basis: grid-dependent Gaussian/RBF, n_basis={trn.n_basis}, hidden={trn.hidden}")
    print(
        f"Grid adaptation: {'on' if trn.grid_adaptation else 'off'}, every={trn.grid_adapt_every}, "
        f"warmup={trn.grid_warmup_epochs}, mix_uniform={trn.grid_mix_uniform:g}"
    )
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

    model = AdaptiveGridRBFKANPINN(
        hidden=trn.hidden,
        n_basis=trn.n_basis,
        width_scale=trn.grid_width_scale,
        min_width=trn.grid_min_width,
    ).to(device)
    model.configure_boundary_ansatz(geo, bc, trn)

    root_outdir = THIS_DIR / "results_strainlimiting_no_crack_adaptive_grid_python"
    outdir, selected_run = base.get_run_outdir(root_outdir, run_name if run_name else None)
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")

    model, best_epoch, best_val, lhist, lpde_hist, lenergy_hist, lbc_hist, val_hist, collocation_counts, train_boundary_points = train_model(
        model, mat, geo, bc, trn, outdir, device, resume=resume_training
    )
    print(f"Best model selected from epoch {best_epoch} with validation score {best_val:.6e}.")

    final_bdata = base.sample_boundary_points(geo, trn.val_n_boundary_each)
    final_bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in final_bdata.items()}
    boundary_points_source = "last_training_epoch" if train_boundary_points else "post_training_validation_sample"
    boundary_points_to_save = train_boundary_points if train_boundary_points else final_bdata
    boundary_points_txt = base.save_boundary_points_text(outdir, boundary_points_to_save, model, bc, device, boundary_points_source)
    print(f"Boundary datapoints saved in: {boundary_points_txt}")

    verification = base.run_cross_verification(model, mat, geo, trn, bc, device, final_bdata_t)
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
    exact_diag = exact_solution_diagnostics(model, geo, bc, device, trn.pointwise_nx, trn.pointwise_ny)
    print(
        "Exact linear solution check: "
        f"rel_L2={exact_diag['relative_l2']:.5e}, mean|err|={exact_diag['mean_abs']:.5e}, "
        f"max|err|={exact_diag['max_abs']:.5e}"
    )
    print(f"Training complete. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
