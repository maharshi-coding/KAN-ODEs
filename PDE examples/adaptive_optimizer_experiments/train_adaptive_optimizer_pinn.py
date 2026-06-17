#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Adaptive-grid optimizer fallback trainer for the no-crack strain-limiting PDE.

This script keeps the PDE, full-square boundary conditions, hard boundary
ansatz, diagnostics, and adaptive-grid RBF KAN layers from
StrainLimiting_NoCrack_AdaptiveGridKAN_PINN.py. The difference is the training
controller: it can run a sequence of optimizers. When an optimizer stalls, the
next optimizer starts as a fresh training attempt from the original initialized
model, while the run keeps all histories and selects the best model overall.

Default stages:
    adamw:10000:1e-4,radam:5000:5e-5,lbfgs:250:0.5

Quick smoke test:
    KAN_PINN_RUN_NAME=optimizer_smoke KAN_OPT_STAGES=adamw:2:1e-4,lbfgs:1:0.5 \
    KAN_PINN_PRETRAIN_EPOCHS=0 KAN_PINN_PDE_RAMP_EPOCHS=1 \
    KAN_PINN_NU=16 KAN_PINN_NB=8 KAN_PINN_VAL_NU=16 KAN_PINN_VAL_NB=8 \
    KAN_PINN_HIDDEN=8 KAN_PINN_N_BASIS=8 KAN_PINN_ADAPTIVE_CANDIDATES=64 \
    KAN_PINN_ADAPTIVE_TOPK=8 KAN_PINN_POINTWISE_NX=31 KAN_PINN_POINTWISE_NY=31 \
    python3 train_adaptive_optimizer_pinn.py
"""

from __future__ import annotations

import copy
import csv
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PDE_DIR = THIS_DIR.parent
ADAPTIVE_SCRIPT = PDE_DIR / "StrainLimiting_NoCrack_AdaptiveGridKAN_PINN.py"
SPEC = importlib.util.spec_from_file_location("_adaptive_grid_no_crack", ADAPTIVE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load adaptive-grid implementation from {ADAPTIVE_SCRIPT}")
adaptive = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adaptive
SPEC.loader.exec_module(adaptive)
base = adaptive.base


MaterialParams = adaptive.MaterialParams
GeometryParams = adaptive.GeometryParams
BCParams = adaptive.BCParams
AdaptiveGridTrainParams = adaptive.AdaptiveGridTrainParams
AdaptiveGridRBFKANPINN = adaptive.AdaptiveGridRBFKANPINN
BOUNDARY_LABELS = adaptive.BOUNDARY_LABELS


SUPPORTED_FIRST_ORDER = {"adam", "adamw", "nadam", "radam", "rmsprop", "sgd"}
SUPPORTED_OPTIMIZERS = SUPPORTED_FIRST_ORDER | {"lbfgs"}


@dataclass(frozen=True)
class OptimizerStage:
    name: str
    epochs: int
    lr: float
    gamma: float
    weight_decay: float = 0.0


@dataclass
class OptimizerControllerParams:
    switch_on_plateau: bool = True
    restart_on_optimizer_change: bool = True
    plateau_patience_validations: int = 8
    plateau_min_delta: float = 1e-5
    min_validations_before_switch: int = 3
    default_gamma: float = 0.9998
    default_weight_decay: float = 0.0
    lbfgs_history_size: int = 25
    lbfgs_max_iter: int = 1


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_optimizer_stages(raw: str, ctrl: OptimizerControllerParams) -> List[OptimizerStage]:
    if not raw.strip():
        raw = "adamw:10000:1e-4,radam:5000:5e-5,lbfgs:250:0.5"
    stages: List[OptimizerStage] = []
    for chunk in raw.split(","):
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) < 3:
            raise ValueError(
                "Each optimizer stage must be name:epochs:lr[:gamma[:weight_decay]], "
                f"got {chunk!r}."
            )
        name = parts[0].lower()
        if name not in SUPPORTED_OPTIMIZERS:
            allowed = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
            raise ValueError(f"Unsupported optimizer {name!r}; allowed: {allowed}")
        epochs = int(parts[1])
        lr = float(parts[2])
        gamma = float(parts[3]) if len(parts) >= 4 and parts[3] else ctrl.default_gamma
        weight_decay = float(parts[4]) if len(parts) >= 5 and parts[4] else ctrl.default_weight_decay
        if epochs <= 0:
            raise ValueError(f"Optimizer stage {chunk!r} must have positive epochs.")
        if lr <= 0.0:
            raise ValueError(f"Optimizer stage {chunk!r} must have positive lr.")
        stages.append(OptimizerStage(name=name, epochs=epochs, lr=lr, gamma=gamma, weight_decay=weight_decay))
    return stages


def build_first_order_optimizer(stage: OptimizerStage, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if stage.name == "adam":
        return torch.optim.Adam(params, lr=stage.lr, weight_decay=stage.weight_decay)
    if stage.name == "adamw":
        return torch.optim.AdamW(params, lr=stage.lr, weight_decay=stage.weight_decay)
    if stage.name == "nadam":
        return torch.optim.NAdam(params, lr=stage.lr, weight_decay=stage.weight_decay)
    if stage.name == "radam":
        return torch.optim.RAdam(params, lr=stage.lr, weight_decay=stage.weight_decay)
    if stage.name == "rmsprop":
        return torch.optim.RMSprop(params, lr=stage.lr, weight_decay=stage.weight_decay, momentum=0.9)
    if stage.name == "sgd":
        return torch.optim.SGD(params, lr=stage.lr, weight_decay=stage.weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"{stage.name} is not a first-order optimizer stage.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage: OptimizerStage,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if stage.gamma <= 0.0 or math.isclose(stage.gamma, 1.0):
        return None
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=stage.gamma)


def save_stage_history(outdir: Path, rows: List[Dict[str, object]]) -> Path:
    out_path = outdir / "optimizer_stage_history.csv"
    fieldnames = [
        "epoch",
        "stage_index",
        "stage_name",
        "attempt_epoch",
        "optimizer",
        "lr",
        "total_loss",
        "pde_loss",
        "energy_loss",
        "boundary_loss",
        "validation_loss",
        "validation_select_loss",
        "grad_norm",
        "pde_weight",
        "energy_weight",
        "grid_updated",
        "grid_updates_done",
        "adaptive_points",
        "restart_from_initial",
        "best_epoch",
        "best_validation",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def make_training_params() -> AdaptiveGridTrainParams:
    default_pretrain_epochs = env_int("KAN_PINN_PRETRAIN_EPOCHS", 1500)
    default_pde_ramp_epochs = env_int("KAN_PINN_PDE_RAMP_EPOCHS", 5000)
    default_model_select_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)
    default_adaptive_start = default_pretrain_epochs + max(400, default_pde_ramp_epochs // 2)
    trn = AdaptiveGridTrainParams(
        adam_epochs=0,
        finetune_epochs=0,
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
        lbfgs_epochs=0,
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
    return trn


def make_controller_params() -> OptimizerControllerParams:
    return OptimizerControllerParams(
        switch_on_plateau=env_bool("KAN_OPT_SWITCH_ON_PLATEAU", True),
        restart_on_optimizer_change=env_bool("KAN_OPT_RESTART_ON_OPTIMIZER_CHANGE", True),
        plateau_patience_validations=env_int("KAN_OPT_PLATEAU_PATIENCE", 8),
        plateau_min_delta=env_float("KAN_OPT_PLATEAU_MIN_DELTA", env_float("KAN_PINN_MIN_IMPROVE", 1e-5)),
        min_validations_before_switch=env_int("KAN_OPT_MIN_VALIDATIONS_BEFORE_SWITCH", 3),
        default_gamma=env_float("KAN_OPT_DEFAULT_GAMMA", env_float("KAN_PINN_LR_GAMMA_ADAM", 0.9998)),
        default_weight_decay=env_float("KAN_OPT_WEIGHT_DECAY", 0.0),
        lbfgs_history_size=env_int("KAN_PINN_LBFGS_HISTORY", 25),
        lbfgs_max_iter=env_int("KAN_PINN_LBFGS_MAX_ITER", 1),
    )


def train_with_optimizer_stages(
    model: AdaptiveGridRBFKANPINN,
    mat: MaterialParams,
    geo: GeometryParams,
    bc: BCParams,
    trn: AdaptiveGridTrainParams,
    stages: List[OptimizerStage],
    ctrl: OptimizerControllerParams,
    outdir: Path,
    device: torch.device,
    resume: bool = False,
):
    total_epochs = sum(stage.epochs for stage in stages)
    val_interior = base.sample_interior_points(geo, trn.val_n_interior_uniform)
    val_bdata = base.sample_boundary_points(geo, trn.val_n_boundary_each)
    val_bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in val_bdata.items()}

    initial_model_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    completed_epochs = 0
    stale_validations_global = 0
    last_collocation_counts: Dict[str, int] = {}
    last_boundary_points: Dict[str, np.ndarray] = {}
    loss_hist: List[float] = []
    pde_hist: List[float] = []
    energy_hist: List[float] = []
    bc_hist: List[float] = []
    val_hist: List[float] = []
    val_select_hist: List[float] = []
    stage_rows: List[Dict[str, object]] = []
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
            "optimizer_stages": [asdict(stage) for stage in stages],
            "controller": asdict(ctrl),
            "initial_model_state": copy.deepcopy(initial_model_state),
            "stage_rows": stage_rows,
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
        save_stage_history(outdir, stage_rows)
        if verbose:
            suffix = " + best_checkpoint.pt" if save_best else ""
            print(f"[checkpoint] epoch={saved_epoch} reason={reason} -> last_checkpoint.pt{suffix}")

    def evaluate_validation(epoch: int, pde_weight: float, energy_weight: float) -> Tuple[float, float]:
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
        _ = epoch
        return base.maybe_float(lval), base.maybe_float(lval_select)

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

    if resume and last_ckpt_path.is_file():
        ckpt = base.load_checkpoint(last_ckpt_path, device)
        last_state_key = "last_model_state" if "last_model_state" in ckpt else "model_state"
        best_state_key = "best_model_state" if "best_model_state" in ckpt else last_state_key
        model.load_state_dict(ckpt[last_state_key])
        initial_model_state = copy.deepcopy(ckpt.get("initial_model_state", initial_model_state))
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
        stage_rows = list(ckpt.get("stage_rows", []))
        last_collocation_counts = dict(ckpt.get("last_collocation_counts", {}))
        loaded_boundary_points = ckpt.get("last_boundary_points", {})
        if isinstance(loaded_boundary_points, dict):
            last_boundary_points = {str(k): np.asarray(v, dtype=np.float32) for k, v in loaded_boundary_points.items()}
        print(f"[resume] epoch={completed_epochs}/{total_epochs} best_epoch={best_epoch} best_val={best_val:.6e}")
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

    def after_epoch_bookkeeping(
        *,
        epoch: int,
        attempt_epoch: int,
        stage_idx: int,
        stage: OptimizerStage,
        lr_now: float,
        ltot_f: float,
        lpde_f: float,
        lenergy_f: float,
        lbc_f: float,
        lval_f: float,
        lval_select_f: float,
        grad_norm: float,
        pde_weight: float,
        energy_weight: float,
        grid_updated: bool,
        adaptive_points: int,
        restart_from_initial: bool,
        do_validate: bool,
        stage_validation_count: int,
    ) -> Tuple[bool, int, int]:
        nonlocal best_state, best_epoch, best_val, stale_validations_global
        loss_hist.append(ltot_f)
        pde_hist.append(lpde_f)
        energy_hist.append(lenergy_f)
        bc_hist.append(lbc_f)
        val_hist.append(lval_f)
        val_select_hist.append(lval_select_f)

        new_best = False
        stage_stale_increment = 0
        if do_validate:
            stage_validation_count += 1
            improves = (not math.isfinite(best_val)) or best_epoch == 0 or lval_select_f < best_val - ctrl.plateau_min_delta
            if improves:
                best_val = lval_select_f
                best_epoch = epoch
                stale_validations_global = 0
                best_state = copy.deepcopy(model.state_dict())
                new_best = True
            elif epoch >= trn.model_select_start_epoch:
                stale_validations_global += 1
                stage_stale_increment = 1

        stage_rows.append(
            {
                "epoch": epoch,
                "stage_index": stage_idx,
                "stage_name": f"{stage_idx}_{stage.name}",
                "attempt_epoch": attempt_epoch,
                "optimizer": stage.name,
                "lr": lr_now,
                "total_loss": ltot_f,
                "pde_loss": lpde_f,
                "energy_loss": lenergy_f,
                "boundary_loss": lbc_f,
                "validation_loss": lval_f,
                "validation_select_loss": lval_select_f,
                "grad_norm": grad_norm,
                "pde_weight": pde_weight,
                "energy_weight": energy_weight,
                "grid_updated": int(grid_updated),
                "grid_updates_done": int(model.grid_updates_done),
                "adaptive_points": adaptive_points,
                "restart_from_initial": int(restart_from_initial),
                "best_epoch": best_epoch,
                "best_validation": best_val,
            }
        )

        periodic_ckpt = trn.checkpoint_every > 0 and epoch % trn.checkpoint_every == 0
        if new_best:
            save_checkpoints(epoch, reason=f"new_best_{stage.name}", save_best=True, verbose=True)
        elif periodic_ckpt:
            save_checkpoints(epoch, reason=f"periodic_{stage.name}", verbose=True)

        should_log = epoch == session_epoch_start or (trn.print_every > 0 and epoch % trn.print_every == 0)
        if should_log:
            elapsed = time.time() - t0
            sec_per_epoch = elapsed / max(1, epoch - session_epoch_start + 1)
            eta_s = sec_per_epoch * max(0, total_epochs - epoch)
            best_disp = best_val if math.isfinite(best_val) else float("nan")
            val_tag = "val" if do_validate else "val(skip)"
            print(
                f"Epoch {epoch:5d}/{total_epochs} | attempt_epoch={attempt_epoch:5d} | opt={stage.name} | L={ltot_f:.5e} | "
                f"Lpde={lpde_f:.5e} | Lenergy={lenergy_f:.5e} | Lbc={lbc_f:.5e} | "
                f"Lval={lval_f:.5e} ({val_tag}) | lr={lr_now:.3e} | grad={grad_norm:.3e} | "
                f"wpde={pde_weight:.3f} | wE={energy_weight:.3f} | adaptive={adaptive_points} | "
                f"grid={'updated' if grid_updated else '-'}#{model.grid_updates_done} | "
                f"best={best_disp:.5e}@{best_epoch} | new_best={'yes' if new_best else 'no'} | "
                f"elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m"
            )

        detailed = epoch == session_epoch_start or (trn.detailed_diag_every > 0 and epoch % trn.detailed_diag_every == 0)
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
        return new_best, stage_stale_increment, stage_validation_count

    def run_first_order_stage(
        stage_idx: int,
        stage: OptimizerStage,
        start_epoch: int,
        end_epoch: int,
        attempt_epoch_offset: int,
        restart_from_initial: bool,
    ) -> bool:
        nonlocal last_collocation_counts, last_boundary_points
        optimizer = build_first_order_optimizer(stage, model)
        scheduler = build_scheduler(optimizer, stage)
        print(f"[stage] {stage_idx}: {stage.name} epochs {start_epoch}..{end_epoch} | lr_start={stage.lr:.3e}")
        adaptive_cache = np.empty((0, 2), dtype=np.float32)
        adaptive_cache_epoch = -10**9
        stage_stale_validations = 0
        stage_validation_count = 0

        for epoch in range(start_epoch, end_epoch + 1):
            attempt_epoch = attempt_epoch_offset + epoch - start_epoch + 1
            model.train()
            interior = base.sample_interior_points(geo, trn.n_interior_uniform)
            collocation_counts = {"uniform": int(interior.shape[0]), "adaptive": 0, "total": int(interior.shape[0])}
            pde_weight = base.pde_curriculum_weight(attempt_epoch, trn)
            energy_weight = base.energy_curriculum_weight(attempt_epoch, trn)

            if trn.adaptive_sampling and pde_weight > 0.0 and attempt_epoch >= trn.adaptive_start_epoch:
                try:
                    n_adapt = min(trn.adaptive_topk, max(0, interior.shape[0] // 4))
                    refresh = adaptive_cache.shape[0] == 0 or attempt_epoch - adaptive_cache_epoch >= max(1, trn.adaptive_refresh_every)
                    if n_adapt > 0 and refresh:
                        adaptive_cache = base.adaptive_residual_points(model, geo, mat, trn, device, n_adapt)
                        adaptive_cache_epoch = attempt_epoch
                    if adaptive_cache.size > 0:
                        interior = np.vstack([interior, adaptive_cache]).astype(np.float32)
                        collocation_counts["adaptive"] = int(adaptive_cache.shape[0])
                        collocation_counts["total"] = int(interior.shape[0])
                except RuntimeError as exc:
                    print(f"[adaptive sampling] RuntimeError; skipping adaptive points this epoch. {exc}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            grid_updated = maybe_adapt_grid(attempt_epoch, interior)
            bdata = base.sample_boundary_points(geo, trn.n_boundary_each)
            bdata_t = {k: base.to_tensor(v, device, requires_grad=False) for k, v in bdata.items()}
            last_collocation_counts = dict(collocation_counts)
            last_boundary_points = {k: v.copy() for k, v in bdata.items()}

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
                break

            if trn.max_grad_norm > 0.0:
                grad_norm = base.maybe_float(torch.nn.utils.clip_grad_norm_(model.parameters(), trn.max_grad_norm))
            else:
                grad_norm = math.sqrt(sum(base.maybe_float(torch.sum(p.grad.detach() ** 2)) for p in model.parameters() if p.grad is not None))
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[non-finite-stop] epoch={epoch} non-finite gradient norm={grad_norm}")
                save_checkpoints(len(loss_hist), reason=f"non_finite_grad_epoch_{epoch}", verbose=True)
                break

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr_now = base.current_lr(optimizer)

            do_validate = epoch == start_epoch or (trn.validation_every > 0 and epoch % trn.validation_every == 0)
            if do_validate:
                lval_f, lval_select_f = evaluate_validation(epoch, pde_weight, energy_weight)
            else:
                lval_f = val_hist[-1] if val_hist else float("nan")
                lval_select_f = val_select_hist[-1] if val_select_hist else float("nan")

            new_best, stale_inc, stage_validation_count = after_epoch_bookkeeping(
                epoch=epoch,
                attempt_epoch=attempt_epoch,
                stage_idx=stage_idx,
                stage=stage,
                lr_now=lr_now,
                ltot_f=ltot_f,
                lpde_f=lpde_f,
                lenergy_f=lenergy_f,
                lbc_f=lbc_f,
                lval_f=lval_f,
                lval_select_f=lval_select_f,
                grad_norm=grad_norm,
                pde_weight=pde_weight,
                energy_weight=energy_weight,
                grid_updated=grid_updated,
                adaptive_points=collocation_counts["adaptive"],
                restart_from_initial=restart_from_initial,
                do_validate=do_validate,
                stage_validation_count=stage_validation_count,
            )
            stage_stale_validations = 0 if new_best else stage_stale_validations + stale_inc
            can_switch = (
                ctrl.switch_on_plateau
                and do_validate
                and stage_validation_count >= ctrl.min_validations_before_switch
                and stage_stale_validations >= ctrl.plateau_patience_validations
                and stage_idx < len(stages) - 1
            )
            if can_switch:
                print(
                    f"[optimizer-switch] {stage.name} plateaued after {stage_stale_validations} stale validations; "
                    "restarting with the next optimizer."
                )
                return True
        return False

    def run_lbfgs_stage(
        stage_idx: int,
        stage: OptimizerStage,
        start_epoch: int,
        end_epoch: int,
        attempt_epoch_offset: int,
        restart_from_initial: bool,
    ) -> bool:
        nonlocal last_collocation_counts, last_boundary_points
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=stage.lr,
            max_iter=max(1, ctrl.lbfgs_max_iter),
            history_size=max(1, ctrl.lbfgs_history_size),
            line_search_fn="strong_wolfe",
        )
        print(f"[stage] {stage_idx}: lbfgs epochs {start_epoch}..{end_epoch} | lr={stage.lr:.3e}")
        stage_validation_count = 0
        for epoch in range(start_epoch, end_epoch + 1):
            attempt_epoch = attempt_epoch_offset + epoch - start_epoch + 1
            model.train()
            interior = base.sample_interior_points(geo, trn.lbfgs_n_uniform)
            grid_updated = maybe_adapt_grid(attempt_epoch, interior)
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
                lval_f, lval_select_f = evaluate_validation(epoch, 1.0, 0.0)
            else:
                lval_f = val_hist[-1] if val_hist else float("nan")
                lval_select_f = val_select_hist[-1] if val_select_hist else float("nan")

            after_epoch_bookkeeping(
                epoch=epoch,
                attempt_epoch=attempt_epoch,
                stage_idx=stage_idx,
                stage=stage,
                lr_now=stage.lr,
                ltot_f=ltot_f,
                lpde_f=lpde_f,
                lenergy_f=lenergy_f,
                lbc_f=lbc_f,
                lval_f=lval_f,
                lval_select_f=lval_select_f,
                grad_norm=grad_norm,
                pde_weight=1.0,
                energy_weight=0.0,
                grid_updated=grid_updated,
                adaptive_points=0,
                restart_from_initial=restart_from_initial,
                do_validate=do_validate,
                stage_validation_count=stage_validation_count,
            )
            del interior_t
        return False

    stage_counts: Dict[int, int] = {}
    for row in stage_rows:
        try:
            idx = int(row.get("stage_index", 0))
        except (TypeError, ValueError):
            idx = 0
        stage_counts[idx] = stage_counts.get(idx, 0) + 1

    start_stage_idx = 0
    while start_stage_idx < len(stages) and stage_counts.get(start_stage_idx, 0) >= stages[start_stage_idx].epochs:
        print(f"[stage] {start_stage_idx}: {stages[start_stage_idx].name} already completed by checkpoint.")
        start_stage_idx += 1

    for stage_idx in range(start_stage_idx, len(stages)):
        stage = stages[stage_idx]
        already_done_in_stage = stage_counts.get(stage_idx, 0) if stage_idx == start_stage_idx else 0
        remaining_in_stage = max(0, stage.epochs - already_done_in_stage)
        if remaining_in_stage <= 0:
            continue
        restart_from_initial = ctrl.restart_on_optimizer_change and stage_idx > 0 and already_done_in_stage == 0
        if restart_from_initial:
            model.load_state_dict(initial_model_state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[restart] Starting {stage.name} from the original initialized model state.")
        start_epoch = len(loss_hist) + 1
        end_epoch = min(total_epochs, start_epoch + remaining_in_stage - 1)
        if stage.name == "lbfgs":
            plateau_switched = run_lbfgs_stage(stage_idx, stage, start_epoch, end_epoch, already_done_in_stage, restart_from_initial)
        else:
            plateau_switched = run_first_order_stage(stage_idx, stage, start_epoch, end_epoch, already_done_in_stage, restart_from_initial)
        completed_epochs = len(loss_hist)
        if plateau_switched:
            save_checkpoints(completed_epochs, reason=f"plateau_switch_from_{stage.name}", verbose=True)
        if completed_epochs >= total_epochs:
            break

    if best_epoch <= 0:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(loss_hist)
        best_val = val_select_hist[-1] if val_select_hist else float("nan")
    model.load_state_dict(best_state)
    final_epoch = len(loss_hist)
    save_checkpoints(final_epoch, reason="training_complete", save_best=True, verbose=True)

    elapsed_total = time.time() - t0
    print("Training summary:")
    print(f"  Total tracked epochs: {final_epoch} / {total_epochs}")
    print(f"  Best validation epoch: {best_epoch}")
    print(f"  Best validation score: {best_val:.6e}")
    print(f"  Final train loss: {loss_hist[-1]:.6e}" if loss_hist else "  Final train loss: n/a")
    print(f"  Final val loss: {val_hist[-1]:.6e}" if val_hist else "  Final val loss: n/a")
    print(f"  Grid updates: {model.grid_updates_done}")
    print(f"  Runtime: {elapsed_total/60:.2f} min")
    print(f"  Stage history: {outdir / 'optimizer_stage_history.csv'}")

    return model, best_epoch, best_val, loss_hist, pde_hist, energy_hist, bc_hist, val_hist, {
        "train_last": last_collocation_counts,
        "validation": {"uniform": int(val_interior.shape[0]), "total": int(val_interior.shape[0])},
    }, last_boundary_points


def main() -> None:
    trn = make_training_params()
    ctrl = make_controller_params()
    stages = parse_optimizer_stages(os.getenv("KAN_OPT_STAGES", ""), ctrl)
    trn.adam_epochs = sum(stage.epochs for stage in stages if stage.name != "lbfgs")
    trn.lbfgs_epochs = sum(stage.epochs for stage in stages if stage.name == "lbfgs")
    adaptive.validate_adaptive_configuration(
        MaterialParams(
            mu=env_float("KAN_PINN_MU", 1.0),
            beta=env_float("KAN_PINN_BETA", 1.0),
            alpha=env_float("KAN_PINN_ALPHA", 0.2),
        ),
        GeometryParams(
            xmin=env_float("KAN_PINN_XMIN", 0.0),
            xmax=env_float("KAN_PINN_XMAX", 1.0),
            ymin=env_float("KAN_PINN_YMIN", 0.0),
            ymax=env_float("KAN_PINN_YMAX", 1.0),
        ),
        trn,
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
    bc = BCParams(sigma0=env_float("KAN_PINN_SIGMA0", 1.0), L=env_float("KAN_PINN_L", 1.0))
    adaptive.validate_adaptive_configuration(mat, geo, trn)

    random.seed(trn.seed)
    np.random.seed(trn.seed)
    torch.manual_seed(trn.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpl_dir = Path("/tmp/mplconfig_kan_odes")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    run_name = os.getenv("KAN_PINN_RUN_NAME", "").strip()
    resume_training = env_bool("KAN_PINN_RESUME", False)
    root_outdir = THIS_DIR / "results_adaptive_optimizer_runs"
    outdir, selected_run = base.get_run_outdir(root_outdir, run_name if run_name else None)

    print("Starting adaptive-grid no-crack strain-limiting KAN-PINN optimizer experiment.")
    print(f"Device: {device}")
    print(f"Run directory: {outdir}")
    print(f"Run ID: {selected_run}")
    print(f"Domain: [{geo.xmin:g},{geo.xmax:g}] x [{geo.ymin:g},{geo.ymax:g}]")
    print(f"Material: mu={mat.mu:g}, beta={mat.beta:g}, alpha={mat.alpha:g}")
    print(f"Boundary conditions: G1 sigma0*L, G2 0, G3 -sigma0*(x-L), G4 sigma0*(L-x)")
    print(f"Basis: adaptive-grid Gaussian/RBF, hidden={trn.hidden}, n_basis={trn.n_basis}")
    print(f"Optimizer stages: {', '.join(f'{s.name}:{s.epochs}:{s.lr:g}' for s in stages)}")
    print(
        f"Plateau switching: {'on' if ctrl.switch_on_plateau else 'off'}, "
        f"patience={ctrl.plateau_patience_validations} validations, min_delta={ctrl.plateau_min_delta:g}"
    )
    print(f"Restart on optimizer change: {'on' if ctrl.restart_on_optimizer_change else 'off'}")
    print(
        f"Adaptive grid: {'on' if trn.grid_adaptation else 'off'}, every={trn.grid_adapt_every}, "
        f"warmup={trn.grid_warmup_epochs}, mix_uniform={trn.grid_mix_uniform:g}"
    )
    print(
        f"Residual adaptive sampling: {'on' if trn.adaptive_sampling else 'off'}, "
        f"candidates={trn.adaptive_candidates}, topk={trn.adaptive_topk}"
    )

    (outdir / "optimizer_experiment_config.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_adaptive_script": str(ADAPTIVE_SCRIPT),
                "pde": "-div(grad(Phi)/(2*mu*(1+beta*|grad(Phi)|^alpha)^(1/alpha))) = 0",
                "domain": "[0,1] x [0,1] by default, configurable through KAN_PINN_* bounds",
                "optimizer_stages": [asdict(stage) for stage in stages],
                "controller": asdict(ctrl),
                "training": asdict(trn),
                "material": asdict(mat),
                "geometry": asdict(geo),
                "boundary_conditions": asdict(bc),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model = AdaptiveGridRBFKANPINN(
        hidden=trn.hidden,
        n_basis=trn.n_basis,
        width_scale=trn.grid_width_scale,
        min_width=trn.grid_min_width,
    ).to(device)
    model.configure_boundary_ansatz(geo, bc, trn)

    model, best_epoch, best_val, lhist, lpde_hist, lenergy_hist, lbc_hist, val_hist, collocation_counts, train_boundary_points = train_with_optimizer_stages(
        model, mat, geo, bc, trn, stages, ctrl, outdir, device, resume=resume_training
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
    adaptive.save_plots(
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
    exact_diag = adaptive.exact_solution_diagnostics(model, geo, bc, device, trn.pointwise_nx, trn.pointwise_ny)
    print(
        "Exact linear solution check: "
        f"rel_L2={exact_diag['relative_l2']:.5e}, mean|err|={exact_diag['mean_abs']:.5e}, "
        f"max|err|={exact_diag['max_abs']:.5e}"
    )
    print(f"Training complete. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
