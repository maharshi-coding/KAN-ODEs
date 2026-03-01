#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch


def load_training_module(script_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("sl_mod", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bool_mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KAN-PINN checkpoint with strict PASS/FAIL gates.")
    parser.add_argument("--root", default="results_strainlimiting_python", help="Results root folder")
    parser.add_argument("--run", default=None, help="Run name. If omitted, uses latest_run.txt")
    parser.add_argument("--checkpoint", default="best_checkpoint.pt", help="Checkpoint filename")

    parser.add_argument("--min-ratio", type=float, default=1.10, help="Minimum acceptable near/far tau_eq ratio")
    parser.add_argument("--max-best-val", type=float, default=32.0, help="Maximum acceptable best validation loss")
    parser.add_argument("--max-sym-mean", type=float, default=0.15, help="Maximum acceptable mean symmetry error")
    parser.add_argument("--max-pde-mean", type=float, default=4.0e2, help="Maximum acceptable mean absolute PDE residual")
    parser.add_argument("--max-bad-outside", type=int, default=0, help="Maximum allowed non-finite grid values outside notch")

    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    script_path = here / "StrainLimiting_KAN_PINN.py"
    mod = load_training_module(script_path)

    root = (here / args.root).resolve()
    if args.run is None:
        latest = root / "latest_run.txt"
        if not latest.exists():
            raise FileNotFoundError(f"Could not find latest run file: {latest}")
        run = latest.read_text().strip()
    else:
        run = args.run

    ckpt_path = root / run / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    best_val = float(ckpt.get("best_val", np.nan))
    best_epoch = int(ckpt.get("best_epoch", 0))
    completed_epochs = int(ckpt.get("completed_epochs", len(ckpt.get("loss_total", []))))

    trn = mod.TrainParams()
    geo = mod.GeometryParams()
    mat = mod.MaterialParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_key = "best_model_state" if "best_model_state" in ckpt else ("model_state" if "model_state" in ckpt else "last_model_state")
    if state_key not in ckpt:
        raise KeyError("No model state found in checkpoint.")

    model = mod.KANPINN(hidden=trn.hidden, n_basis=trn.n_basis).to(device)
    model.load_state_dict(ckpt[state_key])
    model.eval()

    rstats = mod.residual_statistics(model, mat, geo, device)
    sstats = mod.symmetry_error(model, geo, device)
    tip = mod.tip_gradient_indicator(model, geo, device)
    gstats = mod.grid_finite_check(model, geo, device)

    checks = {
        "ratio": tip["ratio"] >= args.min_ratio,
        "best_val": best_val <= args.max_best_val,
        "sym_mean": sstats["mean_abs"] <= args.max_sym_mean,
        "pde_mean_abs": rstats["mean_abs"] <= args.max_pde_mean,
        "finite_outside": gstats["bad_outside"] <= args.max_bad_outside,
    }

    overall = all(checks.values())

    print("=== KAN-PINN PASS/FAIL REPORT ===")
    print(f"run: {run}")
    print(f"checkpoint: {ckpt_path}")
    print(f"state_used: {state_key}")
    print(f"completed_epochs: {completed_epochs}")
    print(f"best_epoch: {best_epoch}")
    print()

    print(f"[{bool_mark(checks['ratio'])}] tip ratio      : {tip['ratio']:.6f}  (threshold >= {args.min_ratio})")
    print(f"[{bool_mark(checks['best_val'])}] best val       : {best_val:.6f}  (threshold <= {args.max_best_val})")
    print(f"[{bool_mark(checks['sym_mean'])}] symmetry mean  : {sstats['mean_abs']:.6e}  (threshold <= {args.max_sym_mean})")
    print(f"[{bool_mark(checks['pde_mean_abs'])}] PDE mean|r|   : {rstats['mean_abs']:.6e}  (threshold <= {args.max_pde_mean})")
    print(f"[{bool_mark(checks['finite_outside'])}] finite check    : bad={gstats['bad_outside']}/{gstats['outside_total']}  (threshold <= {args.max_bad_outside})")
    print()

    print("Detail metrics:")
    print(f"  PDE residual: mean|r|={rstats['mean_abs']:.6e}, rms={rstats['rms']:.6e}, max|r|={rstats['max_abs']:.6e}")
    print(f"  Symmetry    : mean|ΔΦ|={sstats['mean_abs']:.6e}, max|ΔΦ|={sstats['max_abs']:.6e}, pairs={sstats['n_pairs']}")
    print(f"  Tip stress  : near={tip['near_mean']:.6e}, far={tip['far_mean']:.6e}, near/far={tip['ratio']:.6f}")

    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
