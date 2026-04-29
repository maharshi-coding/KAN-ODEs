# Physics-Only KAN-PINN: V-Notch Strain-Limiting PDE

## Implementation Plan
- [x] Audit existing Python code against the target PDE + BC formulation.
- [x] Enforce target boundary formulation: Dirichlet on `Γ1–Γ5` with `Γ5 = 0`.
- [x] Improve training observability with continuous compact progress logs and periodic detailed diagnostics.
- [x] Improve checkpointing/resume reliability with explicit best/last checkpoint artifacts and status metadata.
- [x] Run a real reduced verification training run and confirm artifact generation.
- [x] Document verification evidence, limitations, and remaining risks.

## Audit Notes
- Current script computes Eq. 40 PDE residual via autograd and excludes notch void from interior sampling.
- Current script still defaults `Γ5` to natural BC mode (`KAN_PINN_G5_MODE=natural`) and only conditionally applies Dirichlet-zero.
- Training logs are useful but incomplete for the requested monitoring fields (LR, grad norm, best/new-best status, periodic boundary and residual diagnostics).
- Checkpointing currently stores mixed best/last state in a single `best_checkpoint.pt`; explicit split will improve reliability and transparency.

## Review
- What was wrong:
  - `Γ5` enforcement was optional and defaulted to natural BC mode; target formulation requires `Γ5=0` Dirichlet.
  - Training logs did not expose enough runtime diagnostics (gradient norm, LR, best/new-best status, interval PDE/BC/tip diagnostics).
  - Checkpoint flow used one primary file and did not cleanly separate `best` and `last` artifacts.
  - Config validation was weak for invalid physics/training geometry values.
- What code was modified:
  - `PDE examples/StrainLimiting_KAN_PINN.py`
  - Added strict `canonical_g5_mode(...)` and `validate_configuration(...)`.
  - Updated Dirichlet handling so `Γ5a/Γ5b` are always zero-Dirichlet in this formulation.
  - Added richer training log fields and periodic detailed diagnostics (PDE stats, tip-region diagnostics, boundary diagnostics by label).
  - Added explicit `best_checkpoint.pt` + `last_checkpoint.pt` save/resume behavior and training summary output.
  - Added environment-driven diagnostics controls (`KAN_PINN_DETAILED_DIAG_EVERY`, `KAN_PINN_DIAGNOSTIC_SAMPLES`) and stronger defaults.
  - Added writable Matplotlib config path setup to avoid cache-dir warnings in this environment.
- What code was rewritten:
  - The `train_model(...)` routine was substantially refactored into clearer stage execution with shared per-epoch logic to avoid duplicated Adam/finetune loops and to centralize validation/checkpoint/logging behavior.
- Why it was changed:
  - To make the implementation match the requested PDE+BC formulation exactly.
  - To improve experiment reliability/traceability and make live monitoring genuinely useful during long physics-only training.
  - To make resume/checkpoint semantics explicit and robust.
- How it was verified:
  - Syntax check: `python3 -m py_compile 'PDE examples/StrainLimiting_KAN_PINN.py'`
  - Real reduced training run:
    - Run ID: `codex_verify_smoke` (12 epochs, physics-only).
    - Confirmed continuous logs include epoch, losses, validation, LR, grad norm, ETA, best/new-best, BC diagnostics, PDE residual stats, tip diagnostics, and checkpoint status.
    - Confirmed checkpoints produced: `best_checkpoint.pt`, `last_checkpoint.pt`.
    - Confirmed artifacts produced: `loss_history.png`, `phi_field.png`, `grad_phi_field.png`, `tau_eq_field.png`, `pde_residual_field.png`, `tau_eq_reference_line.png`, `field_diagnostics.npz`, `reference_line_diagnostics.npz`, `reference_line_tau_eq.csv`, `run_diagnostics.json`.
    - Confirmed `run_diagnostics.json` records `g5_mode=dirichlet_zero` and boundary roles with `Γ5a/Γ5b = Dirichlet-zero`.
  - Quick post-polish run:
    - Run ID: `codex_verify_smoke_quick` (1 epoch) to confirm no runtime regressions after final warning fixes.
- Remaining limitations/risks:
  - Verification was intentionally reduced-size for fast turnaround; physical quality is not yet converged.
  - Tip-related auxiliary losses are heuristic stabilizers and may need coefficient tuning for best long-run physics quality.
  - CPU-only verification here is correct but slower; full-scale experiments should run on CUDA with larger epoch budgets.
