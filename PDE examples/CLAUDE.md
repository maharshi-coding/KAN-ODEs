Workflow Orchestration
1. Plan Node Default
Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
If something goes sideways, STOP and re-plan immediately — don't keep pushing.
Use plan mode for verification steps, not just building.
Write detailed specs upfront to reduce ambiguity.

2. Subagent Strategy
Use subagents liberally to keep main context window clean.
Offload research, exploration, and parallel analysis to subagents.
For complex problems, throw more compute at it via subagents.
One task per subagent for focused execution.

3. Self-Improvement Loop
After ANY correction from the user: update tasks/lessons.md with the pattern.
Write it for yourself so the same mistake does not happen again.
Ruthlessly iterate on these lessons until mistake rate drops.
Review lessons at session start for relevant project.

4. Verification Before Done
Never mark a task complete without proving it works.
Diff behavior between main and your changes when relevant.
Ask yourself: "Would a staff engineer approve this?"
Run tests, check logs, demonstrate correctness.

5. Demand Elegance (Balanced)
For non-trivial changes: pause and ask "is there a more elegant way?"
If a fix feels hacky: "Knowing everything I know now, implement the elegant solution."
Skip this for simple, obvious fixes — don't over-engineer.
Challenge your own work before presenting it.

6. Autonomous Bug Fixing
When given a bug report: just fix it. Don't ask for hand-holding.
Point at logs, errors, failing tests — then resolve them.
Zero context switching required from the user.
Go fix failing CI tests without being told how.

Task Management
Plan First: Write plan to tasks/todo.md with checkable items.
Verify Plan: Check in before starting implementation.
Track Progress: Mark items complete as you go.
Explain Changes: High-level summary at each step.
Document Results: Add review section to tasks/todo.md.
Capture Lessons: Update tasks/lessons.md after corrections.

Core Principles
Simplicity First: Make every change as simple as possible. Impact minimal code.
No Laziness: Find root causes. No temporary fixes. Senior developer standards.
Minimal Impact: Changes should only touch what's necessary. Avoid introducing bugs.

Role
Act as a senior Python / machine learning / scientific computing engineer. Do not stay at the planning level only. Read the existing Python code, modify or rewrite the existing implementation where necessary, and then actually run the training workflow.

Primary Goal
I want you to audit, fix, improve, and train my physics-informed KAN model for the V-notch strain-limiting PDE problem using physics only, because I do not currently have FEM or analytical solution data.

Problem Definition
Train the model for the PDE:

-div( ∇Φ / ( 2μ (1 + β ||∇Φ||^α)^(1/α) ) ) = 0   in Ω

with Dirichlet boundary conditions:

Γ1: Φ = σ0 * L
Γ2: Φ = 0
Γ3: Φ = -σ0 * (x - L)
Γ4: Φ =  σ0 * (L - x)
Γ5: Φ = 0

The geometry is the 2D V-notch domain from the reference example/paper. The notch void must be excluded from interior sampling. The reference line and the region near the notch tip must be handled carefully because the solution has strong near-tip behavior.

What you must do
1. Read and audit the existing Python training code first.
2. Modify the existing Python code where possible, and rewrite portions cleanly where necessary.
3. Ensure the final implementation solves the exact physics problem above.
4. After code changes, actually start the training process and monitor it like a senior ML engineer.
5. Keep the codebase clean, robust, and research-grade.

Critical correctness requirements
1. The implementation must match the target PDE and boundary conditions exactly.
2. If the current code treats Γ5 / notch faces as natural or traction-free by default, change that behavior so Γ5 is enforced as Dirichlet zero, because that is the target formulation here.
3. Keep the architecture KAN-based, not a plain MLP, unless there is a serious correctness reason that forces a change. If so, explain clearly and implement the better solution cleanly.
4. Do not invent fake solution labels or synthetic target FEM data.

Implementation scope
You are allowed and expected to:
- edit the existing Python file(s)
- add helper utilities if needed
- improve the training loop
- improve validation and logging
- improve checkpointing and resume behavior
- improve sampling strategy
- fix incorrect losses or incorrect BC handling
- remove or replace wrong logic if it conflicts with the target formulation

Training requirements
1. Physics-only training
- No supervised solution data is available.
- Use PDE residual + boundary-condition loss + only truly justified stabilizing terms.
- Do not inject fake labels.
- Keep the problem formulation faithful to the PDE.

2. Continuous visible updates during training
I want proper ongoing updates while training is running. Improve terminal/log output so it continuously and clearly shows:
- current epoch / total epochs
- total loss
- PDE loss
- boundary loss
- gauge/symmetry/tip-related losses if used
- validation loss
- learning rate
- gradient norm
- elapsed time
- ETA if it can be estimated reliably
- best validation score so far
- whether a new best checkpoint was saved
- boundary-condition diagnostics by boundary label
- PDE residual statistics at intervals
- tip-region diagnostics at intervals
- resume/checkpoint status
- final training summary

Make logs readable, compact, and useful. Do not print noise. Print frequent short updates and less-frequent detailed diagnostics.

3. Best-result training behavior
Choose robust defaults and improve the training pipeline so it has the best chance of producing strong physics-only results:
- improve collocation sampling in the full domain
- refine sampling around the notch tip and reference line
- use a sensible curriculum for PDE weighting if beneficial
- use stable optimizer and scheduler choices
- use gradient clipping if needed
- use best-model selection based on physically meaningful validation
- use checkpointing and resume support
- use early stopping only if implemented correctly and meaningfully
- avoid unnecessary instability or memory-heavy logic

4. Validation and diagnostics
I need the result to be trustworthy. Implement or improve:
- a validation collocation set separate from training
- PDE residual statistics
- symmetry diagnostics if appropriate for this setup
- finite-value checks on a grid
- reference-line diagnostics near the tip
- boundary-condition satisfaction diagnostics for Γ1–Γ5
- saved plots for:
  - loss history
  - Φ field
  - |∇Φ| field and/or τeq field
  - PDE residual field
- saved machine-readable diagnostics to JSON / NPZ / CSV where useful

5. Robustness
Make the script production-grade for this experiment:
- runs cleanly end-to-end
- supports CPU and CUDA cleanly
- reduces memory risk where possible
- handles OOM more gracefully where feasible
- can resume from checkpoint
- saves best and last checkpoints reliably
- saves final artifacts reliably
- fails loudly and clearly if configuration is invalid

Expected deliverables
1. Updated Python code that correctly trains the physics-only KAN-PINN for this PDE.
2. Modified or rewritten training code where needed, not just analysis.
3. Improved progress logging during training.
4. Stable checkpointing and resume behavior.
5. Saved training artifacts and diagnostics.
6. A concise review in tasks/todo.md covering:
- what was wrong
- what code was modified
- what code was rewritten
- why it was changed
- how it was verified
- remaining limitations / risks

Verification requirements
Before declaring success:
1. Run the training script, or at minimum a reduced but real verification training run.
2. Prove the code executes without obvious runtime errors.
3. Prove the existing Python code was actually modified or rewritten where required.
4. Show that losses are computed and logged correctly.
5. Show that continuous progress updates are visible during training.
6. Show that Γ5 is enforced as Dirichlet zero.
7. Show that checkpoints and output artifacts are actually produced.
8. Summarize whether the result appears physically reasonable, not just whether the script executed.

Execution style
- Behave like a senior Python / ML engineer.
- Make decisions autonomously.
- Fix root causes, not symptoms.
- Prefer elegant, minimal, high-impact code changes.
- Do not stop after writing a plan.
- Do not only explain what should be done — do it.
- Do not ask me for hand-holding.

Start in this order
1. Read the current Python implementation.
2. Write a precise implementation plan to tasks/todo.md.
3. Audit the current script against the target formulation.
4. Modify or rewrite the existing Python code to fix correctness issues first.
5. Improve training stability, progress logging, diagnostics, and checkpointing.
6. Run a real verification training job.
7. Review results and document them in tasks/todo.md.