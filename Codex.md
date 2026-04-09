You are Codex operating as a senior computational mechanics engineer, PINN specialist, and Python refactoring agent.

I already have a Python codebase for a strain-limiting KAN/PINN solver for the V-notch anti-plane shear problem. Your task is to directly modify the existing code so the training follows the correct physics and the final solution matches the intended nonlinear notch-tip behavior.

You must work in execution mode:
- inspect the codebase first
- identify the exact physics and implementation mistakes
- make concrete code edits
- run/check the relevant scripts if possible
- save improved diagnostics
- summarize the fixes clearly

Do not give high-level advice only.
Do not leave TODOs.
Do not stop at “I found the issue.”
Fix it in code.

Physics target:
The governing equation is

    -div( grad(Phi) / ( 2*mu*(1 + beta*|grad(Phi)|^alpha)^(1/alpha) ) ) = 0   in Ω

with Dirichlet data Phi = g(x1, x2) only where physically prescribed on the correct boundaries.

Problem context:
My current results are wrong in a physically meaningful way:
- training is boundary-dominated
- the learned Phi field is mostly a smooth interpolation between outer boundaries
- the nonlinear PDE is under-enforced
- the equivalent stress along the reference line stays near zero for most of the domain and spikes incorrectly near the far end
- the notch-tip physics is not being learned properly

Most likely failure modes to audit and fix:
1. Outer Dirichlet boundary loss is too dominant.
2. The notch faces Γ5 are likely being treated as hard zero-Dirichlet boundaries, which is likely incorrect for the formulation and suppresses crack-tip physics.
3. Crack-tip behavior is not sufficiently sampled, regularized, or resolved.
4. PDE residual weighting/scheduling is poor and allows the network to settle into a smooth boundary interpolation.
5. Stress computation, gradient scaling, or residual construction may be numerically unstable or inconsistently scaled.

What you must do in order:

STEP 1 — Audit the implementation
Inspect the current code carefully and identify:
- geometry/domain construction
- notch masking / void handling
- boundary labeling for Γ1, Γ2, Γ3, Γ4, Γ5
- how Dirichlet targets are assigned
- whether Γ5 is included in boundary_loss or dirichlet_target incorrectly
- PDE residual implementation
- gradient/stress computation from Phi
- equivalent stress computation
- training curriculum / loss schedules
- optimizer / LR scheduler
- point sampling strategy
- any disabled tip-loss or local-refinement logic

Then explicitly state:
- what the code is doing right now
- what is physically incorrect
- what must be changed

STEP 2 — Fix the boundary-condition physics
Implement the physically correct treatment of the V-notch problem:
- preserve only the intended prescribed Dirichlet conditions on the correct outer boundaries
- remove any incorrect hard Dirichlet enforcement on notch faces Γ5 unless the paper formulation explicitly proves otherwise
- if Γ5 should be traction-free / natural, enforce it accordingly through the PDE formulation rather than forcing Phi = 0
- make the boundary labeling explicit and auditable
- add per-boundary diagnostics so I can verify Γ1, Γ2, Γ3, Γ4, Γ5 are each being handled correctly

STEP 3 — Fix PDE dominance and training balance
Modify the training so it learns the nonlinear PDE instead of only fitting edges:
- rebalance BC and PDE losses
- redesign the curriculum if needed
- avoid a training schedule where the network overfits boundaries first and never recovers physically
- keep training stable while making PDE residual matter throughout training
- add safe norm computations and stabilization where needed for |grad(Phi)|
- verify that beta, alpha, mu, and all scaling are used correctly in the residual

STEP 4 — Add notch-tip-focused learning
Make the crack-tip region physically resolvable:
- increase collocation density near the notch tip
- add local refinement or adaptive sampling near Γ5 and the notch tip
- improve reference-line sampling
- add optional physically justified tip-focused regularization or guidance if needed
- do not add fake hacks that only make the plots look better; the solution must become physically more correct

STEP 5 — Improve diagnostics and validation
Add and save:
- total loss
- PDE loss
- BC loss
- validation loss
- per-boundary losses
- Phi field
- |grad(Phi)| field
- equivalent stress field
- equivalent stress along the reference line
- PDE residual field
- near-tip vs far-field stress statistics

Also log:
- whether Γ5 is treated as Dirichlet, Neumann, or excluded from Dirichlet loss
- collocation counts by region
- max/mean |grad(Phi)| near the tip
- max/mean PDE residual near the tip and in the far field

STEP 6 — Make code edits directly
Requirements for code editing:
- modify the existing Python files directly
- preserve project structure where possible
- keep changes clean and production-quality
- add concise code comments only where they help explain important physics fixes
- add config flags/toggles for old vs corrected boundary handling and sampling so behavior can be compared

STEP 7 — Verify outcome
After edits, run the relevant training/evaluation flow if available and confirm whether:
- the Phi field is no longer just a smooth left-to-right interpolation
- the PDE residual is genuinely learned
- the notch-tip region develops physically meaningful stress concentration
- the reference-line stress curve now rises in a physically sensible way toward the tip
- the final result looks like a nonlinear strain-limiting notch-tip solution instead of a boundary-dominated harmonic-like field

Output format you must follow:
1. Audit summary
   - what was wrong
   - where in the code it was wrong
2. Code changes made
   - file-by-file summary
3. Physics rationale
   - why each change matters physically
4. Validation results
   - whether the corrected training is now behaving properly
5. Remaining risks or limitations
   - only if truly necessary

Important constraints:
- Do not just optimize for lower losses.
- Do not make cosmetic changes only.
- Do not silently preserve incorrect notch-face treatment.
- Do not assume the current implementation is correct.
- Verify the formulation against the existing code behavior before editing.
- Prioritize physical correctness, stable training, and interpretable diagnostics.

Success criteria:
- boundary fitting no longer overwhelms physics
- Γ5 treatment is corrected
- stress concentration forms near the notch tip
- reference-line stress becomes physically meaningful
- the corrected model behaves like the intended nonlinear strain-limiting V-notch solution

Now start by auditing the current implementation and then make the necessary code changes systematically.