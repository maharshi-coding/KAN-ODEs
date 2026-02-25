# Task: Implement KAN-PINN in Python for Strain-Limiting PDE (Equation 40)

You are given:
- A research paper PDF (strain-limiting elasticity)
- Image of Equation (40)
- Image of the computational domain (notched geometry)
- A previous Julia KAN-PINN implementation

Your task is to write a complete Python program (PyTorch-based) that solves the PDE in Equation (40) using a KAN-based PINN.

---

## 1. Mathematical Model (from the paper)

Implement exactly:

div( ∇Φ / [ 2μ (1 + β |∇Φ|^α)^(1/α) ] ) = 0

Ensure:
- Correct gradient ∇Φ via autograd
- Correct norm |∇Φ|
- Correct nonlinear flux
- Correct divergence
- PDE residual enforced in interior

---

## 2. Geometry (from diagram)

Domain:
- Rectangle [xmin, xmax] × [ymin, ymax]
- V-shaped notch with:
  - Tip (x0, y0)
  - Opening angle θ
  - Length L

Implement:
- point_in_notch_void(x,y)
- sample_interior_points (uniform + refined near tip)
- sample_boundary_points:
  Γ1 (left), Γ2 (right), Γ3 (bottom), Γ4 (top)
  Γ5a, Γ5b (notch faces)

Exclude notch interior points.

---

## 3. Boundary Conditions (from Table 3)

Dirichlet BCs:
- Γ1: Φ = σ₀ L
- Γ2: Φ = 0
- Γ3: Φ = −σ₀ (x − L)
- Γ4: Φ = σ₀ (L − x)

Natural BC:
- Γ5a, Γ5b (no Dirichlet penalty)

Implement BC loss only on Γ1–Γ4.

---

## 4. Neural Network: KAN

Implement a Kolmogorov–Arnold Network using:
- Input: (x,y)
- Hidden KAN layers with spline or Gaussian basis
- Trainable coefficients
- Output: Φ(x,y)

Do NOT use standard MLP.
Do NOT hardcode PDE inside network.

---

## 5. Loss Function

Total loss:
L = L_pde + λ_bc * L_bc + λ_gauge * L_gauge

Where:
- L_pde = mean( residual^2 )
- L_bc = mean Dirichlet error on Γ1–Γ4
- L_gauge = Φ(0,0)^2

Add singular weighting:
Multiply PDE residual by:
w(x) = 1 / (dist_to_tip + ε)

---

## 6. Training Strategy

Use:
- Adam optimizer
- Learning rate schedule
- Gradient clipping
- Early stopping
- Validation set

Refinement:
- Interior points: uniform + refined near notch tip
- Increase sampling near singularity

---

## 7. Outputs

Save:
- Loss history plot
- Φ(x,y) field heatmap
- |∇Φ| along a line approaching the notch tip

Print:
- PDE residual stats
- Symmetry check
- Near-tip vs far gradient ratio

---

## 8. Constraints

IMPORTANT:
- Use Python + PyTorch only
- Use autograd (no finite differences)
- Single script
- No extra files
- No simplification of PDE
- No MLP

---

## Final Goal

Produce a Python KAN-based PINN that:
- Solves Equation (40)
- Uses the notch geometry
- Enforces BCs correctly
- Shows gradient concentration near the notch tip
- Produces physically meaningful strain-limiting behavior

---

## Output Required

Return:
- Full Python code
- Ready to run
- Well-commented