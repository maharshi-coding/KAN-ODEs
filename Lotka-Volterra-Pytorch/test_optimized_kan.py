"""
Quick validation test for the optimized KAN implementation.
Runs a very short training to verify all components work correctly.
"""

import torch
import numpy as np
import sys
import os

sys.path.append("efficient_kan/")
import efficientkan
from optimized_kan import OptimizedKAN, HybridKAN_MLP, DataNormalizer

print("\n" + "="*70)
print("QUICK VALIDATION TEST - Optimized KAN Implementation")
print("="*70 + "\n")

# Test 1: Create models
print("[Test 1] Creating models...")
try:
    # Baseline
    baseline = efficientkan.KAN(layers_hidden=[2, 8, 2], grid_size=5)
    print("  ✓ Baseline KAN created")
    
    # Optimized
    optimized = OptimizedKAN(
        layers_hidden=[2, 8, 2],
        grid_size=5,
        l2_lambda=0.0001,
        smoothness_lambda=0.001,
    )
    print("  ✓ Optimized KAN created")
    
    # Hybrid
    hybrid = HybridKAN_MLP(
        kan_layers=[2, 8, 2],
        mlp_hidden_dims=[8, 4],
        grid_size=5,
    )
    print("  ✓ Hybrid KAN+MLP created")
except Exception as e:
    print(f"  ✗ Error creating models: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n[Test 2] Testing forward pass...")
try:
    x = torch.randn(10, 2)
    
    y_baseline = baseline(x)
    assert y_baseline.shape == (10, 2), f"Expected (10, 2), got {y_baseline.shape}"
    print("  ✓ Baseline forward pass: OK")
    
    y_optimized = optimized(x)
    assert y_optimized.shape == (10, 2), f"Expected (10, 2), got {y_optimized.shape}"
    print("  ✓ Optimized forward pass: OK")
    
    y_hybrid = hybrid(x)
    assert y_hybrid.shape == (10, 2), f"Expected (10, 2), got {y_hybrid.shape}"
    print("  ✓ Hybrid forward pass: OK")
except Exception as e:
    print(f"  ✗ Error in forward pass: {e}")
    sys.exit(1)

# Test 3: Regularization
print("\n[Test 3] Testing regularization...")
try:
    reg_loss = optimized.regularization_loss()
    assert reg_loss.item() > 0, "Regularization loss should be positive"
    print(f"  ✓ Regularization loss: {reg_loss.item():.6f}")
    
    reg_loss_hybrid = hybrid.regularization_loss()
    assert reg_loss_hybrid.item() > 0, "Hybrid regularization loss should be positive"
    print(f"  ✓ Hybrid regularization loss: {reg_loss_hybrid.item():.6f}")
except Exception as e:
    print(f"  ✗ Error in regularization: {e}")
    sys.exit(1)

# Test 4: Backward pass
print("\n[Test 4] Testing backward pass...")
try:
    x = torch.randn(10, 2, requires_grad=True)
    y_target = torch.randn(10, 2)
    
    # Optimized model
    y_pred = optimized(x)
    loss = torch.mean((y_pred - y_target) ** 2)
    loss += optimized.regularization_loss()
    loss.backward()
    
    # Check gradients exist
    grad_found = False
    for param in optimized.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_found = True
            break
    assert grad_found, "No gradients computed"
    print("  ✓ Backward pass: OK (gradients computed)")
except Exception as e:
    print(f"  ✗ Error in backward pass: {e}")
    sys.exit(1)

# Test 5: Data normalizer
print("\n[Test 5] Testing data normalizer...")
try:
    normalizer = DataNormalizer(data_range=(-1, 1))
    
    # Fit to data
    data = np.random.randn(100, 2)
    normalizer.fit_input(data)
    normalizer.fit_output(data)
    
    # Normalize and denormalize
    data_tensor = torch.tensor(data, dtype=torch.float32)
    normalized = normalizer.normalize_input(data_tensor)
    denormalized = normalizer.denormalize_input(normalized)
    
    # Check round-trip
    error = torch.mean((data_tensor - denormalized) ** 2).item()
    assert error < 1e-5, f"Round-trip error too large: {error}"
    print(f"  ✓ Normalization round-trip error: {error:.2e}")
except Exception as e:
    print(f"  ✗ Error in normalizer: {e}")
    sys.exit(1)

# Test 6: Pruning
print("\n[Test 6] Testing model pruning...")
try:
    # Create small random model
    model = OptimizedKAN(
        layers_hidden=[2, 4, 2],
        grid_size=3,
    )
    
    # Train briefly to get some weights
    x = torch.randn(20, 2)
    y = torch.randn(20, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(10):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
    
    # Prune
    pruned_count = model.prune_model(threshold=0.01)
    print(f"  ✓ Pruned {pruned_count} edges")
except Exception as e:
    print(f"  ✗ Error in pruning: {e}")
    sys.exit(1)

# Test 7: ODE integration (quick)
print("\n[Test 7] Testing ODE integration...")
try:
    from torchdiffeq import odeint as torchodeint
    
    model = optimized
    X0 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    t = torch.linspace(0, 1, 10)
    
    def calDeriv(t, X):
        return model(X)
    
    pred = torchodeint(calDeriv, X0, t, method='dopri5', rtol=1e-5, atol=1e-7)
    assert pred.shape == (10, 1, 2), f"Expected (10, 1, 2), got {pred.shape}"
    print(f"  ✓ ODE integration: OK (shape {pred.shape})")
except Exception as e:
    print(f"  ✗ Error in ODE integration: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL VALIDATION TESTS PASSED")
print("="*70 + "\n")
print("The optimized KAN implementation is ready to use!")
print("Run 'python optimized_training.py' for full training and evaluation.\n")
