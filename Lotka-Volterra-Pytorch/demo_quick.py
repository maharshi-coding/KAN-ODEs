"""
Quick Demo - Optimized KAN-ODE Training
Runs a shortened version of the full training for demonstration purposes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import sys
import os

sys.path.append("efficient_kan/")
import efficientkan
from optimized_kan import OptimizedKAN, DataNormalizer

# Configuration
class DemoConfig:
    # ODE Parameters
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
    x0, y0 = 1.0, 1.0
    tf, tf_learn = 14.0, 3.5
    N_t_train = 35
    N_t = int(35 * tf / tf_learn)
    
    # Training (reduced for demo)
    num_epochs = 1000  # Reduced from 5000 (optimized model needs more epochs)
    baseline_lr = 2e-3
    optimized_lr = 1e-3
    
    # Architecture
    baseline_architecture = [2, 10, 2]
    optimized_architecture = [2, 8, 2]
    grid_size = 5
    
    # Regularization
    l2_lambda = 0.0001
    smoothness_lambda = 0.001
    
    # Solver
    rtol, atol = 1e-6, 1e-8
    method = 'dopri5'

def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    x, y = X
    return [alpha*x - beta*x*y, delta*x*y - gamma*y]

def train_quick_demo():
    print("\n" + "="*70)
    print("KAN-ODE QUICK DEMO - Baseline vs Optimized Comparison")
    print("="*70 + "\n")
    
    config = DemoConfig()
    
    # Generate data
    print("Generating data...")
    X0 = np.array([config.x0, config.y0])
    t = np.linspace(0, config.tf, config.N_t)
    soln_arr = scipy.integrate.odeint(
        pred_prey_deriv, X0, t,
        args=(config.alpha, config.beta, config.delta, config.gamma)
    )
    
    t_train = np.linspace(0, config.tf_learn, config.N_t_train)
    soln_train = soln_arr[:config.N_t_train]
    
    # Convert to tensors
    X0_tensor = torch.tensor(X0, dtype=torch.float32).unsqueeze(0)
    soln_train_tensor = torch.tensor(soln_train, dtype=torch.float32)
    t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
    t_full_tensor = torch.tensor(t, dtype=torch.float32)
    
    # Create output directory
    os.makedirs("plots/demo", exist_ok=True)
    
    # ========================================================================
    # Train Baseline Model
    # ========================================================================
    print("\n[1/2] Training Baseline KAN...")
    baseline = efficientkan.KAN(
        layers_hidden=config.baseline_architecture,
        grid_size=config.grid_size
    )
    optimizer_baseline = torch.optim.Adam(baseline.parameters(), lr=config.baseline_lr)
    
    baseline_losses = []
    for epoch in tqdm(range(config.num_epochs), desc="Baseline"):
        optimizer_baseline.zero_grad()
        
        def calDeriv_baseline(t, X):
            return baseline(X)
        
        pred = torchodeint(calDeriv_baseline, X0_tensor, t_train_tensor,
                          method=config.method, rtol=config.rtol, atol=config.atol)
        loss = torch.mean((pred[:, 0, :] - soln_train_tensor) ** 2)
        loss.backward()
        optimizer_baseline.step()
        baseline_losses.append(loss.item())
    
    print(f"  Final loss: {baseline_losses[-1]:.6f}")
    
    # ========================================================================
    # Train Optimized Model
    # ========================================================================
    print("\n[2/2] Training Optimized KAN...")
    optimized = OptimizedKAN(
        layers_hidden=config.optimized_architecture,
        grid_size=config.grid_size,
        l2_lambda=config.l2_lambda,
        smoothness_lambda=config.smoothness_lambda,
    )
    optimizer_optimized = torch.optim.Adam(optimized.parameters(), lr=config.optimized_lr)
    
    # Setup normalizer
    normalizer = DataNormalizer(data_range=(-1, 1))
    normalizer.fit_input(soln_arr)
    normalizer.fit_output(soln_arr)
    
    optimized_losses = []
    for epoch in tqdm(range(config.num_epochs), desc="Optimized"):
        optimizer_optimized.zero_grad()
        
        def calDeriv_optimized(t, X):
            X_norm = normalizer.normalize_input(X)
            dXdt_norm = optimized(X_norm)
            dXdt = dXdt_norm * (normalizer.output_max - normalizer.output_min) / (
                normalizer.input_max - normalizer.input_min)
            return dXdt
        
        pred = torchodeint(calDeriv_optimized, X0_tensor, t_train_tensor,
                          method=config.method, rtol=config.rtol, atol=config.atol)
        loss_data = torch.mean((pred[:, 0, :] - soln_train_tensor) ** 2)
        loss_reg = optimized.regularization_loss()
        loss = loss_data + loss_reg
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(optimized.parameters(), max_norm=1.0)
        
        optimizer_optimized.step()
        optimized_losses.append(loss_data.item())
    
    print(f"  Final loss: {optimized_losses[-1]:.6f}")
    
    # ========================================================================
    # Evaluate and Visualize
    # ========================================================================
    print("\nGenerating predictions...")
    
    with torch.no_grad():
        pred_baseline = torchodeint(calDeriv_baseline, X0_tensor, t_full_tensor,
                                   method=config.method, rtol=config.rtol, atol=config.atol)
        pred_baseline = pred_baseline[:, 0, :].numpy()
        
        pred_optimized = torchodeint(calDeriv_optimized, X0_tensor, t_full_tensor,
                                    method=config.method, rtol=config.rtol, atol=config.atol)
        pred_optimized = pred_optimized[:, 0, :].numpy()
    
    # Compute metrics
    mse_baseline = np.mean((pred_baseline - soln_arr) ** 2)
    mse_optimized = np.mean((pred_optimized - soln_arr) ** 2)
    improvement = ((mse_baseline - mse_optimized) / mse_baseline) * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline MSE:  {mse_baseline:.6f}")
    print(f"Optimized MSE: {mse_optimized:.6f}")
    if improvement > 0:
        print(f"Improvement:   {improvement:+.2f}%")
    else:
        print(f"Change:        {improvement:+.2f}%")
        print("\nNote: Regularized models may need more epochs to converge.")
        print("Run full training (5000 epochs) for best results.")
    print("="*70 + "\n")
    
    # Plot 1: Loss curves
    plt.figure(figsize=(10, 6))
    plt.semilogy(baseline_losses, label='Baseline', linewidth=2)
    plt.semilogy(optimized_losses, label='Optimized', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/demo/loss_comparison.png', dpi=200)
    print("✓ Saved: plots/demo/loss_comparison.png")
    plt.close()
    
    # Plot 2: Trajectories comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Baseline
    axes[0].plot(t, soln_arr[:, 0], 'g-', label='True x', linewidth=2)
    axes[0].plot(t, soln_arr[:, 1], 'b-', label='True y', linewidth=2)
    axes[0].plot(t, pred_baseline[:, 0], 'g--', label='Pred x', linewidth=2, alpha=0.7)
    axes[0].plot(t, pred_baseline[:, 1], 'b--', label='Pred y', linewidth=2, alpha=0.7)
    axes[0].axvline(config.tf_learn, color='r', linestyle=':', label='Training horizon')
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Population', fontsize=11)
    axes[0].set_title(f'Baseline (MSE: {mse_baseline:.4f})', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Optimized
    axes[1].plot(t, soln_arr[:, 0], 'g-', label='True x', linewidth=2)
    axes[1].plot(t, soln_arr[:, 1], 'b-', label='True y', linewidth=2)
    axes[1].plot(t, pred_optimized[:, 0], 'g--', label='Pred x', linewidth=2, alpha=0.7)
    axes[1].plot(t, pred_optimized[:, 1], 'b--', label='Pred y', linewidth=2, alpha=0.7)
    axes[1].axvline(config.tf_learn, color='r', linestyle=':', label='Training horizon')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Population', fontsize=11)
    axes[1].set_title(f'Optimized (MSE: {mse_optimized:.4f})', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Error comparison
    error_baseline = np.abs(pred_baseline - soln_arr)
    error_optimized = np.abs(pred_optimized - soln_arr)
    axes[2].semilogy(t, error_baseline[:, 0], 'g-', label='Baseline x error', linewidth=2)
    axes[2].semilogy(t, error_baseline[:, 1], 'b-', label='Baseline y error', linewidth=2)
    axes[2].semilogy(t, error_optimized[:, 0], 'g--', label='Optimized x error', linewidth=2, alpha=0.7)
    axes[2].semilogy(t, error_optimized[:, 1], 'b--', label='Optimized y error', linewidth=2, alpha=0.7)
    axes[2].axvline(config.tf_learn, color='r', linestyle=':', label='Training horizon')
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_ylabel('Absolute Error (log scale)', fontsize=11)
    axes[2].set_title('Error Comparison', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/demo/trajectory_comparison.png', dpi=200)
    print("✓ Saved: plots/demo/trajectory_comparison.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70 + "\n")
    print("Key optimizations demonstrated:")
    print("  ✓ L2 and smoothness regularization")
    print("  ✓ Input/output normalization")
    print("  ✓ Gradient clipping")
    print("  ✓ High-order adaptive ODE solver")
    print("  ✓ Smaller learning rate for stability")
    print("\nFor full evaluation with all features, run:")
    print("  python optimized_training.py\n")

if __name__ == "__main__":
    train_quick_demo()
