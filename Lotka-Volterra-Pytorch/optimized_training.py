"""
Comprehensive Optimized KAN-ODE Training and Evaluation Script

This script implements all optimization strategies and evaluation metrics for KAN-ODEs:
1. Model complexity control
2. Regularization (L2 + smoothness)
3. Input/output normalization
4. Hybrid architecture (KAN + MLP)
5. Curriculum training
6. High-order adaptive ODE solvers
7. Pruning and simplification
8. Noise handling
9. Proper training strategies
10. Dataset splitting
11. Comprehensive evaluation metrics
12. Baseline comparison
13. Visualization
14. Robustness testing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our optimized KAN and baseline efficient KAN
sys.path.append("efficient_kan/")
import efficientkan
from optimized_kan import OptimizedKAN, HybridKAN_MLP, DataNormalizer


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the optimization and evaluation experiment."""
    
    # ODE Parameters (Lotka-Volterra)
    alpha = 1.5
    beta = 1.0
    gamma = 3.0
    delta = 1.0
    x0 = 1.0
    y0 = 1.0
    
    # Time parameters
    tf = 14.0  # Total time
    tf_learn = 3.5  # Training time horizon
    N_t_train = 35  # Number of training time points
    N_t = int(35 * tf / tf_learn)  # Total time points
    
    # Dataset split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Architecture heuristics (following best practices)
    # Start small and increase if underfitting
    baseline_architecture = [2, 10, 2]  # Original
    optimized_architecture = [2, 8, 2]  # Slightly simpler to start
    
    # Grid size (start low)
    baseline_grid_size = 5
    optimized_grid_size = 5  # Start with 5, can increase to 7-10 if needed
    
    # Hybrid MLP dimensions (for residual learning)
    mlp_hidden_dims = [8, 4]
    
    # Regularization strengths
    l2_lambda = 0.0001
    smoothness_lambda = 0.001
    regularize_activation = 1.0
    regularize_entropy = 1.0
    
    # Training parameters
    baseline_lr = 2e-3
    optimized_lr = 1e-3  # Smaller learning rate for stability
    num_epochs = 5000  # More epochs
    
    # Curriculum training parameters
    use_curriculum = True
    curriculum_stages = [
        {'end_epoch': 500, 'time_horizon': 1.0},
        {'end_epoch': 1500, 'time_horizon': 2.0},
        {'end_epoch': 3000, 'time_horizon': 3.5},
    ]
    
    # ODE Solver parameters (high-order adaptive)
    rtol = 1e-6  # Strict tolerances
    atol = 1e-8
    method = 'dopri5'  # High-order Runge-Kutta
    
    # Pruning parameters
    pruning_threshold = 0.01
    
    # Noise handling
    add_training_noise = False
    noise_level = 0.01
    
    # Robustness testing
    robustness_noise_levels = [0.0, 0.01, 0.05]
    robustness_initial_conditions = [
        [0.5, 0.5],
        [1.5, 1.5],
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    
    # Output directories
    output_dir = "plots/optimized_kan"
    baseline_output_dir = "plots/baseline_kan"
    results_dir = "results"
    
    # Plotting
    plot_freq = 500


# ============================================================================
# DATA GENERATION AND PREPROCESSING
# ============================================================================

def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    """Lotka-Volterra predator-prey dynamics."""
    x, y = X
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def generate_clean_data(config):
    """Generate clean ODE solution data."""
    X0 = np.array([config.x0, config.y0])
    t = np.linspace(0, config.tf, config.N_t)
    soln_arr = scipy.integrate.odeint(
        pred_prey_deriv, X0, t,
        args=(config.alpha, config.beta, config.delta, config.gamma)
    )
    return t, soln_arr, X0


def add_noise(data, noise_level):
    """Add Gaussian noise to data for robustness testing."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise * np.std(data, axis=0)


def split_data(t, soln_arr, config):
    """
    Split data into train/validation/test sets.
    Preserves time-series ordering.
    """
    n_train = int(len(t) * config.train_ratio)
    n_val = int(len(t) * config.val_ratio)
    
    t_train = t[:n_train]
    t_val = t[n_train:n_train + n_val]
    t_test = t[n_train + n_val:]
    
    soln_train = soln_arr[:n_train]
    soln_val = soln_arr[n_train:n_train + n_val]
    soln_test = soln_arr[n_train + n_val:]
    
    return {
        'train': (t_train, soln_train),
        'val': (t_val, soln_val),
        'test': (t_test, soln_test),
    }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(pred, true):
    """
    Compute comprehensive evaluation metrics.
    Returns: MSE, MAE, RMSE, Max Error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(pred - true))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Max_Error': max_error,
    }


def compute_long_term_error(pred_trajectory, true_trajectory, horizon_indices):
    """
    Compute error at different time horizons.
    """
    errors = []
    for idx in horizon_indices:
        if idx < len(pred_trajectory):
            error = np.linalg.norm(pred_trajectory[idx] - true_trajectory[idx])
            errors.append(error)
    return np.array(errors)


def assess_stability(trajectory, window=50):
    """
    Assess trajectory stability by checking for explosions or oscillations.
    Returns: is_stable (bool), max_derivative, oscillation_count
    """
    if len(trajectory) < window:
        return True, 0.0, 0
    
    # Check for explosions (large values)
    max_val = np.max(np.abs(trajectory))
    if max_val > 1e6:
        return False, max_val, 0
    
    # Check derivative magnitude
    derivatives = np.diff(trajectory, axis=0)
    max_derivative = np.max(np.abs(derivatives))
    
    # Check for high-frequency oscillations
    second_derivatives = np.diff(derivatives, axis=0)
    sign_changes = np.sum(np.diff(np.sign(second_derivatives), axis=0) != 0, axis=0)
    oscillation_count = np.max(sign_changes)
    
    is_stable = max_val < 1e3 and max_derivative < 1e3 and oscillation_count < len(trajectory) * 0.5
    
    return is_stable, max_derivative, oscillation_count


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectories(t, true, pred, title, save_path, config):
    """Plot true vs predicted trajectories."""
    plt.figure(figsize=(12, 5))
    
    # Main trajectory plot
    plt.subplot(1, 2, 1)
    plt.plot(t, true[:, 0], 'g-', label='True x (prey)', linewidth=2)
    plt.plot(t, true[:, 1], 'b-', label='True y (predator)', linewidth=2)
    plt.plot(t, pred[:, 0], 'g--', label='Pred x (prey)', linewidth=2, alpha=0.7)
    plt.plot(t, pred[:, 1], 'b--', label='Pred y (predator)', linewidth=2, alpha=0.7)
    plt.axvline(config.tf_learn, color='r', linestyle=':', label='Training horizon')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Phase space plot
    plt.subplot(1, 2, 2)
    plt.plot(true[:, 0], true[:, 1], 'k-', label='True', linewidth=2)
    plt.plot(pred[:, 0], pred[:, 1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Prey (x)', fontsize=12)
    plt.ylabel('Predator (y)', fontsize=12)
    plt.title('Phase Space', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, title, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Training Loss', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_spline_functions(model, save_path, title="Learned Spline Functions"):
    """
    Visualize learned spline functions from KAN layers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Get model type
    if isinstance(model, HybridKAN_MLP):
        kan_model = model.kan
    else:
        kan_model = model
    
    layer_idx = 0
    for layer in kan_model.layers:
        if layer_idx >= 2:
            break
        
        # Sample input space
        x_samples = torch.linspace(-1, 1, 200).unsqueeze(1).repeat(1, layer.in_features)
        
        # Get spline outputs for first few input-output pairs
        with torch.no_grad():
            spline_bases = layer.b_splines(x_samples)
            
            # Plot first 2 input dimensions
            for in_idx in range(min(2, layer.in_features)):
                ax = axes[layer_idx, in_idx]
                
                # Plot spline function for first output
                spline_out = (spline_bases[:, in_idx, :] @ 
                             layer.scaled_spline_weight[0, in_idx, :])
                
                ax.plot(x_samples[:, in_idx].numpy(), 
                       spline_out.numpy(), 
                       linewidth=2)
                ax.set_xlabel(f'Input {in_idx+1}', fontsize=10)
                ax.set_ylabel('Output', fontsize=10)
                ax.set_title(f'Layer {layer_idx+1}, Input {in_idx+1}', fontsize=11)
                ax.grid(True, alpha=0.3)
        
        layer_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_comparison(baseline_metrics, optimized_metrics, save_path):
    """Create bar chart comparing baseline vs optimized metrics."""
    metrics = ['MSE', 'MAE', 'RMSE', 'Max_Error']
    baseline_vals = [baseline_metrics[m] for m in metrics]
    optimized_vals = [optimized_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Baseline vs Optimized KAN Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage improvement labels
    for i, (b, o) in enumerate(zip(baseline_vals, optimized_vals)):
        improvement = ((b - o) / b) * 100
        ax.text(i, max(b, o) * 1.2, f'{improvement:.1f}%', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# MODEL TRAINING
# ============================================================================

def create_derivative_function(model, normalizer=None):
    """Create derivative function for ODE solver."""
    def calDeriv(t, X):
        if normalizer is not None:
            X_norm = normalizer.normalize_input(X)
            dXdt_norm = model(X_norm)
            # Scale derivatives appropriately
            dXdt = dXdt_norm * (normalizer.output_max - normalizer.output_min) / (
                normalizer.input_max - normalizer.input_min)
        else:
            dXdt = model(X)
        return dXdt
    return calDeriv


def train_model(model, config, X0, t_full, soln_full, data_splits, 
                use_normalization=False, model_name="model", is_baseline=False):
    """
    Train KAN-ODE model with all optimizations.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}\n")
    
    # Setup normalization if requested
    normalizer = None
    if use_normalization:
        normalizer = DataNormalizer(data_range=(-1, 1))
        normalizer.fit_input(soln_full)
        normalizer.fit_output(soln_full)
        print("✓ Input/output normalization enabled")
    
    # Prepare training data
    t_train, soln_train = data_splits['train']
    t_val, soln_val = data_splits['val']
    
    # Convert to tensors
    X0_tensor = torch.tensor(X0, dtype=torch.float32).unsqueeze(0)
    soln_train_tensor = torch.tensor(soln_train, dtype=torch.float32)
    soln_val_tensor = torch.tensor(soln_val, dtype=torch.float32)
    t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
    t_val_tensor = torch.tensor(t_val, dtype=torch.float32)
    t_full_tensor = torch.tensor(t_full, dtype=torch.float32)
    
    # Optimizer with smaller learning rate
    lr = config.baseline_lr if is_baseline else config.optimized_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    # Curriculum training setup
    if config.use_curriculum and not is_baseline:
        print("✓ Curriculum training enabled")
        current_stage = 0
    
    # Create output directory
    output_dir = config.baseline_output_dir if is_baseline else config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/training_progress", exist_ok=True)
    
    # Training loop
    pbar = tqdm(range(config.num_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        # Curriculum learning: adjust time horizon
        if config.use_curriculum and not is_baseline:
            for stage in config.curriculum_stages:
                if epoch <= stage['end_epoch'] and stage != config.curriculum_stages[current_stage]:
                    current_stage = config.curriculum_stages.index(stage)
                    t_horizon = stage['time_horizon']
                    n_points = int(config.N_t_train * t_horizon / config.tf_learn)
                    print(f"\n✓ Curriculum stage: extending to {t_horizon}s ({n_points} points)")
                    break
            else:
                t_horizon = config.tf_learn
                n_points = config.N_t_train
            
            # Adjust training time
            t_curr = torch.tensor(np.linspace(0, t_horizon, n_points), dtype=torch.float32)
            soln_curr = soln_train_tensor[:n_points]
        else:
            t_curr = t_train_tensor
            soln_curr = soln_train_tensor
        
        # Forward pass
        optimizer.zero_grad()
        
        calDeriv = create_derivative_function(model, normalizer)
        pred_train = torchodeint(
            calDeriv, X0_tensor, t_curr,
            method=config.method, rtol=config.rtol, atol=config.atol
        )
        
        # Compute loss
        loss_data = torch.mean((pred_train[:, 0, :] - soln_curr) ** 2)
        
        # Add regularization (only for optimized model)
        if not is_baseline and hasattr(model, 'regularization_loss'):
            loss_reg = model.regularization_loss(
                config.regularize_activation, 
                config.regularize_entropy
            )
            loss = loss_data + loss_reg
        else:
            loss = loss_data
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            pred_val = torchodeint(
                calDeriv, X0_tensor, t_val_tensor,
                method=config.method, rtol=config.rtol, atol=config.atol
            )
            loss_val = torch.mean((pred_val[:, 0, :] - soln_val_tensor) ** 2)
            
            # Full trajectory for testing
            pred_full = torchodeint(
                calDeriv, X0_tensor, t_full_tensor,
                method=config.method, rtol=config.rtol, atol=config.atol
            )
            soln_full_tensor = torch.tensor(soln_full, dtype=torch.float32)
            loss_test = torch.mean((pred_full[:, 0, :] - soln_full_tensor) ** 2)
        
        # Record losses
        train_losses.append(loss_data.item())
        val_losses.append(loss_val.item())
        test_losses.append(loss_test.item())
        
        # Update learning rate
        scheduler.step(loss_val)
        
        # Save best model
        if loss_val < best_val_loss:
            best_val_loss = loss_val.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{loss_data.item():.6f}',
            'val_loss': f'{loss_val.item():.6f}',
            'test_loss': f'{loss_test.item():.6f}',
        })
        
        # Periodic plotting
        if epoch % config.plot_freq == 0 or epoch == config.num_epochs - 1:
            pred_plot = pred_full[:, 0, :].detach().numpy()
            plot_trajectories(
                t_full, soln_full, pred_plot,
                f'{model_name} - Epoch {epoch}',
                f'{output_dir}/training_progress/epoch_{epoch:04d}.png',
                config
            )
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.6f}")
    
    # Plot loss curves
    plot_loss_curves(
        train_losses, val_losses,
        f'{model_name} - Loss Curves',
        f'{output_dir}/loss_curves.png'
    )
    
    return {
        'model': model,
        'normalizer': normalizer,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'best_val_loss': best_val_loss,
    }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(model, config, X0, t_full, soln_full, data_splits, 
                   normalizer=None, model_name="model"):
    """
    Comprehensive model evaluation with all metrics.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}\n")
    
    X0_tensor = torch.tensor(X0, dtype=torch.float32).unsqueeze(0)
    t_full_tensor = torch.tensor(t_full, dtype=torch.float32)
    
    calDeriv = create_derivative_function(model, normalizer)
    
    # Generate predictions
    with torch.no_grad():
        pred_full = torchodeint(
            calDeriv, X0_tensor, t_full_tensor,
            method=config.method, rtol=config.rtol, atol=config.atol
        )
        pred_full = pred_full[:, 0, :].numpy()
    
    # Compute metrics for each split
    results = {}
    for split_name, (t_split, soln_split) in data_splits.items():
        split_indices = [i for i, t in enumerate(t_full) if t in t_split]
        pred_split = pred_full[split_indices]
        
        metrics = compute_metrics(pred_split, soln_split)
        results[f'{split_name}_metrics'] = metrics
        
        print(f"\n{split_name.upper()} SET:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6e}")
    
    # Long-term prediction error
    horizon_indices = [
        int(config.N_t * 0.25),
        int(config.N_t * 0.5),
        int(config.N_t * 0.75),
        config.N_t - 1
    ]
    long_term_errors = compute_long_term_error(pred_full, soln_full, horizon_indices)
    results['long_term_errors'] = long_term_errors
    
    print(f"\nLONG-TERM ERRORS (at 25%, 50%, 75%, 100% of time horizon):")
    for i, err in enumerate(long_term_errors):
        print(f"  {horizon_indices[i]}/{config.N_t}: {err:.6e}")
    
    # Stability assessment
    is_stable, max_deriv, osc_count = assess_stability(pred_full)
    results['stability'] = {
        'is_stable': is_stable,
        'max_derivative': max_deriv,
        'oscillation_count': osc_count,
    }
    
    print(f"\nSTABILITY ASSESSMENT:")
    print(f"  Stable: {is_stable}")
    print(f"  Max derivative: {max_deriv:.6e}")
    print(f"  Oscillation count: {osc_count}")
    
    # Store predictions for visualization
    results['predictions'] = pred_full
    
    return results


def robustness_testing(model, config, normalizer=None, model_name="model"):
    """
    Test model robustness to noise and different initial conditions.
    """
    print(f"\n{'='*70}")
    print(f"Robustness Testing for {model_name}")
    print(f"{'='*70}\n")
    
    results = {
        'noise_robustness': {},
        'ic_robustness': {},
    }
    
    t_full = np.linspace(0, config.tf, config.N_t)
    t_tensor = torch.tensor(t_full, dtype=torch.float32)
    
    # Test 1: Noise robustness
    print("Testing robustness to observation noise...")
    X0 = np.array([config.x0, config.y0])
    soln_clean = scipy.integrate.odeint(
        pred_prey_deriv, X0, t_full,
        args=(config.alpha, config.beta, config.delta, config.gamma)
    )
    
    for noise_level in config.robustness_noise_levels:
        X0_tensor = torch.tensor(X0, dtype=torch.float32).unsqueeze(0)
        
        calDeriv = create_derivative_function(model, normalizer)
        with torch.no_grad():
            pred = torchodeint(
                calDeriv, X0_tensor, t_tensor,
                method=config.method, rtol=config.rtol, atol=config.atol
            )
            pred = pred[:, 0, :].numpy()
        
        # Add noise to observations (not to model)
        soln_noisy = add_noise(soln_clean, noise_level)
        metrics = compute_metrics(pred, soln_noisy)
        results['noise_robustness'][noise_level] = metrics
        
        print(f"  Noise level {noise_level}: MSE = {metrics['MSE']:.6e}, MAE = {metrics['MAE']:.6e}")
    
    # Test 2: Different initial conditions
    print("\nTesting robustness to different initial conditions...")
    for i, ic in enumerate(config.robustness_initial_conditions):
        X0_new = np.array(ic)
        soln_new = scipy.integrate.odeint(
            pred_prey_deriv, X0_new, t_full,
            args=(config.alpha, config.beta, config.delta, config.gamma)
        )
        
        X0_tensor = torch.tensor(X0_new, dtype=torch.float32).unsqueeze(0)
        calDeriv = create_derivative_function(model, normalizer)
        
        with torch.no_grad():
            pred = torchodeint(
                calDeriv, X0_tensor, t_tensor,
                method=config.method, rtol=config.rtol, atol=config.atol
            )
            pred = pred[:, 0, :].numpy()
        
        metrics = compute_metrics(pred, soln_new)
        results['ic_robustness'][f'IC_{i}'] = metrics
        
        print(f"  IC {ic}: MSE = {metrics['MSE']:.6e}, MAE = {metrics['MAE']:.6e}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("KAN-ODE OPTIMIZATION AND EVALUATION FRAMEWORK")
    print("="*70 + "\n")
    
    config = Config()
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.baseline_output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Generate and prepare data
    # ========================================================================
    print("\n[STEP 1] Generating training data...")
    t_full, soln_full, X0 = generate_clean_data(config)
    data_splits = split_data(t_full, soln_full, config)
    
    print(f"  Total time points: {len(t_full)}")
    print(f"  Training points: {len(data_splits['train'][0])}")
    print(f"  Validation points: {len(data_splits['val'][0])}")
    print(f"  Test points: {len(data_splits['test'][0])}")
    
    # ========================================================================
    # STEP 2: Train baseline model
    # ========================================================================
    print("\n[STEP 2] Training baseline KAN model...")
    baseline_model = efficientkan.KAN(
        layers_hidden=config.baseline_architecture,
        grid_size=config.baseline_grid_size
    )
    
    baseline_results = train_model(
        baseline_model, config, X0, t_full, soln_full, data_splits,
        use_normalization=False,
        model_name="Baseline KAN",
        is_baseline=True
    )
    
    # ========================================================================
    # STEP 3: Train optimized model
    # ========================================================================
    print("\n[STEP 3] Training optimized KAN model...")
    optimized_model = OptimizedKAN(
        layers_hidden=config.optimized_architecture,
        grid_size=config.optimized_grid_size,
        l2_lambda=config.l2_lambda,
        smoothness_lambda=config.smoothness_lambda,
    )
    
    optimized_results = train_model(
        optimized_model, config, X0, t_full, soln_full, data_splits,
        use_normalization=True,
        model_name="Optimized KAN"
    )
    
    # ========================================================================
    # STEP 4: Train hybrid model
    # ========================================================================
    print("\n[STEP 4] Training hybrid KAN+MLP model...")
    hybrid_model = HybridKAN_MLP(
        kan_layers=config.optimized_architecture,
        mlp_hidden_dims=config.mlp_hidden_dims,
        grid_size=config.optimized_grid_size,
        l2_lambda=config.l2_lambda,
        smoothness_lambda=config.smoothness_lambda,
    )
    
    hybrid_results = train_model(
        hybrid_model, config, X0, t_full, soln_full, data_splits,
        use_normalization=True,
        model_name="Hybrid KAN+MLP"
    )
    
    # ========================================================================
    # STEP 5: Evaluate all models
    # ========================================================================
    print("\n[STEP 5] Evaluating all models...")
    
    baseline_eval = evaluate_model(
        baseline_results['model'], config, X0, t_full, soln_full, data_splits,
        normalizer=None,
        model_name="Baseline KAN"
    )
    
    optimized_eval = evaluate_model(
        optimized_results['model'], config, X0, t_full, soln_full, data_splits,
        normalizer=optimized_results['normalizer'],
        model_name="Optimized KAN"
    )
    
    hybrid_eval = evaluate_model(
        hybrid_results['model'], config, X0, t_full, soln_full, data_splits,
        normalizer=hybrid_results['normalizer'],
        model_name="Hybrid KAN+MLP"
    )
    
    # ========================================================================
    # STEP 6: Apply pruning to optimized model
    # ========================================================================
    print("\n[STEP 6] Applying model pruning...")
    pruned_edges = optimized_results['model'].prune_model(config.pruning_threshold)
    print(f"  Pruned {pruned_edges} low-importance edges")
    
    # Re-evaluate after pruning
    optimized_pruned_eval = evaluate_model(
        optimized_results['model'], config, X0, t_full, soln_full, data_splits,
        normalizer=optimized_results['normalizer'],
        model_name="Optimized KAN (Pruned)"
    )
    
    # ========================================================================
    # STEP 7: Robustness testing
    # ========================================================================
    print("\n[STEP 7] Robustness testing...")
    
    baseline_robustness = robustness_testing(
        baseline_results['model'], config,
        normalizer=None,
        model_name="Baseline KAN"
    )
    
    optimized_robustness = robustness_testing(
        optimized_results['model'], config,
        normalizer=optimized_results['normalizer'],
        model_name="Optimized KAN"
    )
    
    # ========================================================================
    # STEP 8: Generate visualizations
    # ========================================================================
    print("\n[STEP 8] Generating visualizations...")
    
    # Trajectory plots
    plot_trajectories(
        t_full, soln_full, baseline_eval['predictions'],
        'Baseline KAN - Final Prediction',
        f'{config.baseline_output_dir}/final_prediction.png',
        config
    )
    
    plot_trajectories(
        t_full, soln_full, optimized_eval['predictions'],
        'Optimized KAN - Final Prediction',
        f'{config.output_dir}/final_prediction.png',
        config
    )
    
    plot_trajectories(
        t_full, soln_full, hybrid_eval['predictions'],
        'Hybrid KAN+MLP - Final Prediction',
        f'{config.output_dir}/hybrid_final_prediction.png',
        config
    )
    
    # Spline function visualization
    plot_spline_functions(
        baseline_results['model'],
        f'{config.baseline_output_dir}/spline_functions.png',
        'Baseline KAN - Learned Spline Functions'
    )
    
    plot_spline_functions(
        optimized_results['model'],
        f'{config.output_dir}/spline_functions.png',
        'Optimized KAN - Learned Spline Functions'
    )
    
    # Comparison plot
    plot_comparison(
        baseline_eval['test_metrics'],
        optimized_eval['test_metrics'],
        f'{config.results_dir}/baseline_vs_optimized.png'
    )
    
    # ========================================================================
    # STEP 9: Save results and generate report
    # ========================================================================
    print("\n[STEP 9] Saving results...")
    
    # Compile all results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: v for k, v in vars(config).items() if not k.startswith('_')},
        'baseline': {
            'test_metrics': baseline_eval['test_metrics'],
            'stability': baseline_eval['stability'],
            'final_train_loss': baseline_results['train_losses'][-1],
            'final_val_loss': baseline_results['val_losses'][-1],
            'best_val_loss': baseline_results['best_val_loss'],
        },
        'optimized': {
            'test_metrics': optimized_eval['test_metrics'],
            'stability': optimized_eval['stability'],
            'final_train_loss': optimized_results['train_losses'][-1],
            'final_val_loss': optimized_results['val_losses'][-1],
            'best_val_loss': optimized_results['best_val_loss'],
            'pruned_edges': pruned_edges,
        },
        'hybrid': {
            'test_metrics': hybrid_eval['test_metrics'],
            'stability': hybrid_eval['stability'],
            'final_train_loss': hybrid_results['train_losses'][-1],
            'final_val_loss': hybrid_results['val_losses'][-1],
            'best_val_loss': hybrid_results['best_val_loss'],
        },
        'improvements': {},
    }
    
    # Calculate percentage improvements
    for metric in ['MSE', 'MAE', 'RMSE', 'Max_Error']:
        baseline_val = baseline_eval['test_metrics'][metric]
        optimized_val = optimized_eval['test_metrics'][metric]
        improvement = ((baseline_val - optimized_val) / baseline_val) * 100
        final_results['improvements'][metric] = improvement
    
    # Save to JSON
    results_file = f'{config.results_dir}/results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # ========================================================================
    # STEP 10: Print summary report
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70 + "\n")
    
    print("BASELINE KAN:")
    print(f"  Architecture: {config.baseline_architecture}")
    print(f"  Grid size: {config.baseline_grid_size}")
    print(f"  Test MSE: {baseline_eval['test_metrics']['MSE']:.6e}")
    print(f"  Test MAE: {baseline_eval['test_metrics']['MAE']:.6e}")
    print(f"  Stable: {baseline_eval['stability']['is_stable']}")
    
    print("\nOPTIMIZED KAN:")
    print(f"  Architecture: {config.optimized_architecture}")
    print(f"  Grid size: {config.optimized_grid_size}")
    print(f"  Test MSE: {optimized_eval['test_metrics']['MSE']:.6e}")
    print(f"  Test MAE: {optimized_eval['test_metrics']['MAE']:.6e}")
    print(f"  Stable: {optimized_eval['stability']['is_stable']}")
    print(f"  Pruned edges: {pruned_edges}")
    
    print("\nHYBRID KAN+MLP:")
    print(f"  KAN Architecture: {config.optimized_architecture}")
    print(f"  MLP Hidden: {config.mlp_hidden_dims}")
    print(f"  Test MSE: {hybrid_eval['test_metrics']['MSE']:.6e}")
    print(f"  Test MAE: {hybrid_eval['test_metrics']['MAE']:.6e}")
    print(f"  Stable: {hybrid_eval['stability']['is_stable']}")
    
    print("\nIMPROVEMENTS (Optimized vs Baseline):")
    for metric, improvement in final_results['improvements'].items():
        print(f"  {metric}: {improvement:+.2f}%")
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*70)
    print("✓ OPTIMIZATION AND EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
