# Optimized KAN-ODE Implementation

This directory contains a comprehensive optimized implementation of Kolmogorov-Arnold Networks (KANs) for learning dynamical systems, specifically applied to the Lotka-Volterra predator-prey model.

## Overview

The implementation includes all recommended optimization strategies and evaluation metrics for KAN-based ODE learning:

### Key Features

#### 1. **Model Complexity Control** ✓
- Starts with minimal architecture (2-3 layers, 5-20 neurons)
- Low initial spline grid resolution (5-10)
- Gradual capacity increase if needed

#### 2. **Enhanced Regularization** ✓
- **L2 regularization** on spline coefficients (prevents overfitting)
- **Smoothness penalties** via second-derivative regularization (prevents oscillatory behavior)
- **L1 and entropy regularization** (from original efficient-kan)

#### 3. **Input/Output Normalization** ✓
- Automatic normalization to [-1, 1] range
- Ensures proper spline knot placement
- Scales outputs to avoid magnitude variations

#### 4. **Hybrid Architecture** ✓
- **OptimizedKAN**: Enhanced KAN with all optimizations
- **HybridKAN_MLP**: KAN for main dynamics + MLP for residuals
- Learned weight balancing between components

#### 5. **Curriculum Training** ✓
- Progressive time horizon extension
- Starts with short intervals (1.0s → 2.0s → 3.5s)
- Gradual complexity increase

#### 6. **High-Order Adaptive Solvers** ✓
- Dormand-Prince (dopri5) method
- Strict tolerances (rtol=1e-6, atol=1e-8)
- Stable and accurate ODE integration

#### 7. **Pruning and Simplification** ✓
- Post-training edge pruning
- Removes near-zero contribution edges
- Improves interpretability and efficiency

#### 8. **Noise Handling** ✓
- Noise robustness testing
- Derivative-based regularization
- Smooth training data processing

#### 9. **Architecture Heuristics** ✓
- Recommended: 2-3 layers
- Width: 5-20 neurons
- Grid size: 5-10
- Lower learning rates (1e-3 vs 2e-3)

#### 10. **Comprehensive Training Strategy** ✓
- Smaller learning rates
- Learning rate scheduling
- Gradient clipping for stability
- Extended training (5000 epochs)
- Early stopping on validation loss

---

## Evaluation Framework

### 11. **Dataset Splitting** ✓
- Train: 70%
- Validation: 15%
- Test: 15%
- Preserved time-series ordering

### 12. **Evaluation Metrics** ✓
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Maximum Error**
- **Long-term prediction error** (at 25%, 50%, 75%, 100% horizons)
- **Stability metrics** (max derivatives, oscillation counts)

### 13. **Baseline Comparison** ✓
- Side-by-side training of baseline and optimized models
- Percentage improvement calculations
- Statistical significance testing

### 14. **Visualization** ✓
- True vs predicted trajectories (time series)
- Phase space plots
- Training vs validation loss curves
- Learned spline functions
- Baseline vs optimized comparison charts

### 15. **Robustness Testing** ✓
- **Unseen initial conditions**: Tests 4 different ICs
- **Noise robustness**: Tests at 0%, 1%, 5% noise levels
- **Long-time integration**: Full 14s trajectory stability

---

## File Structure

```
Lotka-Volterra-Pytorch/
├── optimized_kan.py           # Optimized KAN implementation
│   ├── OptimizedKANLinear     # Enhanced KAN layer with regularization
│   ├── OptimizedKAN           # Full optimized KAN model
│   ├── HybridKAN_MLP          # Hybrid KAN+MLP architecture
│   └── DataNormalizer         # Input/output normalization utility
│
├── optimized_training.py      # Comprehensive training & evaluation script
│   ├── Data generation and preprocessing
│   ├── Model training with all optimizations
│   ├── Comprehensive evaluation
│   ├── Robustness testing
│   └── Visualization generation
│
├── efficient_kan/             # Baseline efficient-kan implementation
│   ├── efficientkan.py        # Original KAN implementation
│   └── __init__.py
│
├── predator_prey.py           # Original baseline script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- `torch>=1.9.0` - PyTorch for neural network implementation
- `numpy>=1.19.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting and visualization
- `scipy>=1.5.0` - ODE integration and scientific computing
- `torchdiffeq>=0.2.0` - Neural ODE solvers
- `tqdm>=4.50.0` - Progress bars

---

## Usage

### Quick Start

Run the comprehensive optimization and evaluation:

```bash
cd Lotka-Volterra-Pytorch
python optimized_training.py
```

This will:
1. Generate Lotka-Volterra training data
2. Train a baseline KAN model
3. Train an optimized KAN model with all enhancements
4. Train a hybrid KAN+MLP model
5. Evaluate all models on train/val/test sets
6. Perform robustness testing
7. Generate comprehensive visualizations
8. Save results and metrics

### Expected Output

The script will create the following directories:

```
plots/
├── baseline_kan/
│   ├── training_progress/     # Training evolution plots
│   ├── final_prediction.png   # Final trajectory plot
│   ├── loss_curves.png        # Training/validation loss
│   └── spline_functions.png   # Learned spline visualizations
│
└── optimized_kan/
    ├── training_progress/
    ├── final_prediction.png
    ├── hybrid_final_prediction.png
    ├── loss_curves.png
    └── spline_functions.png

results/
└── results_YYYYMMDD_HHMMSS.json  # Detailed metrics and comparisons
```

---

## Implementation Details

### Optimized KAN Components

#### 1. L2 Regularization
```python
def l2_regularization(self):
    """Penalizes large spline coefficients"""
    return self.l2_lambda * torch.sum(self.spline_weight ** 2)
```

#### 2. Smoothness Penalty
```python
def smoothness_regularization(self):
    """Penalizes high curvature (second derivative approximation)"""
    second_diff = (
        self.spline_weight[:, :, :-2] 
        - 2 * self.spline_weight[:, :, 1:-1] 
        + self.spline_weight[:, :, 2:]
    )
    return self.smoothness_lambda * torch.sum(second_diff ** 2)
```

#### 3. Total Loss Function
```python
loss = loss_data + loss_l2_reg + loss_smoothness + loss_l1 + loss_entropy
```

### Training Enhancements

#### Curriculum Learning
The model progressively trains on longer time horizons:
- **Stage 1 (epochs 0-500)**: 1.0s horizon
- **Stage 2 (epochs 501-1500)**: 2.0s horizon  
- **Stage 3 (epochs 1501-3000)**: 3.5s horizon (full training)

#### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500
)
```

#### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Hybrid Architecture

The `HybridKAN_MLP` combines:
- **KAN component**: Learns primary system dynamics
- **MLP component**: Captures residual errors
- **Learned weighting**: Balances contributions

```python
output = (1 - alpha) * kan_output + alpha * mlp_output
```

---

## Results Interpretation

### Metrics Explained

1. **MSE (Mean Squared Error)**: Average squared difference - penalizes large errors heavily
2. **MAE (Mean Absolute Error)**: Average absolute difference - robust to outliers
3. **RMSE (Root Mean Squared Error)**: Square root of MSE - same units as data
4. **Max Error**: Largest point-wise error - indicates worst-case performance

### Stability Assessment

The framework checks:
- **Value explosions**: max(|trajectory|) < 1000
- **Derivative magnitude**: Ensures smooth trajectories
- **Oscillation frequency**: Detects unstable numerical behavior

### Improvement Calculation

```
Improvement (%) = (Baseline_Metric - Optimized_Metric) / Baseline_Metric × 100
```

Positive values indicate improvement over baseline.

---

## Configuration

Key parameters can be adjusted in the `Config` class in `optimized_training.py`:

```python
# Architecture
optimized_architecture = [2, 8, 2]    # [input, hidden, output]
optimized_grid_size = 5               # Spline grid resolution

# Regularization
l2_lambda = 0.0001                    # L2 penalty strength
smoothness_lambda = 0.001             # Smoothness penalty strength

# Training
optimized_lr = 1e-3                   # Learning rate
num_epochs = 5000                     # Training duration

# Curriculum
use_curriculum = True                 # Enable curriculum learning
curriculum_stages = [...]             # Time horizon progression

# ODE Solver
method = 'dopri5'                     # Solver method
rtol = 1e-6                           # Relative tolerance
atol = 1e-8                           # Absolute tolerance
```

---

## Expected Improvements

Based on the optimization strategies, you should observe:

✓ **Better Generalization**: Lower test error compared to baseline  
✓ **Improved Stability**: Smoother trajectories, fewer oscillations  
✓ **Longer Horizon Accuracy**: Better predictions beyond training range  
✓ **Noise Robustness**: More consistent under noisy observations  
✓ **IC Robustness**: Better generalization to unseen initial conditions  
✓ **Reduced Overfitting**: Smaller train-validation gap  
✓ **Smoother Splines**: Less oscillatory learned functions  
✓ **Interpretability**: Pruned models are simpler and clearer  

---

## Troubleshooting

### Common Issues

1. **NaN losses during training**
   - Reduce learning rate (try 5e-4)
   - Increase regularization strength
   - Check input normalization

2. **Poor convergence**
   - Increase number of epochs
   - Adjust curriculum schedule
   - Try different grid sizes (7-10)

3. **Overfitting (large train-val gap)**
   - Increase L2/smoothness regularization
   - Reduce model capacity
   - Add more training data

4. **Instability in predictions**
   - Use stricter ODE solver tolerances
   - Apply gradient clipping
   - Reduce learning rate

---

## Comparison with Original Implementation

### Original (`predator_prey.py`)
- Basic KAN architecture
- Minimal regularization
- Fixed time horizon training
- No input normalization
- Single model type
- Limited evaluation metrics

### Optimized (`optimized_training.py`)
- ✓ Multiple architectures (KAN, Hybrid)
- ✓ Comprehensive regularization (L2, smoothness, L1, entropy)
- ✓ Curriculum learning
- ✓ Input/output normalization
- ✓ Baseline comparison
- ✓ 15+ evaluation metrics
- ✓ Robustness testing
- ✓ Extensive visualization
- ✓ Model pruning
- ✓ Stability assessment

---

## Citation

If you use this optimized implementation, please cite the original KAN-ODEs paper:

```bibtex
@article{koenig2024kanodes,
  title = {KAN-ODEs: Kolmogorov–Arnold network ordinary differential equations for learning dynamical systems and hidden physics},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {432},
  pages = {117397},
  year = {2024},
  author = {Benjamin C. Koenig and Suyong Kim and Sili Deng},
}
```

---

## Contributing

Suggestions for further improvements:
- [ ] Multi-GPU training support
- [ ] Hyperparameter optimization (Optuna/Ray Tune)
- [ ] Additional dynamical systems examples
- [ ] Uncertainty quantification
- [ ] Symbolic regression integration
- [ ] Real-time training visualization (TensorBoard/Weights & Biases)

---

## License

This code follows the license of the parent repository.

---

## Acknowledgments

- Original efficient-kan implementation: [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)
- Neural ODE solver: [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- KAN-ODEs framework: [DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs)

---

**Last Updated**: 2024-02-19  
**Version**: 1.0.0
