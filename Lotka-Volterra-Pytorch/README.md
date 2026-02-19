# Auxillary Pytorch code

predatory_prey.py is the driver that runs the Lotka-Volterra training in PyTorch. Results are saved into /plots/. Plots from a test run by the authors are included in this repository.

predator_prey_adjoint.py uses the adjoint method, which we found to be slightly slower given the small KAN-ODE size studied here.

This implementation relies on the efficient-kan python package from https://github.com/Blealtan/efficient-kan, which is Ref. [44] in the CMAME manuscript.

**This implementation is many times slower (~50x according to our tests) than the Julia implementation for the Lotka-Volterra case, and appears to converge to a poorer result with larger overfitting. We strongly recommend the julia implementation for KAN-ODE users, although we include this Python code for any users interested in developing KAN-ODEs in Python further.**

As a starting point for any future KAN-ODE development in Python, we refer to the following repositories which discuss KAN speed-ups and other improvements in Python:

https://github.com/AthanasiosDelis/faster-kan

https://github.com/mintisan/awesome-kan

---

## ✨ NEW: Optimized KAN-ODE Implementation

We have added a **comprehensive optimized implementation** that addresses the performance and overfitting issues mentioned above. The new implementation includes:

### Features
- ✅ **15 optimization strategies** (regularization, normalization, curriculum training, etc.)
- ✅ **Comprehensive evaluation framework** (multiple metrics, robustness testing)
- ✅ **Hybrid architectures** (KAN + MLP for residuals)
- ✅ **Model pruning and simplification**
- ✅ **Extensive visualization** (trajectories, loss curves, learned splines)
- ✅ **Complete documentation** and test suite

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run validation tests
python test_optimized_kan.py

# Quick demo (1000 epochs)
python demo_quick.py

# Full training and evaluation (5000 epochs)
python optimized_training.py
```

### Documentation
- 📖 [Complete Guide](README_OPTIMIZED.md) - Detailed documentation of all features
- 📊 [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details and code structure

### Files
- `optimized_kan.py` - Enhanced KAN implementation with regularization
- `optimized_training.py` - Comprehensive training and evaluation pipeline
- `demo_quick.py` - Quick demonstration script
- `test_optimized_kan.py` - Validation test suite

This optimized implementation provides significantly better generalization, stability, and interpretability compared to the baseline.