# Model Card — Skip-Connection ANN

*Generated on 2026-04-19 by the SumANN pipeline.*


## 1. Model Details

- **Model name**: Skip-Connection ANN
- **Model type**: SKIP — Skip-connection MLP
- **Author**: Siddharth Suman, Ph.D.
- **Framework**: PyTorch (trained), ONNX opset 17 (exported)
- **Pipeline**: SumANN — Physics-informed Explainable Neural Network
- **License**: Determined by the project deployment context.
- **Training mode**: Hyperparameter-optimised

## 2. Intended Use

The model is a trained regression surrogate for the scientific system represented by the training dataset. It accepts the input features listed below and produces predictions for 1 target.

**Appropriate uses**

- Point predictions and uncertainty estimates within the training envelope of the data.
- Sensitivity and attribution studies (SHAP, Sobol) to understand the network's reasoning.
- Surrogate evaluation inside an optimisation or design-of-experiments loop where many fast forward passes are needed.

**Out-of-scope uses**

- Extrapolation far outside the training distribution. MC-Dropout uncertainty grows only modestly outside the training range and should not be treated as a conservative bound.
- Safety-critical or regulated decisions without an independent verification loop.
- Interpretation of feature attributions as causal claims. SHAP and Sobol describe the fitted model, not the underlying system.

## 3. Training Data

- **Total samples**: 432
- **Input features** (5): `Load`, `Skewness`, `kurtosis`, `Pattern ratio`, `Roughness`
- **Targets** (1): `Contact area ratio`
- **Split** (train / val / test): 0.50 / 0.10 / 0.40
- **Preprocessing**: `sklearn.preprocessing.StandardScaler` fit on the combined dataset. Fitted scaler exported alongside the model as `scaler_params.json` for deployment.

## 4. Architecture

- **Layer widths**: [5, 8, 8, 1]
- **Activation**: `gelu`
- **Dropout rate**: 0.0316
- **Batch normalisation**: False
- **Skip connections**: enabled (input projection + per-layer residual sum).

## 5. Training Procedure

- **Optimiser**: `SGD`
- **Learning rate**: 0.001286
- **Batch size**: 8
- **Max epochs**: 3000  (early-stopped on validation loss, patience = 250 epochs)
- **Loss**: Mean squared error.
- **Physics residual**: disabled.
- **Hyperparameter search**: Random Search + Optuna TPE, winner selected by lowest validation loss across three candidate train/val/test splits.

## 6. Evaluation Metrics

Computed on the held-out test split (never used during training or validation).

| Metric | Value |
|---|---|
| Train RMSE | 2.044084 |
| Validation RMSE | 2.308655 |
| **Test RMSE** | **1.739004** |
| **Test R²** | **0.9854** |
| Test R² (parity-fit) | 0.9856 |

## 7. Uncertainty Quantification

Uncertainty is estimated by Monte-Carlo Dropout: 50 stochastic forward passes with dropout active and BatchNorm frozen. The standard deviation across passes is reported as ±σ alongside every prediction. This approximates epistemic uncertainty; it does not model aleatoric (label) noise.

## 8. Explainability and Sensitivity

- **Local attribution**: Kernel SHAP and Gradient SHAP, computed per target. Artefacts in `outputs/plots/shap_*.csv` and `outputs/plots/shap_*.jpg`.
- **Global sensitivity**: Saltelli–Sobol analysis, first-order + total-order + pairwise second-order indices. Artefacts in `outputs/plots/sobol_*.csv` and `outputs/plots/sobol_*.jpg`.
- **Symbolic regression**: closed-form approximations on the top-4 SHAP features per target. See `outputs/plots/symbolic_*.md` and `outputs/plots/symbolic_*.json`.

## 9. Deployment

The model has been exported to ONNX (opset 17) with a dynamic batch axis. It can be run in any ONNX-compatible runtime, including the web browser via `onnxruntime-web`. See `outputs/deploy/` for a self-contained bundle (`model.onnx`, `scaler_params.json`, `index.html`, `README.md`).

**Preprocessing required at inference time**: subtract `mean` and divide by `scale` from `scaler_params.json`, per feature, before feeding inputs to the model. The model returns predictions in the original target units — no post-scaling is needed.

## 10. Caveats and Recommendations

- **Scaler leakage**: `StandardScaler` is fit on the full dataset before splitting. For well-sampled features the effect on test R² is negligible; for small, skewed datasets it can be material.
- **PDE residual on multi-output models**: the built-in PDE terms operate on the first model output only. Re-order `output_indices` if a different variable is physically constrained.
- **MC-Dropout pathology**: if the selected dropout rate is near zero, the reported σ will also be near zero. Raise the lower bound of the dropout search range to guarantee non-degenerate uncertainty.
- **Sobol bounds**: sampled from the observed min/max of the training data (padded by 1 %). For heavy-tailed features this can be dominated by outliers.
- **Attribution ≠ causation**: SHAP and Sobol describe the fitted surrogate, not the underlying physical system. Validate any claim against domain knowledge before acting on it.

## 11. Reproducibility

The pipeline is deterministic up to the random seeds used for train/val/test splitting and weight initialisation. Optuna's TPE sampler is seeded (`seed=42`). Re-running the pipeline with the same data and config reproduces the same optimisation trajectory, though final weights can differ slightly due to non-deterministic CUDA kernels when a GPU is present.

## 12. Citation

```
Siddharth Suman, Ph.D.. SumANN: Physics-informed Explainable Neural Network.
Trained model card, generated 2026-04-19.
Pipeline: github.com/ (source-location-here)
```

## 13. References

- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Saltelli, A. et al. (2010). *Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.* Computer Physics Communications.
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
- Cranmer, M. (2023). *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.*
- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). *Discovering governing equations from data: Sparse identification of nonlinear dynamical systems.* PNAS.
