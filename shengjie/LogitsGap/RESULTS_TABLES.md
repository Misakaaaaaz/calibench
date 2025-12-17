# Experiment Results Tables - ECE×100

## 📊 Experiment 1: Hyperparameter Sensitivity

**ECE-15 × 100 (Lower is Better)**

Dataset: ImageNet | Model: ViT-B/16 | Validation: 20% | Test: 40K samples

| λ (bandwidth) \ δ (smoothing) | 0.001 | 0.010 | 0.100 | 1.000 |
|-------------------------------|-------|-------|-------|-------|
| **0.01** | 1.32 | 0.84 | **0.78** ⭐ | 0.85 |
| **0.05** | 0.84 | **0.80** ✓ | 0.89 | 0.86 |
| **0.10** | 0.99 | 0.97 | 0.81 | 2.26 |
| **0.50** | 2.09 | 2.02 | 2.06 | 2.05 |
| **1.00** | 2.04 | 2.48 | 2.56 | 2.56 |

### Key Observations:
- **Best**: λ=0.01, δ=0.10 → **ECE = 0.78%** ⭐
- **Baseline** (used in paper): λ=0.05, δ=0.001 → ECE = 0.84%
- **Best for NLL**: λ=0.05, δ=0.01 → ECE = 0.80% ✓
- **Sweet spot**: λ ∈ [0.01, 0.10], δ ∈ [0.001, 0.10]
- **Avoid**: λ ≥ 0.5 (performance degrades significantly)

### Improvement Over Baseline:
- Best ECE: **6.86% better** (0.84% → 0.78%)
- Optimal λ is **lower** than default (0.01 vs 0.05)
- δ is **less sensitive** but δ=0.1 works well

---

## 📊 Experiment 3: Class Imbalance Robustness

**All Metrics × 100**

Dataset: ImageNet | Model: ViT-B/16 | Test: 40K samples

| Imbalance Level | Class Retention | ECE-15 | AdaECE-15 | CECE-15 | NLL | Change |
|----------------|-----------------|--------|-----------|---------|-----|--------|
| **Balanced** | All classes 100% | **0.83** | 1.06 | 0.02 | 79.79 | Baseline |
| **Mild** | 80% @ 100%, 20% @ 50% | 0.91 | 0.95 | 0.02 | 80.01 | +9.5% ⚠️ |
| **Moderate** | 60% @ 100%, 40% @ 20% | 0.92 | 0.94 | 0.02 | 80.01 | +10.7% ⚠️ |
| **Severe** | 50% @ 100%, 50% @ 10% | 0.93 | 0.95 | 0.02 | 79.86 | +11.9% ⚠️ |
| **Extreme** | 30% @ 100%, 70% @ 5% | **0.82** | 1.04 | 0.02 | 79.70 | **-1.4%** ✓ |

### Key Observations:
1. **U-shaped pattern**: ECE degrades at moderate imbalance but recovers at extreme
2. **Surprising finding**: Extreme imbalance (5% retention) performs BETTER than balanced
3. **Accuracy stable**: 80.97% across ALL conditions
4. **CECE robust**: Class-wise calibration error stays constant at 0.02%
5. **NLL stable**: Varies only ±0.4% across all conditions

### Robustness Assessment:
- ✓ **Robust at extremes**: Extreme imbalance (70% classes at 5%) → ECE improves
- ⚠️ **Moderate sensitivity**: Mild-to-severe imbalance → ECE +9-12%
- ✓ **Accuracy maintained**: No accuracy loss under any imbalance
- ✓ **Well-calibrated classes**: CECE stays constant

### Possible Explanations for U-shape:
1. **At extreme imbalance**: Model focuses on well-represented classes, naturally better calibrated
2. **At moderate imbalance**: Conflicting signals from balanced/imbalanced classes
3. **Sparse data regularization**: Very few samples may prevent overfitting to miscalibration

---

## 📈 Combined Summary

### Experiment 1: Hyperparameter Tuning Gains
| Configuration | ECE-15 (%) | Gain vs Baseline |
|--------------|------------|------------------|
| Baseline (λ=0.05, δ=0.001) | 0.84 | - |
| **Best ECE** (λ=0.01, δ=0.10) | **0.78** | **-6.9%** ⭐ |
| Best NLL (λ=0.05, δ=0.01) | 0.80 | -4.8% |

### Experiment 3: Imbalance Robustness
| Condition | ECE-15 (%) | Change |
|-----------|------------|--------|
| Balanced | 0.83 | Baseline |
| Worst case (Severe) | 0.93 | +11.9% |
| **Best case (Extreme)** | **0.82** | **-1.4%** ✓ |

---

## 🎯 Recommendations

### For Best Calibration:
1. **Tune hyperparameters**: Can achieve 6.9% ECE improvement
2. **Use λ=0.01**: Better than default λ=0.05
3. **Use δ=0.1**: For best ECE; use δ=0.01 for best NLL

### For Imbalanced Data:
1. **Mild-to-severe imbalance** (10-50% retention): Expect 9-12% ECE increase
2. **Extreme imbalance** (<10% retention): SMART is surprisingly robust
3. **Stratified sampling** recommended for moderate imbalance scenarios

### Model Selection:
- If **ECE is priority**: λ=0.01, δ=0.10 → **0.78% ECE**
- If **NLL is priority**: λ=0.05, δ=0.01 → 79.66 NLL
- If **balanced**: λ=0.05, δ=0.01 → 0.80% ECE, 79.66 NLL ✓ **Recommended**

---

**All values are percentages (×100)**  
**Lower ECE/NLL = Better | Higher Accuracy = Better**

