# Numerical Data

Practical notes on numeric types, precision, and numerical stability in ML.

---

## Numeric Types and Precision

- **Floating point**: `float32` (single) and `float64` (double). `float64` offers better precision but uses more memory.
- **Integers**: Used for counts/indices. Avoid integer arithmetic when you need averages or divisions.

---

## Floating-Point Issues

### Rounding Errors

Floating point numbers are approximations; sums or differences of numbers with very different magnitudes can lose precision.

```python
# Adding a very small number to a large number
a = 1e16
b = 1.0
print(a + b - a)  # Expected 1.0 but may yield 0.0 due to rounding
```

### Catastrophic Cancellation

Subtracting nearly equal numbers can eliminate significant digits.

```python
import math
x = 1e10
print(math.sqrt(x+1) - math.sqrt(x))              # Unstable
print(1.0 / (math.sqrt(x+1) + math.sqrt(x)))     # Stable alternative
```

### NaN/Inf

Check for NaN or infinite values during preprocessing — they break most algorithms.

```python
import numpy as np
# Detect NaN/Inf
np.isnan(data)      # Find NaN values
np.isfinite(data)   # Find finite values
```

---

## Numerical Stability in ML

Compute losses from raw logits when possible:

```python
# Unstable: softmax then log
logits = np.array([1000, 1000])
probs = np.exp(logits) / np.sum(np.exp(logits))  # Overflow!

# Stable: use logsumexp trick
from scipy.special import logsumexp
log_probs = logits - logsumexp(logits)
```

Use stable functions from libraries (`scipy.special.logsumexp`, `numpy.matmul`).

---

## Quick Reference

| Issue | Solution |
|-------|----------|
| Rounding errors | Use stable formulas |
| Catastrophic cancellation | Rewrite expressions algebraically |
| NaN/Inf | Detect and impute/drop |
| Overflow in softmax | Compute from logits directly |

---

## Next Steps

- [Gradient Descent](../gradient-descent/notes.md) — optimization fundamentals
- [Multiple Linear Regression](../multiple-linear-regression/notes.md) — vectorization
