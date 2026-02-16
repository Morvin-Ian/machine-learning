````markdown
# Numerical Data (Practical Notes)

This note covers practical topics related to numerical data that are useful across machine learning workflows: numeric types, precision, stability, and common pitfalls.

## Numeric Types and Precision
- Floating point: `float32` (single) and `float64` (double) are most common. `float64` offers better precision but uses more memory; deep learning often uses `float32` for speed and memory efficiency.
- Integers: `int` types are used for counts/indices. Avoid integer arithmetic when you need averages or divisions.

## Floating-Point Issues
- **Rounding errors:** Floating point numbers are approximations; sums or differences of numbers with very different magnitudes can lose precision.
- **Catastrophic cancellation:** Subtracting nearly equal numbers can eliminate significant digits; rewrite expressions to avoid this when possible.
- **NaN/Inf:** Check for NaN or infinite values during preprocessing — they break most algorithms.

## Numerical Stability in ML Algorithms
- Compute losses from raw logits when possible (e.g., stable softmax + cross-entropy) instead of converting to probabilities and then taking logs.
- Use numerically stable functions from libraries (e.g., `scipy.special.logsumexp`, `numpy.matmul`/`@` for matrix mult).

## Memory & Type Choices
- For very large datasets or GPU training, prefer `float32`; for small numerical experiments or high-precision needs, use `float64`.
- When converting types, do so explicitly and consistently (e.g., `X.astype(np.float32)`) and ensure downstream code expects that dtype.

## Handling Large or Small Values
- Standardize or normalize features to avoid extremely large or tiny values that can harm optimization.
- When computing exponentials, clip inputs (or use stable implementations) to avoid overflow.

## Summary Checklist
- Ensure the correct dtype (`float32` vs `float64`) for your task.
- Check for NaN/Inf and handle missing values before training.
- Prefer stable numerical routines (log-sum-exp, stable softmax) for probability computations.
- Standardize features when appropriate to improve numeric conditioning.

````
