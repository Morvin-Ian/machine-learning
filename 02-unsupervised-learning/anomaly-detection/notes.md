# Anomaly Detection

Techniques for identifying rare, unusual, or suspicious patterns in data.

---

## What is Anomaly Detection?

**Anomaly detection** (also called outlier detection) identifies data points that deviate significantly from the expected pattern.

```
Normal data points:        With anomaly:
    ●●●●●●                    ●●●●●●
    ●●●●●●                    ●●●●●●
    ●●●●●●        →           ●●●●●●
    ●●●●●●                    ●●●●●●
                                      ×  ← Anomaly!
```

**Real-world applications:**
- **Fraud detection:** Unusual credit card transactions
- **Manufacturing:** Defective products in assembly line
- **Network security:** Intrusion detection
- **Healthcare:** Abnormal patient vitals
- **Finance:** Market manipulation

---

## Types of Anomalies

### 1. Point Anomalies

Individual data points that are unusual:

```
    │
 10 │                    ×  ← Point anomaly
    │
  5 │  ●  ●  ●  ●  ●  ●
    │
  0 └─────────────────────
      1  2  3  4  5  6  7
```

**Example:** A transaction of $10,000 when typical transactions are $50-100.

### 2. Contextual Anomalies

Points that are anomalous only in specific contexts:

```
Temperature (°C)
    │
 30 │              ×  ← Anomaly (30°C in December!)
    │        ●  ●
 20 │  ●  ●      
 10 │              ●  ●  ← Normal for winter
    └─────────────────────
      Jun Jul Aug Sep Dec Jan
```

**Example:** 30°C is normal in summer but anomalous in winter.

### 3. Collective Anomalies

A group of points that together form an anomaly:

```
    │
    │        ╱╲╱╲╱╲  ← Collective anomaly (unusual burst)
    │       ╱      ╲
    │──────╱        ╲──────
    └─────────────────────
              Time
```

**Example:** A sudden spike in web server requests (possible DDoS attack).

---

## Gaussian Distribution Method

The simplest approach: assume data follows a normal distribution.

### The Idea

If data is normally distributed, most points lie within 2-3 standard deviations of the mean. Points outside this range are anomalies.

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

```
                   ╱╲
                  ╱  ╲
                 ╱    ╲
            ───╱────────╲───
               │   μ   │
           ×   │←──2σ──→│    ×
         Anomaly       Normal       Anomaly
```

### Algorithm

1. **Fit Gaussian to data:** Compute $\mu$ and $\sigma^2$ for each feature
2. **Compute probability:** For new point x, compute $p(x)$
3. **Flag anomalies:** If $p(x) < \epsilon$ (threshold), it's an anomaly

### Multivariate Gaussian

For data with correlations between features:

$$p(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

Where:
- $\mu$ = mean vector
- $\Sigma$ = covariance matrix

### Choosing the Threshold (ε)

Use a **validation set with labeled anomalies**:

1. Try different ε values
2. Compute precision, recall, F1-score
3. Choose ε that maximizes F1-score

### Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate normal data with some anomalies
np.random.seed(42)
X_normal = np.random.randn(100, 2) * 2 + 5  # Normal data
X_anomaly = np.array([[15, 15], [0, 0], [12, 3]])  # Anomalies
X = np.vstack([X_normal, X_anomaly])

# Fit Gaussian (compute mean and covariance)
mu = np.mean(X_normal, axis=0)
sigma = np.cov(X_normal.T)

# Compute probability for each point
from scipy.stats import multivariate_normal
rv = multivariate_normal(mu, sigma)
probabilities = rv.pdf(X)

# Set threshold (e.g., bottom 5% probability)
epsilon = np.percentile(probabilities, 5)
anomalies = probabilities < epsilon

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', label='Normal')
plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', s=100, marker='x', label='Anomaly')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian-based Anomaly Detection')
plt.legend()
plt.show()
```

---

## Isolation Forest

A tree-based method that isolates anomalies instead of profiling normal data.

### Key Insight

Anomalies are **easier to isolate** than normal points. Random partitioning will separate anomalies with fewer cuts.

```
Normal point (hard to isolate):     Anomaly (easy to isolate):
    ┌───────────────┐                   ┌───────────────┐
    │ ●●●│●●●●●●●●● │                   │ ●●●●●●●●●●●●● │
    │ ●●●│●●●●●●●●● │                   │ ●●●●●●●●●●●●● │
    │────┼──────────│                   │───────────────│
    │ ●●●│●  Many   │                   │               │×
    │ ●●●│  splits! │                   │    1 split!   │
    └────┴──────────┘                   └───────────────┘
```

### Algorithm

1. **Build forest:** Create multiple isolation trees
   - Randomly select a feature
   - Randomly select a split value
   - Recursively partition until each point is isolated
2. **Compute path length:** Average number of splits to isolate each point
3. **Anomaly score:** Shorter path = more anomalous

### Anomaly Score

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where:
- $h(x)$ = path length to isolate x
- $c(n)$ = normalization factor
- Score close to 1 = anomaly
- Score close to 0.5 = normal

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Fast** | O(n log n) training, O(log n) prediction |
| **Scalable** | Works well with large datasets |
| **Few parameters** | Just number of trees and sample size |
| **No distribution assumption** | Works for non-Gaussian data |

### Python Implementation

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X_normal = np.random.randn(300, 2)
X_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_anomaly])

# Train Isolation Forest
clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expected proportion of anomalies
    random_state=42
)
predictions = clf.fit_predict(X)

# -1 = anomaly, 1 = normal
anomalies = predictions == -1

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', alpha=0.6, label='Normal')
plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', s=100, marker='x', label='Anomaly')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Isolation Forest Anomaly Detection')
plt.legend()
plt.show()

# Get anomaly scores
scores = clf.decision_function(X)  # Lower = more anomalous
plt.hist(scores, bins=50)
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.show()
```

---

## One-Class SVM

**Support Vector Machine** trained on only normal data. Learns a boundary around normal points.

### How It Works

1. Map data to high-dimensional space (kernel trick)
2. Find hyperplane that separates data from origin with maximum margin
3. Points on the "wrong" side of the hyperplane are anomalies

```
Feature space:
    ┌─────────────────┐
    │   ╱─────────╲   │
    │  ╱ ●●●●●●●●● ╲  │  ← Boundary around normal data
    │ │  ●●●●●●●●●  │ │
    │  ╲ ●●●●●●●●● ╱  │
    │   ╲─────────╱   │
    │         ×       │  ← Anomaly (outside boundary)
    └─────────────────┘
```

### Python Implementation

```python
from sklearn.svm import OneClassSVM
import numpy as np

# Generate data
np.random.seed(42)
X_train = np.random.randn(100, 2)  # Train on normal data only
X_test = np.vstack([
    np.random.randn(50, 2),        # Normal
    np.random.randn(10, 2) * 3 + 5  # Anomalies
])

# Train One-Class SVM
clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
clf.fit(X_train)

# Predict
predictions = clf.predict(X_test)
anomalies = predictions == -1

print(f"Detected {anomalies.sum()} anomalies out of {len(X_test)} points")
```

---

## Comparison of Methods

| Method | Best For | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Gaussian** | Simple, univariate | Easy, interpretable | Assumes normality |
| **Isolation Forest** | General use | Fast, no assumptions | Parameter tuning |
| **One-Class SVM** | High-dimensional | Flexible kernel | Slow on large data |
| **LOF** | Varying density | Local outliers | Slow, many parameters |

### Decision Guide

```
What type of data?
    │
    ├── Low-dimensional, Gaussian-like
    │       → Gaussian Method
    │
    ├── High-dimensional, tree-friendly
    │       → Isolation Forest
    │
    ├── Complex boundaries needed
    │       → One-Class SVM
    │
    └── Local density variations
            → Local Outlier Factor (LOF)
```

---

## Evaluation Metrics

Anomaly detection is often evaluated like classification:

- **Precision:** Of detected anomalies, how many are real?
- **Recall:** Of real anomalies, how many did we detect?
- **F1-Score:** Balance of precision and recall

> [!IMPORTANT]
> **Challenge:** Labels are often unavailable or unreliable. Domain expertise is crucial for evaluating results!

---

## Best Practices

### 1. Feature Engineering

Good features make anomalies more detectable:
- Time-based patterns (hour, day of week)
- Aggregations (transaction count, average amount)
- Ratios and differences

### 2. Threshold Tuning

- Start conservative (low false positives)
- Adjust based on business cost of missed anomalies vs false alarms
- Consider different thresholds for different severity levels

### 3. Ensemble Methods

Combine multiple anomaly detectors:
```python
# Combine scores from multiple methods
score_if = isolation_forest.decision_function(X)
score_lof = local_outlier_factor.negative_outlier_factor_
combined_score = (normalize(score_if) + normalize(score_lof)) / 2
```

---

## Quick Reference

| Concept | Definition |
|---------|------------|
| **Point anomaly** | Single unusual data point |
| **Contextual anomaly** | Unusual in specific context |
| **Collective anomaly** | Group that's unusual together |
| **Contamination** | Expected proportion of anomalies |
| **Isolation Forest** | Trees isolate anomalies quickly |
| **One-Class SVM** | Learns boundary around normal data |

---

## Summary

1. **Anomalies** are rare, unusual patterns in data
2. **Gaussian method** works for simple, normally-distributed data
3. **Isolation Forest** is the go-to for general-purpose detection
4. **Threshold selection** depends on business trade-offs
5. **Evaluation** is challenging without labels—use domain expertise

---

## Next Steps

- [Clustering](../clustering/notes.md) - Anomalies often stand alone (no cluster)
- [Dimensionality Reduction](../dimensionality-reduction/notes.md) - Reduce noise before detection
