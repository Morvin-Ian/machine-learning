# Dimensionality Reduction

Techniques for reducing the number of features while preserving important information.

---

## Why Reduce Dimensions?

### The Curse of Dimensionality

As dimensions increase, data becomes increasingly sparse:

```
1D:  ●●●●●●●●●●  (dense)
2D:  ●  ●  ●
       ●  ●     (sparser)
        ●
3D+: Points become very far apart!
```

**Problems with high dimensions:**
- **Computational cost:** More features = more time and memory
- **Overfitting:** Too many features relative to samples
- **Distance metrics fail:** All points become equidistant
- **Visualization impossible:** Can't plot more than 3D

### Benefits of Dimensionality Reduction

1. **Visualization:** Project data to 2D/3D for plotting
2. **Speed:** Faster algorithms with fewer features
3. **Noise reduction:** Remove uninformative features
4. **Storage:** Compress data efficiently
5. **Better models:** Reduce overfitting

---

## Principal Component Analysis (PCA)

The most widely used dimensionality reduction technique. Finds orthogonal directions of maximum variance.

### Intuition

PCA finds the axes along which your data varies most:

```
Original:                  After PCA:
    y                         PC2
    │    ●  ●                  │    ●
    │  ●  ●  ●                 │  ●
    │    ●  ●    →         ────●────────── PC1
    │  ●    ●                  │    ●
    └──────────x               │  ●

The data varies most along the diagonal (PC1).
PC2 captures remaining variance perpendicular to PC1.
```

**Key idea:** If data varies along certain directions more than others, those directions capture the most information.

### The Algorithm

**Input:** Data matrix $X$ with $m$ samples and $n$ features.

**Step 1: Standardize the data**
$$X_{std} = \frac{X - \mu}{\sigma}$$

**Step 2: Compute covariance matrix**
$$C = \frac{1}{m-1} X_{std}^T X_{std}$$

**Step 3: Find eigenvalues and eigenvectors of C**
- Eigenvectors = principal components (directions)
- Eigenvalues = variance explained by each component

**Step 4: Select top k components**
Sort by eigenvalue magnitude, keep top $k$.

**Step 5: Project data**
$$X_{reduced} = X_{std} \cdot W_k$$

Where $W_k$ is the matrix of top $k$ eigenvectors.

### Choosing the Number of Components

**Explained Variance Ratio:** The percentage of total variance captured by each component.

```
Explained Variance
        │
  40%   │████████
  30%   │████████████████
  20%   │███████████████████████
  10%   │██████████████████████████████
        └────────────────────────────────
          PC1    PC2    PC3    PC4

Cumulative: 40% → 70% → 90% → 100%
```

**Rule of thumb:** Keep enough components to explain 90-95% of variance.

### Mathematical Details

**Variance along direction $\vec{v}$:**
$$\text{Var}(X \cdot \vec{v}) = \vec{v}^T C \vec{v}$$

**Maximizing variance** (subject to $||\vec{v}|| = 1$) gives:
$$C \vec{v} = \lambda \vec{v}$$

This is the eigenvalue equation! The eigenvector with largest eigenvalue captures most variance.

### PCA Properties

| Property | Description |
|----------|-------------|
| **Orthogonal** | Principal components are perpendicular |
| **Ordered** | PC1 has most variance, PC2 second most, etc. |
| **Linear** | Only finds linear relationships |
| **Unsupervised** | Doesn't use labels |
| **Reversible** | Can reconstruct (approximate) original data |

### When PCA Works Well

✅ Data has linear correlations between features
✅ Most variance is concentrated in few directions
✅ Goal is compression or speed-up

### When PCA Fails

❌ Non-linear relationships (use t-SNE or kernel PCA)
❌ All dimensions equally important
❌ Categorical or sparse data

### Python Implementation

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
X = np.random.randn(200, 5)
X[:, 1] = X[:, 0] * 2 + np.random.randn(200) * 0.3  # Correlated features
X[:, 2] = X[:, 0] * 1.5 + np.random.randn(200) * 0.5

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, 6), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each PC')

plt.subplot(1, 2, 2)
plt.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.tight_layout()
plt.show()

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA (explains {pca_2d.explained_variance_ratio_.sum():.1%} of variance)')
plt.show()
```

---

## t-SNE (t-distributed Stochastic Neighbor Embedding)

Non-linear technique designed specifically for **visualization** in 2D or 3D.

### How t-SNE Works

1. **Compute pairwise similarities** in high-dimensional space (Gaussian distribution)
2. **Compute similarities** in low-dimensional space (t-distribution)
3. **Minimize divergence** between the two distributions using gradient descent

**Key insight:** t-SNE preserves local structure—nearby points stay nearby.

### PCA vs t-SNE

```
Original (3 clusters):

PCA result:                  t-SNE result:
    ● ○ ○                       ●●●
  ●●●○○○                        ●●●
    ● ○ ○                    ○○○    ▲▲▲
  ● ● ▲ ▲                    ○○○    ▲▲▲
    ▲ ▲                          

Clusters overlap             Clusters clearly separated
(preserves global variance)  (preserves local structure)
```

### Key Parameter: Perplexity

**Perplexity** ≈ expected number of nearest neighbors. Typical values: 5-50.

| Perplexity | Effect |
|------------|--------|
| Too low (5) | Small, disconnected clusters |
| Good (30) | Balanced local/global structure |
| Too high (100) | Clusters merge together |

### Important Caveats

> [!WARNING]
> **t-SNE is for visualization only!**
> - Results vary with random initialization—run multiple times
> - Distances between clusters are NOT meaningful
> - Not suitable for preprocessing before other algorithms
> - Computationally expensive for large datasets

### When to Use t-SNE vs PCA

| Use PCA When... | Use t-SNE When... |
|-----------------|-------------------|
| Need reproducible results | Only need visualization |
| Want interpretable components | Data has non-linear structure |
| Preprocessing for ML | Exploring cluster separability |
| Large datasets | Moderate-sized datasets |

### Python Implementation

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset (8x8 images = 64 features)
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    n_iter=1000
)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE of Handwritten Digits')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Compare with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
axes[0].set_title('PCA')
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
axes[1].set_title('t-SNE')
plt.tight_layout()
plt.show()
```

---

## Other Techniques (Brief Overview)

### UMAP (Uniform Manifold Approximation and Projection)

Modern alternative to t-SNE:
- **Faster** than t-SNE
- **Better preserves global structure**
- Can be used for preprocessing (unlike t-SNE)
- Good for very large datasets

```python
from umap import UMAP

reducer = UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
```

### Kernel PCA

PCA for non-linear data using the kernel trick:
- Maps data to higher dimension
- Applies PCA in that space
- Returns to lower dimension

### Autoencoders

Neural network approach:
- **Encoder:** Compresses input to lower dimension
- **Decoder:** Reconstructs original from compressed
- Learns non-linear transformations
- More flexible but requires more data

---

## Quick Reference

| Method | Type | Best For | Preserves |
|--------|------|----------|-----------|
| **PCA** | Linear | General reduction | Global variance |
| **t-SNE** | Non-linear | 2D/3D visualization | Local structure |
| **UMAP** | Non-linear | Large datasets | Local + some global |
| **Kernel PCA** | Non-linear | Non-linear patterns | Non-linear variance |

### Decision Flowchart

```
Need dimensionality reduction?
        │
        ▼
  For visualization?
   ├── YES → Is data non-linear?
   │           ├── YES → t-SNE or UMAP
   │           └── NO  → PCA (2-3 components)
   │
   └── NO → For preprocessing/speed?
              ├── YES → PCA (95% variance)
              └── NO  → Consider domain-specific methods
```

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Curse of dimensionality** | High-D data is sparse, distances meaningless |
| **PCA** | Linear, finds max variance directions |
| **Explained variance** | Keep 90-95% of variance |
| **t-SNE** | Non-linear, for visualization only |
| **Perplexity** | t-SNE's main parameter (try 5-50) |

---

## Next Steps

- [Clustering](../clustering/notes.md) - Apply after reducing dimensions
- [Anomaly Detection](../anomaly-detection/notes.md) - Find outliers in reduced space
