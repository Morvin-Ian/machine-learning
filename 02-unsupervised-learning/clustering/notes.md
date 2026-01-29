# Clustering

Clustering is the task of grouping similar data points together without using predefined labels. It's one of the most fundamental unsupervised learning techniques.

---

## What is Clustering?

**Clustering** automatically discovers natural groupings in data. Points within a cluster are more similar to each other than to points in other clusters.

```
Before Clustering:            After Clustering:
    •  •                         ●  ●
  •    •  •                    ●    ●  ●
    •  •                         ●  ●        Cluster 1
                        →
      • •                          ○ ○
    •  •  •                      ○  ○  ○     Cluster 2
       •                            ○
```

**Use cases:**
- Customer segmentation
- Image compression
- Anomaly detection
- Document topic discovery
- Biological taxonomy

---

## K-Means Clustering

The most popular clustering algorithm. It partitions data into exactly $K$ clusters.

### Algorithm Steps

1. **Initialize:** Randomly select $K$ cluster centers (centroids)
2. **Assign:** Assign each point to the nearest centroid
3. **Update:** Move each centroid to the mean of its assigned points
4. **Repeat:** Steps 2-3 until centroids stop moving

### Visual Walkthrough

```
Step 1: Initialize (K=2)         Step 2: Assign to nearest
    ×₁  •  •                         ×₁  ●  ●
  •    •  •                        ●    ●  ●
    •  •                             ●  ●
      • •                              ○ ○
    •  •  ×₂                         ○  ○  ×₂
       •                                ○

Step 3: Update centroids         Step 4: Repeat until stable
    ×₁  ●  ●                         ×₁  ●  ●
  ●    ●  ●                        ●    ●  ●
    ●  ●         ←centroids          ●  ●
      ○ ○          move→               ○ ○
    ○  ○  ○                          ○  ○  ○
       ×₂                               ×₂

× = centroid, ● = cluster 1, ○ = cluster 2
```

### The Math

**Objective:** Minimize the sum of squared distances from points to their cluster centroids.

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Where:
- $K$ = number of clusters
- $C_k$ = set of points in cluster $k$
- $\mu_k$ = centroid of cluster $k$
- $||x_i - \mu_k||^2$ = squared Euclidean distance

**Assignment step:**
$$c^{(i)} = \arg\min_k ||x^{(i)} - \mu_k||^2$$

**Update step:**
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

### Choosing K: The Elbow Method

Plot the cost function $J$ for different values of $K$:

```
Cost J
  │
  │╲
  │ ╲
  │  ╲_____
  │    ╲___╲_____ ← "Elbow" (K=3)
  │         ╲____╲____
  └──────────────────────→ K
    1   2   3   4   5   6
```

Choose $K$ at the "elbow" where the cost stops decreasing significantly.

### Silhouette Score

Measures how similar a point is to its own cluster vs other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean distance to points in same cluster
- $b(i)$ = mean distance to points in nearest other cluster

**Interpretation:**
- $s ≈ 1$: Well-clustered
- $s ≈ 0$: On boundary between clusters
- $s ≈ -1$: Probably in wrong cluster

### K-Means++ Initialization

Random initialization can lead to poor clustering. **K-Means++** improves this:

1. Choose first centroid randomly from data points
2. For each remaining centroid:
   - Compute distance $D(x)$ from each point to nearest existing centroid
   - Choose next centroid with probability proportional to $D(x)^2$
3. Proceed with standard K-Means

**Result:** Better initialization → faster convergence → better clusters.

### Limitations of K-Means

| Limitation | Explanation |
|------------|-------------|
| Must specify $K$ | Need to know number of clusters beforehand |
| Spherical clusters only | Assumes clusters are roughly circular |
| Sensitive to outliers | Outliers can pull centroids away |
| Sensitive to initialization | Different runs may give different results |
| Uniform cluster sizes | Tends to create equal-sized clusters |

### Python Implementation

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [10, 0]
])

# Train K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker='X', s=200, c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

# Find optimal K using elbow method
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (cost)')
plt.title('Elbow Method')
plt.show()
```

---

## Hierarchical Clustering

Builds a hierarchy of clusters—either from bottom-up (agglomerative) or top-down (divisive).

### Agglomerative Clustering (Bottom-Up)

1. Start with each point as its own cluster
2. Find the two closest clusters
3. Merge them into one cluster
4. Repeat until only one cluster remains

### Dendrograms

A **dendrogram** visualizes the hierarchical clustering process:

```
Height (distance)
    │
  6 ├───────────────────┬───────────────────┐
    │                   │                   │
  4 ├───────┬───────┐   │       ┌───────────┤
    │       │       │   │       │           │
  2 ├───┬───┤   ┌───┤   ├───┐   │   ┌───────┤
    │   │   │   │   │   │   │   │   │       │
  0 └───┴───┴───┴───┴───┴───┴───┴───┴───────┴───
        A   B   C   D   E   F   G   H   I   J
                    Data Points

Cut at height 4 → 3 clusters: {A,B,C,D}, {E,F,G}, {H,I,J}
```

**Cutting the dendrogram:** Draw a horizontal line at desired height to get clusters.

### Linkage Types

How do we measure distance between clusters?

| Linkage | Definition | Characteristics |
|---------|------------|-----------------|
| **Single** | Min distance between any two points | Creates elongated clusters |
| **Complete** | Max distance between any two points | Creates compact, spherical clusters |
| **Average** | Mean distance between all pairs | Balance between single/complete |
| **Ward** | Minimizes variance within clusters | Creates similarly sized clusters |

```
Single Linkage:     Complete Linkage:    Average Linkage:
   ●●   ○○             ●●   ○○              ●●   ○○
   ●●   ○○             ●●   ○○              ●●   ○○
     ↖↗                  ↖---↗                ↖-↗
   Closest             Farthest              Mean of
   pair                pair                  all pairs
```

### When to Use Hierarchical Clustering

| Pros | Cons |
|------|------|
| No need to specify $K$ upfront | $O(n^2)$ space, $O(n^3)$ time |
| Produces dendrogram for visualization | Sensitive to noise and outliers |
| Works with any distance metric | Cannot easily handle large datasets |

### Python Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.randn(50, 2)

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Get cluster assignments
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Hierarchical Clustering (3 clusters)')
plt.show()
```

---

## DBSCAN

**Density-Based Spatial Clustering of Applications with Noise.** Finds clusters based on density, not distance to centroids.

### Key Concepts

- **eps (ε):** Maximum distance between neighbors
- **min_samples:** Minimum points to form a dense region

**Point types:**
- **Core point:** Has ≥ min_samples within eps distance
- **Border point:** Within eps of a core point, but not core itself
- **Noise point:** Neither core nor border—outlier!

```
        ●●●
       ●○●●●               ● = Core point (≥3 neighbors)
        ●●●                ○ = Border point
                           • = Noise (outlier)
    •         
              •
```

### Algorithm

1. For each unvisited point:
   - If it's a core point, create new cluster
   - Add all density-reachable points to this cluster
   - If it's noise, label as outlier (can change later if reachable)
2. Repeat until all points visited

### Advantages over K-Means

| Feature | K-Means | DBSCAN |
|---------|---------|--------|
| **Cluster shape** | Spherical only | Any shape |
| **Must specify K** | Yes | No (finds K automatically) |
| **Handles outliers** | Poorly | Labels them as noise |
| **Cluster sizes** | Similar | Can vary |

### When to Use DBSCAN

✅ Use when:
- Clusters have irregular shapes
- You don't know number of clusters
- Data has outliers/noise
- Clusters may have different densities

❌ Avoid when:
- Clusters have varying densities
- Data is very high-dimensional (eps becomes meaningless)

### Python Implementation

```python
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate data with clusters of different shapes
np.random.seed(42)
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.05)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot (noise points labeled as -1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'DBSCAN Clustering ({len(set(labels)) - 1} clusters found)')
plt.colorbar(label='Cluster')
plt.show()

# Compare with K-Means (fails on non-spherical data)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=dbscan.fit_predict(X), cmap='viridis')
axes[0].set_title('DBSCAN (handles moons)')
axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].set_title('K-Means (fails on moons)')
plt.show()
```

---

## Quick Reference: Choosing a Clustering Algorithm

| Scenario | Best Algorithm |
|----------|----------------|
| Known number of spherical clusters | **K-Means** |
| Need to visualize hierarchy | **Hierarchical Clustering** |
| Arbitrary-shaped clusters with noise | **DBSCAN** |
| Very large dataset | **K-Means** or **Mini-batch K-Means** |
| Clusters of varying densities | **HDBSCAN** or **OPTICS** |

---

## Summary

| Algorithm | Key Parameter(s) | Cluster Shape | Handles Outliers |
|-----------|------------------|---------------|------------------|
| **K-Means** | K (number) | Spherical | No |
| **Hierarchical** | Linkage type, cut height | Any | No |
| **DBSCAN** | eps, min_samples | Any | Yes |

---

## Next Steps

- [Dimensionality Reduction](../dimensionality-reduction/notes.md) - Reduce features before clustering
- [Anomaly Detection](../anomaly-detection/notes.md) - Identify outliers in data
