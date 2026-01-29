# Unsupervised Learning

Machine learning algorithms that discover patterns in unlabeled data.

## Topics Covered

- **[Clustering](./clustering/notes.md)** - Grouping similar data points
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN

- **[Dimensionality Reduction](./dimensionality-reduction/notes.md)** - Reducing feature count
  - Principal Component Analysis (PCA)
  - t-SNE for visualization

- **[Anomaly Detection](./anomaly-detection/notes.md)** - Finding outliers
  - Gaussian-based methods
  - Isolation Forest
  - One-Class SVM

## Structure

```
02-unsupervised-learning/
├── README.md
├── notes.md                    (Overview)
├── clustering/
│   └── notes.md               (K-Means, Hierarchical, DBSCAN)
├── dimensionality-reduction/
│   └── notes.md               (PCA, t-SNE)
└── anomaly-detection/
    └── notes.md               (Isolation Forest, etc.)
```

## Purpose

This section explores algorithms that work without labeled training data, focusing on discovering hidden structures and patterns in datasets.

## Prerequisites

Complete the following sections first:
- [01-supervised-learning](../01-supervised-learning) - Supervised learning techniques

For deeper understanding:
- [03-deep-dives](../03-deep-dives) - Advanced mathematical concepts

## Applications

| Technique | Use Cases |
|-----------|-----------|
| **Clustering** | Customer segmentation, image compression, document grouping |
| **Dimensionality Reduction** | Visualization, data compression, noise reduction |
| **Anomaly Detection** | Fraud detection, manufacturing defects, intrusion detection |

## Learning Order

1. Start with **Clustering** (K-Means) for intuition on grouping data
2. Progress to **Dimensionality Reduction** (PCA) for preprocessing
3. Explore **Anomaly Detection** for finding outliers
