# Unsupervised Learning

Machine learning algorithms that discover patterns in unlabeled data.

---

## Overview

Unlike supervised learning, unsupervised learning works with **unlabeled data**—the algorithm must find structure on its own without knowing the "right answers."

```
Supervised:                  Unsupervised:
Input → Label                Input → ???
  ●  →  "cat"                  ●  →  Find patterns!
  ○  →  "dog"                  ●●●  → Group similar items
  ●  →  "cat"                  ○○   → Discover structure
```

---

## Topics Covered

### 1. [Clustering](./clustering/notes.md)
Group similar data points together.

| Algorithm | Key Feature |
|-----------|-------------|
| **K-Means** | Fast, spherical clusters |
| **Hierarchical** | Dendrogram visualization |
| **DBSCAN** | Arbitrary shapes, handles noise |

**Applications:** Customer segmentation, image compression, document grouping

---

### 2. [Dimensionality Reduction](./dimensionality-reduction/notes.md)
Reduce the number of features while preserving information.

| Algorithm | Best For |
|-----------|----------|
| **PCA** | Linear relationships, preprocessing |
| **t-SNE** | 2D/3D visualization |
| **UMAP** | Large datasets, preserves structure |

**Applications:** Visualization, noise reduction, speeding up ML algorithms

---

### 3. [Anomaly Detection](./anomaly-detection/notes.md)
Identify unusual patterns or outliers.

| Algorithm | Approach |
|-----------|----------|
| **Gaussian** | Statistical probability |
| **Isolation Forest** | Tree-based isolation |
| **One-Class SVM** | Boundary around normal data |

**Applications:** Fraud detection, manufacturing defects, network security

---

## Learning Path

```
Start here:
    ↓
┌─────────────────────┐
│    Clustering       │  ← Learn to group data
│    (K-Means first)  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Dimensionality    │  ← Learn to reduce features
│    Reduction (PCA)  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Anomaly Detection  │  ← Learn to find outliers
│ (Isolation Forest)  │
└─────────────────────┘
```

---

## Prerequisites

Complete these sections first:
- [01-supervised-learning](../01-supervised-learning) - Regression and classification basics

For deeper understanding of optimization:
- [03-deep-dives](../03-deep-dives) - Gradient descent and core algorithms

---

## Quick Comparison

| Task | Supervised | Unsupervised |
|------|------------|--------------|
| **Data** | Labeled | Unlabeled |
| **Goal** | Predict labels | Discover structure |
| **Evaluation** | Accuracy, F1, etc. | Harder (silhouette, visual) |
| **Examples** | Classification, Regression | Clustering, PCA |