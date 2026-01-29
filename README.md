# Machine Learning Learning Path

A comprehensive, structured curriculum for learning machine learning from basics to advanced theory.

## 📚 Project Structure

This repository is organized into a clear learning progression with three main sections:

### [01-Supervised Learning](./01-supervised-learning) — **Start Here!**
Algorithms that learn from labeled training data to make predictions. This is the best starting point.

- **[Linear Regression](./01-supervised-learning/linear-regression/notes.md)** - Predicting continuous values
- **[Logistic Regression](./01-supervised-learning/logistic-regression/notes.md)** - Binary and multi-class classification
- **[Classification](./01-supervised-learning/classification/notes.md)** - Evaluation metrics and thresholds

### [02-Unsupervised Learning](./02-unsupervised-learning)
Algorithms that discover patterns in unlabeled data.

- **[Clustering](./02-unsupervised-learning/clustering/notes.md)** - K-Means, Hierarchical, DBSCAN
- **[Dimensionality Reduction](./02-unsupervised-learning/dimensionality-reduction/notes.md)** - PCA, t-SNE
- **[Anomaly Detection](./02-unsupervised-learning/anomaly-detection/notes.md)** - Isolation Forest, One-Class SVM

### [03-Deep Dives](./03-deep-dives)
Advanced mathematical concepts and in-depth explanations of core algorithms.

- **[Gradient Descent](./03-deep-dives/gradient-descent/notes.md)** - The optimization algorithm that powers ML
  - Intuitive explanation with visual analogies
  - Mathematical derivation and update rules
  - Variants: Batch, Stochastic, Mini-batch
  - Advanced optimizers: Momentum, Adam

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip or uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd machine-learning

# Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync
```

### Recommended Learning Order

```
┌─────────────────────────────────────────┐
│  1. SUPERVISED LEARNING (Start Here!)  │
│     Linear Regression → Logistic       │
│     Regression → Classification         │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  2. UNSUPERVISED LEARNING              │
│     Clustering → Dimensionality        │
│     Reduction → Anomaly Detection      │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  3. DEEP DIVES (Reference as needed)   │
│     Gradient Descent (optimization)    │
│     [More topics coming...]            │
└─────────────────────────────────────────┘
```

1. **Start with**: [01-supervised-learning](./01-supervised-learning)
   - Begin with linear regression for regression basics
   - Progress to logistic regression for classification
   - Learn classification metrics and evaluation

2. **Then explore**: [02-unsupervised-learning](./02-unsupervised-learning)
   - Start with clustering (K-Means)
   - Learn dimensionality reduction (PCA)
   - Explore anomaly detection

3. **Reference as needed**: [03-deep-dives](./03-deep-dives)
   - Deep dive into gradient descent when you want to understand optimization
   - Use as reference when you encounter these concepts in practice

## 📋 Dependencies

See `pyproject.toml` for full list. Key packages include:

- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib & plotly** - Data visualization
- **scikit-learn** - Machine learning algorithms
- **tensorflow & keras** - Deep learning

## 📖 How to Use This Repository

Each section contains:
- **notes.md** - Detailed explanations, theory, and examples
- **code files** - Practical implementations
- **README.md** - Section-specific guidance and prerequisites

Start with the notes to understand the theory, then explore the code implementations.

## 🔗 File Structure Overview

```
machine-learning/
├── README.md                           (This file)
├── pyproject.toml                      (Project configuration)
│
├── 01-supervised-learning/             ← START HERE
│   ├── README.md
│   ├── linear-regression/
│   │   ├── main.py
│   │   ├── model.py
│   │   └── notes.md
│   ├── logistic-regression/
│   │   └── notes.md
│   └── classification/
│       └── notes.md
│
├── 02-unsupervised-learning/
│   ├── README.md
│   ├── notes.md                        (Overview)
│   ├── clustering/
│   │   └── notes.md
│   ├── dimensionality-reduction/
│   │   └── notes.md
│   └── anomaly-detection/
│       └── notes.md
│
└── 03-deep-dives/                      ← Reference material
    ├── README.md
    └── gradient-descent/
        └── notes.md                    (Comprehensive GD guide)
```

## 💡 Tips for Success

1. **Follow the order** - Start with supervised learning, the most intuitive introduction
2. **Understand the theory** - Read the notes before running code
3. **Experiment** - Modify code examples and explore variations
4. **Use deep dives** - Reference the advanced topics when you want to understand "how it works"
5. **Practice** - Implement algorithms from scratch when possible

## 🤝 Contributing

Feel free to improve this learning path! Suggestions are welcome.

## 📝 License

[Add your license information here]

---

**Happy Learning!** 🎓

Start with [01-supervised-learning/linear-regression](./01-supervised-learning/linear-regression/notes.md) and follow the learning path outlined above.
