# Supervised Learning — Start Here! 🚀

Machine learning algorithms that learn from labeled training data.

## Why Start Here?

Supervised learning is the most intuitive introduction to ML:
- You have **labeled data** (inputs with known correct outputs)
- The goal is clear: **predict the label for new data**
- Results are easy to evaluate

## Topics Covered

### 1. [Linear Regression](./linear-regression/notes.md) — **First Topic**
Predict continuous values. Perfect for understanding ML fundamentals.

- What is regression?
- Loss functions (MSE, MAE)
- How models learn (gradient descent basics)
- Multiple features

### 2. [Logistic Regression](./logistic-regression/notes.md)
Binary classification using probability.

- Sigmoid function
- Log loss / Cross-entropy
- Decision boundaries
- Regularization

### 3. [Classification](./classification/notes.md)
How to evaluate classification models.

- Confusion matrix
- Precision, Recall, F1-Score
- ROC curves and AUC
- Multi-class strategies

## Learning Order

```
Start here:
    ↓
┌─────────────────────┐
│  Linear Regression  │  ← Learn regression + how models train
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Logistic Regression │  ← Apply to classification
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Classification    │  ← Learn how to evaluate models
└──────────┬──────────┘
           ↓
    Continue to 02-unsupervised-learning
```

## Structure

```
01-supervised-learning/
├── README.md              (This file)
├── linear-regression/
│   ├── main.py           (Implementation)
│   ├── model.py          (Model code)
│   └── notes.md          (Theory + examples)
├── logistic-regression/
│   └── notes.md
└── classification/
    └── notes.md
```

## Prerequisites

- Basic Python programming
- High school math (algebra, basic statistics)

For deeper mathematical understanding, see [03-deep-dives](../03-deep-dives/README.md).

## Next Steps

After completing supervised learning:
- [02-unsupervised-learning](../02-unsupervised-learning) - Discover patterns in unlabeled data
- [03-deep-dives/gradient-descent](../03-deep-dives/gradient-descent/notes.md) - Deep dive into optimization
