# Deep Dives

In-depth mathematical concepts and detailed explanations of core machine learning algorithms.

## Purpose

This section contains **advanced theoretical content** that explains the "how" and "why" behind machine learning algorithms. Reference these topics when you want a deeper understanding of the foundations.

## Topics Covered

- **[Gradient Descent](./gradient-descent/notes.md)** - The optimization algorithm that powers ML
  - Intuitive "ball rolling downhill" analogy
  - Mathematical derivation step-by-step
  - Variants: Batch, Stochastic, Mini-batch
  - Learning rate and convergence
  - Advanced optimizers: Momentum, RMSprop, Adam

## When to Read These

| Scenario | Recommendation |
|----------|----------------|
| Learning ML for the first time | Start with `01-supervised-learning` instead |
| "How does training actually work?" | Read **Gradient Descent** |
| Model not converging | Reference **Gradient Descent** troubleshooting |
| Want to understand optimizers | Read **Advanced Optimizers** section |

## Structure

```
03-deep-dives/
├── README.md          (This file)
└── gradient-descent/
    └── notes.md       (Comprehensive 400+ line guide)
```

## Prerequisites

These topics assume familiarity with:
- Basic calculus (derivatives)
- Linear algebra basics (vectors, matrices)
- Python programming

## Suggested Reading Order

1. Read supervised learning topics first (`01-supervised-learning`)
2. When you encounter gradient descent in those notes, come here for the full explanation
3. Return to supervised/unsupervised learning with deeper understanding

## Coming Soon

Future deep dives planned:
- Regularization (L1, L2, Elastic Net)
- Bias-Variance Tradeoff
- Cross-Validation
- Feature Engineering
- Neural Network Backpropagation

---

**Note:** These are reference materials, not the starting point. Begin with [01-supervised-learning](../01-supervised-learning/README.md).
