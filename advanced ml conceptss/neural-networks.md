```markdown
# Neural Networks — Practical Overview

## What is a Neural Network?
An artificial neural network (ANN) is a parametric function approximator inspired by biological neurons. It maps inputs to outputs via layers of interconnected units (neurons) with learnable weights and biases.

## Core Components
- **Neuron:** Computes a weighted sum of inputs plus bias, passes result through an activation function.
- **Layer:** Collection of neurons. Common layers: dense (fully connected), convolutional, recurrent, normalization, attention.
- **Activation functions:** Non-linearities enabling networks to learn complex functions. Common choices:
  - ReLU: `max(0, x)` — simple, efficient, avoids vanishing gradients in many nets
  - Leaky ReLU / ELU / GELU — variations to fix dead-ReLU problems
  - Sigmoid / Tanh — useful in output layers or small nets, but saturate for large inputs

## Forward and Backward Pass
- **Forward pass:** Compute outputs from inputs through layers.
- **Loss:** Quantifies error (e.g., MSE for regression, cross-entropy for classification).
- **Backpropagation:** Use chain rule to compute gradients of loss w.r.t. parameters.
- **Optimization:** Update parameters with optimizers (SGD, SGD+Momentum, Adam, AdamW).

## Architecture Patterns
- **MLP (Multi-Layer Perceptron):** Dense layers, good for tabular data and simple tasks.
- **CNN (Convolutional Neural Network):** Convolutions + pooling, for images and local patterns.
- **RNN / LSTM / GRU:** Sequence models (previously common for NLP/time series), now often replaced by transformers for many tasks.
- **Transformers:** Self-attention based; state-of-the-art for NLP and increasingly for vision and multimodal tasks.

## Regularization and Generalization
- **Weight decay (L2)** penalizes large weights.
- **Dropout** randomly disables neurons during training to reduce co-adaptation.
- **Early stopping** monitors validation loss to prevent overfitting.
- **Data augmentation** increases diversity of the training set (images, audio, text augmentation).

## Training Best Practices
- Normalize or standardize inputs; batch normalization or layer normalization helps training stability.
- Use appropriate batch size; tune learning rate (often the most important hyperparameter).
- Use learning rate schedules (cosine decay, step decay, warmup) for better convergence.
- Monitor training curves (train vs validation loss) and metrics; use checkpoints.

## Initialization & Stability
- Use standard initializations (He for ReLU, Xavier for tanh/sigmoid) to avoid vanishing/exploding activations.
- Gradient clipping helps with exploding gradients (especially for RNNs or large models).

## Debugging Tips
- Check data pipeline first (labels, shuffling, leaks).
- Overfit a tiny subset (e.g., 100 samples) — model should fit it; if not, there's a bug.
- Inspect gradient norms, weight distributions, and activations for anomalies.

## When to Use Neural Networks
- Large, complex datasets with non-linear structure (images, audio, raw text).
- When feature engineering is hard and representation learning is beneficial.

## Quick References
- Loss functions: cross-entropy, MSE, BCE
- Optimizers: SGD, Adam, AdamW
- Normalization: BatchNorm, LayerNorm
- Frameworks: PyTorch, TensorFlow, JAX

```
