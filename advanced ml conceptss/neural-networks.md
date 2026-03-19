# Neural Networks — Practical Overview

## Motivation & Use Cases
Neural networks exist because many real‑world problems involve highly nonlinear, high‑dimensional relationships that are difficult to capture with hand‑crafted features. As **universal function approximators**, they can learn arbitrary mappings from input to output given enough data and capacity. Typical domains include:

- **Vision:** object recognition, segmentation, image generation
- **Speech & audio:** recognition, synthesis, enhancement
- **Natural language:** translation, summarization, question answering
- **Time series & control:** forecasting, robot policy learning
- **Any task where feature engineering is hard or representation learning is valuable**

The intuitive motivation is: rather than manually designing features, let a network **learn representations** through layers of simple computational units.

## Key Components of an Architecture

### Nodes (Neurons)
Each neuron receives one or more inputs `x_i`, computes a weighted sum plus a bias,

```
z = \sum_i w_i x_i + b
```

and then applies an **activation function** `φ(z)` to produce its output. Neurons are the basic compute units.

### Layers
- **Input layer:** passes raw features into the network (no computations aside from maybe scaling).
- **Hidden layers:** stacks of neurons that transform representations; more layers allow learning hierarchical features.
- **Output layer:** produces final predictions (scores, probabilities, regression values).

A network with two or more hidden layers is considered “deep.” Each layer’s outputs become the next layer’s inputs.

### Activation Functions
Non‑linearities enable networks to approximate complex functions:

- **ReLU:** `max(0, z)` — simple, efficient, avoids vanishing gradients in many architectures.
- **Leaky ReLU / ELU / GELU:** variations to keep a small gradient when `z<0`.
- **Sigmoid / Tanh:** smooth, useful in output layers or small nets, but saturate and slow training in deep nets.
- **Softmax:** used in classification output layers to convert raw scores to a probability distribution.

Choosing an activation depends on the task and depth of the model.

## Inference: Step‑Through Prediction
1. **Start with input vector** `x` (e.g. pixel values, feature vector).
2. **First layer:** compute `z^1 = W^1 x + b^1`, then apply φ to get `a^1 = φ(z^1)`.
3. **Propagate through hidden layers:** for `ℓ = 2…L`, compute `z^ℓ = W^ℓ a^{ℓ-1} + b^ℓ`, then `a^ℓ = φ(z^ℓ)`.
4. **Output layer:** produce raw scores `s = W^{L+1} a^L + b^{L+1}` and convert (e.g. via softmax) to probabilities.
5. **Prediction:** choose class with highest probability or return the regression value.

This “forward pass” is pure arithmetic—matrix multiplies followed by nonlinearities.

## Training & Backpropagation (Intuitive View)
Training adjusts weights and biases so that the network’s outputs match the desired targets.

- Define a **loss function** `L(y,ŷ)` that measures error (e.g., cross‑entropy for classification).
- Perform a forward pass to compute predictions and loss for a batch of training examples.
- **Backpropagation** computes gradients of the loss w.r.t. every parameter by applying the chain rule backwards through the network:
  - Start at the output layer: how does changing each weight affect the loss?
  - Propagate these sensitivities layer by layer, accumulating gradients `∂L/∂W^ℓ` and `∂L/∂b^ℓ`.
- **Optimization step:** update parameters using an algorithm such as SGD or Adam: `W ← W − η ∂L/∂W` (η is the learning rate).

Over many iterations (epochs) of presenting data, the network’s parameters converge to values that minimize the loss.

### Simple Example (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# toy dataset: y = 2x + 1 with noise
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# small network: one hidden layer
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    for xb, yb in dataloader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss {loss.item():.4f}")

# inspect learned parameters
for name, param in model.named_parameters():
    print(name, param.data)
```

This script defines a small multilayer perceptron, runs a training loop with backpropagation, and prints the loss over epochs.

## Multi‑class Classification Strategies
When a network naturally outputs scores for `k` classes (via softmax), it directly handles multi‑class classification. However, alternative schemes are:

- **One‑vs‑all (OvA):** train `k` binary classifiers, each distinguishing one class from the rest. At inference, run all classifiers and pick the one with the highest confidence.
- **One‑vs‑one (OvO):** for `k` classes, train `k(k−1)/2` binary classifiers, each discriminating between a pair of classes. Use voting or aggregated scores to decide the final class.

These strategies are often used with simpler models (e.g. SVMs) but the “softmax output” of a neural network is essentially an OvA formulation internally.

## Additional Topics & Best Practices
- **Architectural patterns:** MLPs for tabular data; CNNs for images; RNNs/LSTMs/transformers for sequences; attention mechanisms; residual connections for very deep nets.
- **Regularization:** weight decay, dropout, batch/layer normalization, data augmentation, and early stopping to improve generalization.
- **Initialization:** use schemes like He (for ReLU) or Xavier/Glorot (for sigmoid/tanh) to keep activations stable at the start of training.
- **Hyperparameters:** tune learning rate, batch size, network width/depth, and optimizer settings. Learning rate schedules and warm‑up are crucial for large models.
- **Debugging:** verify data pipelines, try over‑fitting a tiny dataset, monitor gradients & activations, visualize training/validation curves.

## When to Reach for Neural Networks
Use them when you have ample data with complex structure, when handcrafted features fall short, or when you need models capable of learning representations directly from raw inputs. For small or simple datasets, classical methods may perform just as well with far less compute.

## Next Steps

- [Embeddings](./embeddings.md) — dense vector representations
- [Intro to LLMs](./intro-to-llms.md) — large language models
- [Production ML](../real%20world%20ml/production.md) — deploy models to production```
