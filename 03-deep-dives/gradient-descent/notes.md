# Gradient Descent

The optimization algorithm that powers machine learning. This guide will take you from intuition to implementation.

---

## What is Gradient Descent?

**Gradient Descent** is an iterative optimization algorithm used to find the minimum of a function. In machine learning, we use it to find the optimal weights and biases that minimize our loss (error) function.

### The Intuitive "Ball Rolling Downhill" Analogy

Imagine you're blindfolded on a hilly landscape and need to find the lowest valley:

```
      Start (random position)
         ↓
    ╱╲   •
   ╱  ╲  │    ← Feel the slope
  ╱    ╲ ↓    ← Take a step downhill
 ╱      ╲•
╱        ╲│   ← Repeat
          ↓
    Valley (minimum) ← Stop when flat
```

**The process:**
1. **Check the slope** beneath your feet (compute the gradient)
2. **Take a step** in the downhill direction (update parameters)
3. **Repeat** until you reach the bottom (convergence)

The **gradient** tells you which direction is uphill. By moving in the **negative gradient** direction, you go downhill toward lower values.

---

## Why Do We Need It?

In machine learning, we define a **loss function** (also called cost function) that measures how wrong our model's predictions are:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

Where:
- $J$ = loss (what we want to minimize)
- $w$ = weights, $b$ = bias (parameters to optimize)
- $\hat{y}$ = predicted value
- $y$ = actual value
- $m$ = number of training examples

**Goal:** Find values of $w$ and $b$ that make $J$ as small as possible.

For simple problems, we could solve this mathematically. But for complex models with millions of parameters, gradient descent is the practical solution.

---

## The Mathematics

### The Gradient

The **gradient** is a vector of partial derivatives—it points in the direction of steepest increase:

$$\nabla J = \begin{bmatrix} \frac{\partial J}{\partial w_1} \\ \frac{\partial J}{\partial w_2} \\ \vdots \\ \frac{\partial J}{\partial w_n} \\ \frac{\partial J}{\partial b} \end{bmatrix}$$

Each partial derivative tells us: "If I increase this parameter slightly, how much does the loss change?"

### The Update Rule

To minimize the loss, we move in the **opposite** direction of the gradient:

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where:
- $:=$ means "update to"
- $\alpha$ (alpha) = **learning rate** (step size)
- $\frac{\partial J}{\partial w}$ = gradient (slope) with respect to $w$
- The **negative sign** ensures we go downhill

### Worked Example

**Given:** A simple model $\hat{y} = wx$ with one weight $w$.

**Loss function:** $J(w) = (y - wx)^2$

**Data point:** $x = 2$, $y = 6$ (so the true relationship is $y = 3x$)

**Step 1: Initialize**
- Start with random weight: $w = 0$
- Learning rate: $\alpha = 0.1$

**Step 2: Compute gradient**
$$\frac{\partial J}{\partial w} = \frac{\partial}{\partial w}(y - wx)^2 = -2x(y - wx)$$

With $w = 0$: $\frac{\partial J}{\partial w} = -2(2)(6 - 0) = -24$

**Step 3: Update weight**
$$w = 0 - 0.1 \times (-24) = 2.4$$

**Step 4: Repeat**

| Iteration | $w$ | Prediction $\hat{y}$ | Loss $J$ | Gradient |
|-----------|-----|---------------------|----------|----------|
| 0 | 0.0 | 0.0 | 36.0 | -24.0 |
| 1 | 2.4 | 4.8 | 1.44 | -4.8 |
| 2 | 2.88 | 5.76 | 0.058 | -0.96 |
| 3 | 2.976 | 5.95 | 0.002 | -0.19 |
| ... | ... | ... | ... | ... |
| ∞ | 3.0 | 6.0 | 0.0 | 0.0 |

The weight converges to $w = 3$, exactly matching the true relationship!

---

## The Learning Rate

The **learning rate** ($\alpha$) controls how big each step is. It's one of the most important hyperparameters.

### Effect of Different Learning Rates

```
Loss                     Loss                     Loss
  │                        │                        │
  │\                       │\                       │ ╱\  ╱\
  │ \                      │ \                      │╱  ╲╱  ╲ Diverging!
  │  \                     │  ╲                     │
  │   \                    │   ╲___                 │
  │    \_____              │       ╲___             │
  └──────────→ Iterations  └───────────→ Iter.     └──────────→ Iter.
  
   α = 0.001 (too small)    α = 0.01 (good)         α = 1.0 (too large)
   Very slow convergence    Smooth convergence      Overshoots and diverges
```

### Guidelines for Choosing Learning Rate

| Learning Rate | Behavior | When to Use |
|---------------|----------|-------------|
| $10^{-1}$ to $10^{-2}$ | Can be unstable | Starting point for experimentation |
| $10^{-2}$ to $10^{-3}$ | Usually good | Default for most problems |
| $10^{-3}$ to $10^{-4}$ | Slow but stable | When training is unstable |
| $10^{-5}$ or smaller | Very slow | Fine-tuning pre-trained models |

### Learning Rate Schedules

Instead of a fixed learning rate, you can decrease it over time:

**Step Decay:**
$$\alpha_t = \alpha_0 \times 0.1^{\lfloor t / \text{step} \rfloor}$$

**Exponential Decay:**
$$\alpha_t = \alpha_0 \times e^{-kt}$$

**1/t Decay:**
$$\alpha_t = \frac{\alpha_0}{1 + kt}$$

This allows fast initial progress (large steps) followed by fine-tuning (small steps).

---

## Types of Gradient Descent

### 1. Batch Gradient Descent (BGD)

Uses **all** training examples to compute the gradient before each update.

$$w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} \frac{\partial J^{(i)}}{\partial w}$$

| Pros | Cons |
|------|------|
| Stable, smooth updates | Slow for large datasets |
| Accurate gradient estimation | High memory usage |
| Guaranteed convergence for convex functions | Gets stuck in local minima |

**Best for:** Small datasets that fit in memory.

### 2. Stochastic Gradient Descent (SGD)

Uses **one** random training example per update.

$$w := w - \alpha \frac{\partial J^{(i)}}{\partial w}$$

| Pros | Cons |
|------|------|
| Fast updates | Noisy, erratic loss curve |
| Low memory usage | May not converge exactly |
| Can escape local minima | Requires learning rate tuning |
| Works with streaming data | |

**Best for:** Very large datasets, online learning.

### 3. Mini-batch Gradient Descent

Uses a **small batch** of examples (typically 32-256) per update. **Most commonly used in practice.**

$$w := w - \alpha \frac{1}{b} \sum_{i=1}^{b} \frac{\partial J^{(i)}}{\partial w}$$

| Pros | Cons |
|------|------|
| Balances speed and stability | Requires batch size tuning |
| Efficient hardware utilization (GPU) | Still somewhat noisy |
| Can escape shallow local minima | |

**Common batch sizes:** 32, 64, 128, 256

### Visual Comparison

```
                    Batch GD              Stochastic GD         Mini-batch GD
                    (smooth)              (very noisy)          (balanced)
Path to    
minimum:           ↘                      ↘↗↙↘                   ↘↗↘
                    ↘                     ↙↘↗↘                    ↘↘
                     ↘                   ↗↙↘↗↙                     ↘↗↘
                      •                    •                        •
                   
Updates              Fewer               Many                   Moderate
per epoch:           (1)                 (=m)                   (m/batch_size)
```

---

## Convergence

### What is Convergence?

**Convergence** occurs when the loss stops decreasing significantly—the algorithm has found a minimum.

### Detecting Convergence

1. **Loss threshold:** Stop when $|J_{new} - J_{old}| < \epsilon$ (e.g., $\epsilon = 10^{-6}$)
2. **Gradient threshold:** Stop when $||\nabla J|| < \epsilon$
3. **Fixed iterations:** Run for a set number of epochs

### Convex vs Non-Convex Functions

**Convex function** (bowl-shaped): Has one global minimum. Gradient descent always finds it.
```
        ╱╲
       ╱  ╲
      ╱    ╲
     ╱      ╲
    ╱   •    ╲  ← Only one minimum (global)
```

**Non-convex function**: Has multiple local minima. Gradient descent might get stuck.
```
       ╱╲   ╱╲
      ╱  ╲ ╱  ╲
     ╱    •    ╲  ← Local minimum (might get stuck here)
    ╱           ╲
   ╱      •      ╲  ← Global minimum (want to find this)
```

**Good news:** Linear regression has a convex loss function—always finds the global minimum!

---

## Common Problems and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| **Loss oscillates** | Learning rate too high | Decrease $\alpha$ |
| **Loss decreases too slowly** | Learning rate too low | Increase $\alpha$ |
| **Loss explodes (NaN)** | Learning rate way too high | Decrease $\alpha$ significantly |
| **Loss plateaus early** | Stuck in local minimum | Use SGD/mini-batch for noise, try different initialization |
| **Vanishing gradients** | Deep networks, bad activation | Use ReLU, batch norm, different architecture |
| **Exploding gradients** | Bad initialization, deep networks | Gradient clipping, weight initialization |

---

## Advanced Optimizers

Standard gradient descent has limitations. Modern optimizers add improvements:

### Momentum

Adds "velocity" to updates—accelerates in consistent directions:

$$v_t = \beta v_{t-1} + \alpha \nabla J$$
$$w := w - v_t$$

**Benefit:** Faster convergence, helps escape shallow local minima.

### RMSprop

Adapts learning rate per parameter based on recent gradient magnitudes:

$$s_t = \beta s_{t-1} + (1-\beta)(\nabla J)^2$$
$$w := w - \frac{\alpha}{\sqrt{s_t + \epsilon}} \nabla J$$

**Benefit:** Handles sparse features, prevents exploding/vanishing updates.

### Adam (Adaptive Moment Estimation)

Combines Momentum + RMSprop. **Most popular optimizer today.**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla J$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla J)^2$$
$$w := w - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters:** $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

---

## Python Implementation

### From Scratch

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Simple gradient descent for linear regression.
    
    Args:
        X: Features (m x n numpy array)
        y: Target values (m x 1 numpy array)
        learning_rate: Step size (alpha)
        iterations: Number of updates
    
    Returns:
        w: Optimized weights
        b: Optimized bias
        history: List of loss values
    """
    m, n = X.shape
    w = np.zeros((n, 1))  # Initialize weights to zero
    b = 0                  # Initialize bias to zero
    history = []
    
    for i in range(iterations):
        # Forward pass: predictions
        y_pred = X @ w + b
        
        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)
        history.append(loss)
        
        # Compute gradients
        dw = (2/m) * X.T @ (y_pred - y)  # Gradient w.r.t weights
        db = (2/m) * np.sum(y_pred - y)  # Gradient w.r.t bias
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.6f}")
    
    return w, b, history


# Example usage
if __name__ == "__main__":
    # Generate sample data: y = 3x + 2 + noise
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2
    y = 3 * X + 2 + np.random.randn(100, 1) * 0.5
    
    # Train
    w, b, history = gradient_descent(X, y, learning_rate=0.1, iterations=500)
    
    print(f"\nLearned: w = {w[0,0]:.4f}, b = {b:.4f}")
    print(f"True:    w = 3.0000, b = 2.0000")
```

### Using Scikit-learn

```python
from sklearn.linear_model import SGDRegressor
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.ravel() + 2 + np.random.randn(100) * 0.5

# Create and train model
model = SGDRegressor(
    learning_rate='constant',
    eta0=0.01,           # Initial learning rate
    max_iter=1000,
    tol=1e-6,
    random_state=42
)
model.fit(X, y)

print(f"Learned: w = {model.coef_[0]:.4f}, b = {model.intercept_[0]:.4f}")
```

---

## Quick Reference

| Concept | Formula / Definition |
|---------|----------------------|
| **Gradient** | $\nabla J$ = vector of partial derivatives |
| **Update Rule** | $w := w - \alpha \nabla_w J$ |
| **Learning Rate** | $\alpha$ = step size (typically 0.001 to 0.1) |
| **Batch GD** | Uses all $m$ examples per update |
| **SGD** | Uses 1 example per update |
| **Mini-batch** | Uses $b$ examples (32-256) per update |
| **Convergence** | When loss stops decreasing |

### Key Takeaways

1. **Gradient descent** finds minimum loss by iteratively moving downhill
2. **Learning rate** is crucial—too high causes divergence, too low is slow
3. **Mini-batch** is the standard choice—balances speed and stability
4. **Adam** is the most popular optimizer for deep learning
5. **Convex problems** (like linear regression) always find the global minimum

---

## Next Steps

After mastering gradient descent, explore:
- [Linear Regression](../../01-supervised-learning/linear-regression/notes.md) - Apply gradient descent to regression
- [Logistic Regression](../../01-supervised-learning/logistic-regression/notes.md) - Apply to classification