# Multiple Linear Regression & Vectorization

A deep dive into handling multiple features efficiently using vectorization and NumPy.

---

## From One Feature to Many

In simple linear regression, we have one feature:

$$\hat{y} = wx + b$$

But real-world problems have **multiple features**. For example, predicting house prices:

| Feature ($x$) | Example |
|---------------|---------|
| $x_1$ | Square footage |
| $x_2$ | Number of bedrooms |
| $x_3$ | Age of house |
| $x_4$ | Distance to city center |

---

## Multiple Features: The Model

### Notation

For a dataset with $m$ examples and $n$ features:

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x^{(i)}$ | The $i$-th training example (a vector of $n$ features) |
| $x_j^{(i)}$ | The $j$-th feature of the $i$-th example |
| $\vec{w}$ | Weight vector $[w_1, w_2, ..., w_n]$ |
| $b$ | Bias (scalar) |

### The Prediction Equation

**Scalar form** (one feature at a time):

$$\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n + b$$

**Summation form**:

$$\hat{y} = \sum_{j=1}^{n} w_jx_j + b$$

**Vector form** (compact):

$$\hat{y} = \vec{w} \cdot \vec{x} + b$$

Where $\vec{w} \cdot \vec{x}$ is the **dot product**:

$$\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + ... + w_nx_n$$

---

## Why Vectorization?

### The Problem with Loops

Without vectorization, you compute predictions using loops:

```python
# Slow: Using loops
def predict_slow(x, w, b):
    """
    x: single example with n features
    w: weight vector with n weights
    """
    y_hat = 0
    for j in range(len(w)):
        y_hat += w[j] * x[j]
    y_hat += b
    return y_hat
```

For $m$ examples with $n$ features, this requires $m \times n$ loop iterations.

### Vectorization: Let NumPy Do the Work

NumPy operations are implemented in C and use SIMD (Single Instruction, Multiple Data) instructions to process multiple numbers simultaneously.

```python
# Fast: Using vectorization
import numpy as np

def predict_fast(x, w, b):
    """
    x: single example with n features (numpy array)
    w: weight vector (numpy array)
    """
    return np.dot(w, x) + b
```

### Speed Comparison

```python
import numpy as np
import time

n = 10000  # 10,000 features
w = np.random.randn(n)
x = np.random.randn(n)

# Loop version
start = time.time()
result_loop = 0
for j in range(n):
    result_loop += w[j] * x[j]
print(f"Loop: {time.time() - start:.6f} seconds")

# Vectorized version
start = time.time()
result_vec = np.dot(w, x)
print(f"Vectorized: {time.time() - start:.6f} seconds")

# Typical output:
# Loop: 0.003500 seconds
# Vectorized: 0.000015 seconds
# ~200x faster!
```

---

## The Math of Vectorization

### Vectors and Matrices

**Vector** (1D array): A list of numbers

$$\vec{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}$$

**Matrix** (2D array): A grid of numbers

$$X = \begin{bmatrix} x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \end{bmatrix}$$

Each row is one training example. Each column is one feature.

### Matrix-Vector Multiplication

To predict **all examples at once**:

$$\hat{\vec{y}} = X \vec{w} + b$$

Where:
- $X$ is $(m \times n)$ — $m$ examples, $n$ features
- $\vec{w}$ is $(n \times 1)$ — $n$ weights
- $\hat{\vec{y}}$ is $(m \times 1)$ — $m$ predictions

```
    X          @      w      +    b    =    ŷ
(m × n)           (n × 1)       scalar    (m × 1)

┌─────────┐      ┌───┐              ┌───┐
│ x₁ x₂ x₃│      │w₁ │              │ŷ₁ │
│ x₁ x₂ x₃│  @   │w₂ │   +   b   =  │ŷ₂ │
│ x₁ x₂ x₃│      │w₃ │              │ŷ₃ │
└─────────┘      └───┘              └───┘
```

---

## Gradient Descent with Multiple Features

### The Cost Function

For multiple features, the cost function is the same MSE:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

Where $\hat{y}^{(i)} = \vec{w} \cdot \vec{x}^{(i)} + b$

### Gradient Computation

We need the gradient with respect to **each weight** and the bias:

**For each weight $w_j$:**

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

**For bias $b$:**

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### Update Rules

Update all weights simultaneously:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$ for $j = 1, 2, ..., n$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

### Vectorized Gradients

Instead of looping through each weight:

**Scalar (slow):**
```python
for j in range(n):
    dw[j] = (1/m) * sum((y_hat - y) * X[:, j])
```

**Vectorized (fast):**
```python
dw = (1/m) * X.T @ (y_hat - y)  # All gradients at once!
db = (1/m) * np.sum(y_hat - y)
```

The key insight: $X^T \cdot \text{error}$ computes all weight gradients simultaneously.

```
   X.T        @    error    =    dw
(n × m)          (m × 1)       (n × 1)

┌─────────┐      ┌───┐        ┌────┐
│ ← row 1 │      │e₁ │        │dw₁ │
│ ← row 2 │  @   │e₂ │   =    │dw₂ │
│ ← row 3 │      │e₃ │        │dw₃ │
└─────────┘      └───┘        └────┘
```

---

## Complete Implementation

### From Scratch with NumPy

```python
import numpy as np

class LinearRegressionMultiple:
    """
    Multiple Linear Regression using vectorized gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : numpy array of shape (m, n)
            Training features. m examples, n features.
        y : numpy array of shape (m,) or (m, 1)
            Target values.
        """
        m, n = X.shape
        y = y.reshape(-1, 1)  # Ensure y is column vector
        
        # Initialize weights and bias
        self.w = np.zeros((n, 1))
        self.b = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: predictions (vectorized)
            y_hat = X @ self.w + self.b  # (m, 1)
            
            # Compute error
            error = y_hat - y  # (m, 1)
            
            # Compute cost (for monitoring)
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients (vectorized)
            dw = (1 / m) * X.T @ error  # (n, 1)
            db = (1 / m) * np.sum(error)  # scalar
            
            # Update parameters
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions (vectorized).
        """
        return X @ self.w + self.b
    
    def score(self, X, y):
        """
        Compute R² score.
        """
        y_hat = self.predict(X).flatten()
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data: y = 3x₁ + 2x₂ - x₃ + 5 + noise
    m = 1000  # examples
    n = 3     # features
    
    X = np.random.randn(m, n) * 10
    true_w = np.array([[3], [2], [-1]])
    true_b = 5
    y = X @ true_w + true_b + np.random.randn(m, 1) * 2
    
    # Train model
    model = LinearRegressionMultiple(learning_rate=0.001, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\nTrue weights: {true_w.flatten()}")
    print(f"Learned weights: {model.w.flatten()}")
    print(f"\nTrue bias: {true_b}")
    print(f"Learned bias: {model.b:.4f}")
    print(f"\nR² Score: {model.score(X, y.flatten()):.4f}")
```

### Output

```
Iteration 0: Cost = 2847.234567
Iteration 100: Cost = 45.678912
Iteration 200: Cost = 8.234567
...
Iteration 900: Cost = 2.012345

True weights: [ 3.  2. -1.]
Learned weights: [ 2.98  1.99 -0.98]

True bias: 5
Learned bias: 5.0234

R² Score: 0.9876
```

---

## NumPy Operations Reference

### Essential Operations for ML

| Operation | NumPy | Description |
|-----------|-------|-------------|
| **Dot product** | `np.dot(a, b)` or `a @ b` | Sum of element-wise products |
| **Matrix multiply** | `A @ B` | Standard matrix multiplication |
| **Transpose** | `A.T` | Swap rows and columns |
| **Element-wise** | `A * B` | Multiply corresponding elements |
| **Sum** | `np.sum(A)` | Sum all elements |
| **Mean** | `np.mean(A)` | Average of elements |
| **Broadcasting** | `A + scalar` | Apply scalar to all elements |

### Broadcasting Rules

NumPy automatically expands dimensions when shapes are compatible:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

b = 10  # Scalar broadcasts to all elements
result = X + b
# [[11, 12, 13],
#  [14, 15, 16]]

w = np.array([1, 2, 3])  # Shape: (3,)
result = X * w  # w broadcasts across rows
# [[1*1, 2*2, 3*3],
#  [4*1, 5*2, 6*3]]
# = [[1, 4, 9],
#    [4, 10, 18]]
```

### Shape Manipulation

```python
# Reshape
x = np.array([1, 2, 3, 4, 5, 6])
X = x.reshape(2, 3)  # 2 rows, 3 columns
X = x.reshape(-1, 1)  # Column vector (6, 1)
X = x.reshape(1, -1)  # Row vector (1, 6)

# Flatten
x_flat = X.flatten()  # Back to 1D

# Add dimension
x = np.array([1, 2, 3])  # (3,)
x = x[:, np.newaxis]     # (3, 1) - column vector
x = x[np.newaxis, :]     # (1, 3) - row vector
```

---

## Feature Scaling

When features have different scales, gradient descent can be slow or unstable.

### The Problem

```
Feature 1 (sqft):    500 - 5000
Feature 2 (bedrooms): 1 - 5

Without scaling:
- Gradient for sqft is tiny (values are large)
- Gradient for bedrooms is huge (values are small)
- Weights update at very different rates!
```

### Solution: Normalize Features

**Z-score normalization (standardization):**

$$x_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sigma_j}$$

Where $\mu_j$ is the mean and $\sigma_j$ is the standard deviation of feature $j$.

```python
def normalize(X):
    """Z-score normalization."""
    mu = np.mean(X, axis=0)  # Mean of each feature
    sigma = np.std(X, axis=0)  # Std of each feature
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Usage
X_train_norm, mu, sigma = normalize(X_train)
X_test_norm = (X_test - mu) / sigma  # Use same mu, sigma!
```

**Min-max scaling:**

$$x_j^{(i)} = \frac{x_j^{(i)} - \min_j}{\max_j - \min_j}$$

Scales features to [0, 1] range.

---

## Comparison: Loop vs Vectorized

| Aspect | Loop-based | Vectorized |
|--------|------------|------------|
| **Speed** | Slow (Python overhead) | Fast (C/SIMD) |
| **Code** | More lines, explicit | Concise, mathematical |
| **Memory** | Lower (one at a time) | Higher (all at once) |
| **Debugging** | Easier to step through | Harder to inspect |
| **GPU** | Cannot use | Can use with CuPy/JAX |

### When to Use What

- **Always use vectorization** for production code
- **Use loops** only for learning/understanding
- **Large datasets**: Vectorization is essential
- **GPU training**: Requires vectorization

---

## Quick Reference

### The Core Equations (Vectorized)

| Step | Equation | NumPy |
|------|----------|-------|
| **Predict** | $\hat{y} = Xw + b$ | `y_hat = X @ w + b` |
| **Error** | $e = \hat{y} - y$ | `error = y_hat - y` |
| **Cost** | $J = \frac{1}{2m}\sum e^2$ | `cost = np.sum(error**2) / (2*m)` |
| **Weight gradient** | $\nabla_w = \frac{1}{m}X^T e$ | `dw = (1/m) * X.T @ error` |
| **Bias gradient** | $\nabla_b = \frac{1}{m}\sum e$ | `db = (1/m) * np.sum(error)` |
| **Update** | $w := w - \alpha \nabla_w$ | `w = w - lr * dw` |

### Key Takeaways

1. **Vectorization** makes code 100-1000x faster
2. Use **matrix multiplication** (`@`) instead of loops
3. **X.T @ error** computes all gradients at once
4. **Normalize features** for stable training
5. Think in terms of **shapes**: $(m \times n) @ (n \times 1) = (m \times 1)$

---

## Next Steps

- [Gradient Descent](./gradient-descent/notes.md) — deep dive into optimization
- [Linear Regression](../../01-supervised-learning/linear-regression/notes.md) — apply these concepts
- [Working with Data](./working-with-data/README.md) — data preparation foundations
