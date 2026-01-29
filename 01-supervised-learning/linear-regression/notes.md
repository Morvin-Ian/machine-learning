# Linear Regression

## Overview

Linear regression is a fundamental supervised learning algorithm that models the **linear relationship** between one or more independent variables (features) and a dependent variable (target output). It's one of the simplest yet most powerful algorithms in machine learning.

### Basic Linear Regression Equation

For a single feature, the linear regression model is represented as:

$$\hat{y} = b + w_1x_1$$

Where:
- $\hat{y}$ = predicted output (dependent variable)
- $b$ = **bias** (y-intercept) — the value of $y$ when all features are zero
- $w_1$ = **weight** (slope/coefficient) — indicates how much the output changes when the feature changes by one unit
- $x_1$ = input feature (independent variable)

### Understanding the Components

**Bias (b):** 
The bias term shifts the line up or down on the y-axis. It's the base value that the model always adds, regardless of input.

**Weight (w₁):** 
The weight determines the **slope** of the line. A positive weight means the line slopes upward (positive relationship), while a negative weight means it slopes downward (negative relationship).

**Example:** 
If we're predicting house prices based on square footage:
$$\text{Price} = 50,000 + 100 \times \text{SquareFootage}$$

Here, $b = 50,000$ (base price) and $w_1 = 100$ (price increase per additional square foot). A 2,000 sq ft house would be predicted as: $50,000 + 100 \times 2,000 = 250,000$

## Multiple Linear Regression

Real-world problems rarely depend on a single feature. Multiple linear regression extends the simple model to include multiple features, each with its own weight.

### Equation for Multiple Features

$$\hat{y} = b + w_1x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n$$

Or in vector notation:

$$\hat{y} = b + \sum_{i=1}^{n} w_ix_i$$

Where:
- $n$ = number of features
- $w_i$ = weight (coefficient) for feature $i$
- $x_i$ = value of feature $i$

### Example: Predicting Salary

Suppose we're predicting employee salary based on three features:

$$\text{Salary} = 20,000 + 1,500 \times \text{Experience} + 5,000 \times \text{Education\_Level} + 0.5 \times \text{Performance\_Score}$$

Where:
- $b = 20,000$ (base salary)
- $w_1 = 1,500$ (salary increase per year of experience)
- $w_2 = 5,000$ (salary increase per education level)
- $w_3 = 0.5$ (salary increase per performance point)

For an employee with 5 years experience, education level 4, and performance score 85:
$$\text{Salary} = 20,000 + (1,500 \times 5) + (5,000 \times 4) + (0.5 \times 85)$$
$$= 20,000 + 7,500 + 20,000 + 42.5 = 47,542.5$$

## Loss/Cost Functions

Loss (also called **error** or **cost**) is a numerical measure that quantifies how wrong the model's predictions are. It measures the distance between predicted values and actual values. **Lower loss indicates better model performance.**

### Types of Loss Functions

#### 1. **L¹ Loss (Absolute Error)**
$$L_1 = |y - \hat{y}|$$

The absolute difference between predicted and actual values.

**Example:** 
- Actual value: 24, Predicted: 23.5
- L₁ Loss = |24 - 23.5| = 0.5

#### 2. **Mean Absolute Error (MAE)**
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

Average of all L₁ losses across $m$ examples. **More interpretable** because it's in the same units as the target variable.

**Example with 3 predictions:**
- Actual: [10, 20, 30], Predicted: [11, 19, 32]
- Errors: [1, 1, 2]
- MAE = (1 + 1 + 2) / 3 = 1.33

#### 3. **L² Loss (Squared Error)**
$$L_2 = (y - \hat{y})^2$$

Squares the difference, which **penalizes larger errors more heavily**.

**Example:**
- Actual value: 24, Predicted: 23.1
- L₂ Loss = (24 - 23.1)² = (0.9)² = 0.81

#### 4. **Mean Squared Error (MSE)**
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Average of all L₂ losses. **Widely used in regression problems.**

**Example with 3 predictions:**
- Actual: [10, 20, 30], Predicted: [11, 19, 32]
- Errors: [1, 1, 2]
- MSE = (1² + 1² + 2²) / 3 = 6/3 = 2

#### 5. **Root Mean Squared Error (RMSE)**
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}$$

Square root of MSE. **Returns error in the original scale of the target variable**, making it more interpretable than MSE.

**Example:**
- MSE = 2
- RMSE = √2 ≈ 1.41

### When to Use Which Loss Function

| Loss | Characteristics | Use Case |
|------|---|---|
| **MAE** | Robust to outliers, linear penalty | When outliers shouldn't heavily influence training |
| **MSE/RMSE** | Sensitive to outliers, quadratic penalty | When large errors need to be penalized heavily |

**Key Insight:** Squaring amplifies larger errors. A difference of 10 becomes 100 in L₂, making the model more sensitive to large mistakes.

## Gradient Descent

**Gradient Descent** is the optimization algorithm used to find the optimal weights ($w$) and bias ($b$) that minimize the loss function. Instead of searching randomly, it **iteratively adjusts parameters in the direction that reduces loss**.

### How Gradient Descent Works

The algorithm updates parameters using the gradient (slope) of the loss function:

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where:
- $\alpha$ = **learning rate** (step size)
- $\frac{\partial J}{\partial w}$ = partial derivative of loss with respect to weight (direction of steepest ascent)
- The negative sign ensures we move **downhill** toward lower loss

### Intuitive Explanation

Imagine you're on a hillside in fog trying to reach the valley (lowest point):
1. You check the slope beneath your feet (compute gradient)
2. You take a step in the direction that goes downhill (negative gradient)
3. You repeat until you reach the valley (convergence)

### Variants of Gradient Descent

#### 1. **Batch Gradient Descent (BGD)**
Uses **all** training examples to compute the gradient before updating weights.

$$w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} \frac{\partial J^{(i)}}{\partial w}$$

**Advantages:**
- Stable, smooth updates
- Accurate gradient estimation

**Disadvantages:**
- Slow for large datasets
- Can be memory-intensive

**Example:** With 1,000 samples, you compute loss for all 1,000 before updating once.

#### 2. **Stochastic Gradient Descent (SGD)**
Updates weights using **one single example** at a time.

$$w := w - \alpha \frac{\partial J^{(i)}}{\partial w}$$

for one random example $i$

**Advantages:**
- Fast updates
- Can escape local minima due to noise
- Works well for streaming data

**Disadvantages:**
- Noisy, erratic loss curve
- May never converge to true minimum
- Takes longer to reach convergence

**Example:** With 1,000 samples, you update weights 1,000 times per epoch, once per sample.

#### 3. **Mini-batch Gradient Descent**
Compromise between BGD and SGD—updates using a **small batch** of examples.

$$w := w - \alpha \frac{1}{b} \sum_{i=1}^{b} \frac{\partial J^{(i)}}{\partial w}$$

where $b$ = batch size

**Advantages:**
- More stable than SGD but faster than BGD
- Better hardware utilization
- Most commonly used in practice

**Disadvantages:**
- Requires tuning batch size
- Still noisier than full batch

**Example:** With 1,000 samples and batch size 32, you have ~31 updates per epoch.

### Gradient Descent with Example

Given: $\hat{y} = 2 + 3x$, learning rate $\alpha = 0.01$

**Initial state:** $w = 0, b = 0$

**Data point:** $x = 2, y_{actual} = 8$

**Step 1:**
- Prediction: $\hat{y} = 0 + 0 \times 2 = 0$
- Error: $8 - 0 = 8$
- Gradient: $\frac{\partial J}{\partial w} = -2x \times (y - \hat{y}) = -2 \times 2 \times 8 = -32$
- Update: $w := 0 - 0.01 \times (-32) = 0.32$

**Step 2:**
- Prediction: $\hat{y} = 0 + 0.32 \times 2 = 0.64$
- Error: $8 - 0.64 = 7.36$
- ... (process repeats)

## Model Convergence and Loss Curves

When training a model, the **loss curve** visualizes how loss changes over training iterations or epochs. It's critical for understanding if training is progressing correctly.

### What is Convergence?

**Convergence** occurs when the model's loss stabilizes and stops decreasing significantly. At this point, the model has likely found optimal (or near-optimal) weights and bias.

### Typical Loss Curve Pattern

```
Loss
  |     ╱╲╱╲
  |    ╱  ╲  ╲
  |   ╱    ╲  ╲____  ← Convergence (stable)
  |  ╱      ╲      ╲
  |_╱________╲______╲_____ Iterations
  0          50      100
```

**Good convergence:** Loss decreases smoothly and stabilizes.

### Convex Functions and Linear Regression

The loss function for linear regression models is **convex**, meaning it has a single global minimum—no local minima to get stuck in.

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

This is **bowl-shaped** in parameter space, guaranteeing that gradient descent will eventually find the global minimum.

**Implication:** Unlike complex neural networks, linear regression always finds the optimal solution (if it converges).

### Common Convergence Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Loss oscillates wildly** | Learning rate too high | Decrease learning rate |
| **Loss decreases very slowly** | Learning rate too low | Increase learning rate |
| **Loss increases** | Learning rate too high (bouncing over minimum) | Decrease learning rate significantly |
| **Loss stagnates (stops improving)** | May have converged; or stuck in plateau | Check if loss is truly stable or try different learning rate |

## Hyperparameters

**Hyperparameters** are configuration settings that we **set before training**. They control how the learning process works and significantly impact model performance. Unlike weights and biases (which are learned), hyperparameters are manually tuned.

### 1. Learning Rate (α)

The **learning rate** controls the size of steps taken during gradient descent. It directly affects convergence speed and stability.

$$w := w - \alpha \frac{\partial J}{\partial w}$$

The parameter $\alpha$ scales how much we move in the gradient direction.

#### Effect of Learning Rate

| Learning Rate | Effect | Loss Curve |
|---|---|---|
| **Too Low** ($\alpha = 0.0001$) | Training very slow, takes many iterations to converge | Decreases slowly, smooth |
| **Good** ($\alpha = 0.01$) | Balanced: converges reasonably fast and smoothly | Decreases smoothly to minimum |
| **Too High** ($\alpha = 0.1$) | Bounces around the minimum, may diverge | Oscillates or increases |
| **Way Too High** ($\alpha = 1$) | Diverges completely, loss increases | Explodes rapidly upward |

#### Example: Finding Optimal Weight

Suppose true weight is $w^* = 5$.

**With $\alpha = 0.1$ (too high):**
- Iteration 0: $w = 0$, gradient = -10
- Iteration 1: $w = 0 - 0.1 \times (-10) = 1$
- Iteration 2: $w = 1 - 0.1 \times (-8) = 1.8$
- ... but overshoots and bounces

**With $\alpha = 0.01$ (good):**
- Iteration 0: $w = 0$, gradient = -10
- Iteration 1: $w = 0 - 0.01 \times (-10) = 0.1$
- Iteration 2: $w = 0.1 - 0.01 \times (-9.8) = 0.198$
- ... smoothly approaches $w^* = 5$

### 2. Batch Size

**Batch size** determines how many examples are used to compute one gradient update.

$$\text{Number of updates per epoch} = \frac{\text{Total examples}}{\text{Batch size}}$$

#### Batch Size Strategies

| Strategy | Batch Size | Updates/Epoch | Noise | Stability |
|---|---|---|---|---|
| **Full Batch** | = Total examples | 1 | Low | Very stable |
| **Mini-batch** | 32-256 | Many | Medium | Balanced |
| **Stochastic** | 1 | Very many | Very high | Unstable but escapes local minima |

#### Stochastic Gradient Descent (SGD)

Uses batch size of **1** — updates weights after each example.

**Advantages:**
- Fast iterations
- Can escape local minima due to noise
- Works with streaming data

**Disadvantages:**
- Very noisy gradient estimates
- Erratic loss curve
- May never perfectly converge

**Example:** Dataset with 1,000 samples, 20 epochs
- Updates per epoch: 1,000
- Total updates: 20,000

#### Mini-batch SGD

Uses batch size between 1 and total examples—the **best compromise**.

**Example:** Dataset with 1,000 samples, batch size = 32, 20 epochs
- Updates per epoch: 1,000 / 32 = 31.25 ≈ 31
- Total updates: 620

**Advantages:**
- Faster training than full batch
- More stable than single-sample SGD
- Better hardware utilization (parallelization)

#### Full Batch Gradient Descent

Uses all data for one update—the most stable but slowest.

**Example:** Dataset with 1,000 samples, 20 epochs
- Updates per epoch: 1
- Total updates: 20

**Advantages:**
- Most accurate gradient estimate
- Smoothest training curve

**Disadvantages:**
- Very slow for large datasets
- High memory usage
- May get stuck in local minima (though linear regression has none)


### 3. Epochs

An **epoch** is one complete pass through the entire training dataset. After each epoch, the model has seen and learned from every training example once.

$$\text{Iterations per epoch} = \frac{\text{Total training examples}}{\text{Batch size}}$$

#### Number of Epochs

- **Too few epochs:** Model underfits (doesn't learn enough)
- **Too many epochs:** Model overfits (memorizes training data)
- **Optimal:** Model generalizes well to unseen data

**Example:** 
- Dataset: 1,000 samples
- Batch size: 100

**One epoch requires:**
- 1,000 / 100 = 10 iterations

If we train for 100 epochs:
- Total iterations: 10 × 100 = 1,000

### 4. Relationship Between Batch Size and Epochs

The total number of **updates** depends on both batch size and epochs:

$$\text{Total Updates} = \text{Epochs} \times \left\lceil \frac{\text{Dataset Size}}{\text{Batch Size}} \right\rceil$$

#### Comparison of Update Frequencies

Given a dataset of **1,000 examples**, training for **20 epochs**:

| Method | Batch Size | Updates/Epoch | Total Updates |
|--------|-----------|---|---|
| **Full Batch** | 1,000 | 1 | 20 |
| **Mini-batch** | 100 | 10 | 200 |
| **Mini-batch** | 32 | 31 | 620 |
| **SGD** | 1 | 1,000 | 20,000 |

#### Visual Comparison

```
Training Progress Over 20 Epochs
Dataset: 1,000 samples

Full Batch (20 updates total):
Epoch: |====|====|====|====|====|
       ↑    ↑    ↑    ↑    ↑    (Only 1 update per epoch)

Mini-batch size=100 (200 updates):
Epoch: |=|=|=|=|=|=|=|=|=|=|=|
       10 updates per epoch

Mini-batch size=32 (620 updates):
Epoch: |=|=|=|=|=|=|=|=|=|=|=|=|=|=|=|
       31 updates per epoch

SGD (20,000 updates):
Epoch: ||||||||||||||||||||||||||||
       1,000 updates per epoch
```

### Hyperparameter Tuning Tips

1. **Learning Rate:** Start with 0.01, adjust up if too slow or down if oscillating
2. **Batch Size:** Common values are 32, 64, 128, 256
3. **Epochs:** Use early stopping—monitor validation loss and stop when it increases
4. **Trade-offs:**
   - Larger batch size → faster training but less frequent updates
   - Smaller batch size → slower but more frequent updates with more noise
   - Higher learning rate → faster convergence but risk of overshooting
   - Lower learning rate → slower but more stable convergence 