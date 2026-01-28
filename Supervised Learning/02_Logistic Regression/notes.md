# Logistic Regression

## Overview
Logistic regression is a **classification algorithm** that predicts the probability of a binary outcome (0 or 1). Despite the name "regression," it's fundamentally used for classification tasks. The key to logistic regression is the **sigmoid function**, which ensures all predictions are bounded between 0 and 1, representing probabilities.

## The Sigmoid Function

The sigmoid (or logistic) function transforms any real-valued input into a probability between 0 and 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Where:**
- $\sigma(x)$ is the output of the sigmoid function
- $e$ is Euler's number (≈ 2.71828)
- $x$ is the input to the sigmoid function

**Key Properties:**
- When $x = 0$, $\sigma(x) = 0.5$
- As $x \to \infty$, $\sigma(x) \to 1$
- As $x \to -\infty$, $\sigma(x) \to 0$
- The function is S-shaped and differentiable everywhere
- Always produces values strictly between 0 and 1 (never exactly 0 or 1)

## The Linear Combination (Log-Odds)

First, we compute a linear combination of features (similar to linear regression):

$$z = b + w_1x_1 + w_2x_2 + \cdots + w_nx_n$$

**Where:**
- $z$ is the linear output, also called the **log-odds**
- $b$ is the bias term
- $w_i$ are the learned weights for each feature
- $x_i$ are the feature values

## Logistic Regression Prediction

The $z$ value is passed through the sigmoid function to obtain a probability:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Where:**
- $\hat{y}$ is the predicted probability (between 0 and 1)
- $z$ is the linear output from above

**Interpretation:**
- If $\hat{y} \geq 0.5$, classify as class 1
- If $\hat{y} < 0.5$, classify as class 0

## Loss Function: Log Loss (Binary Cross-Entropy)

Unlike linear regression which uses Mean Squared Error, logistic regression uses **Log Loss** (also called Binary Cross-Entropy):

$$L = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Where:**
- $m$ is the number of examples
- $y_i$ is the true label (0 or 1)
- $\hat{y}_i$ is the predicted probability
- $\log$ is the natural logarithm

**Intuition:**
- When $y=1$: Loss = $-\log(\hat{y})$ (penalizes low predictions for positive class)
- When $y=0$: Loss = $-\log(1-\hat{y})$ (penalizes high predictions for negative class)

## Regularization

Regularization prevents overfitting by penalizing large weights. Two common approaches:

### L1 Regularization (Lasso)
$$L_{total} = L + \lambda\sum_{i=1}^{n}|w_i|$$

**Effect:** Can zero out some weights (feature selection)

### L2 Regularization (Ridge)
$$L_{total} = L + \lambda\sum_{i=1}^{n}w_i^2$$

**Effect:** Shrinks weights toward zero proportionally

**Where:**
- $\lambda$ (lambda) is the regularization strength (hyperparameter)
- Higher $\lambda$ = stronger regularization = simpler model

## Training: Gradient Descent

The model learns weights by minimizing the loss function using **gradient descent**:

$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

**Where:**
- $\alpha$ (alpha) is the learning rate
- $\frac{\partial L}{\partial w}$ is the gradient (direction of steepest increase)

The gradient for logistic regression is:
$$\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)x_i$$

## Key Differences from Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| **Output** | Continuous values | Probability (0 to 1) |
| **Use Case** | Regression/prediction | Classification |
| **Activation Function** | None (identity) | Sigmoid |
| **Loss Function** | MSE (Mean Squared Error) | Log Loss (Cross-Entropy) |
| **Output Interpretation** | Direct value | Probability of class 1 |

## Decision Boundary

The **decision boundary** is where the model's prediction equals 0.5:

$$\hat{y} = 0.5 \Rightarrow \sigma(z) = 0.5 \Rightarrow z = 0$$

Therefore, the decision boundary is:
$$b + w_1x_1 + w_2x_2 + \cdots + w_nx_n = 0$$

For 2D data, this is a line; for 3D, a plane; for higher dimensions, a hyperplane.

## Evaluation Metrics

Since logistic regression is a classification algorithm, we evaluate it differently than regression:

- **Accuracy:** Percentage of correct predictions
- **Precision:** Of predicted positives, how many are actually positive
- **Recall (Sensitivity):** Of actual positives, how many did we identify
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve
- **Confusion Matrix:** Visual representation of true/false positives and negatives

## Assumptions

1. **Binary outcome:** Target variable is binary (0 or 1)
2. **Independence:** Observations are independent
3. **Linear relationship:** Log-odds have a linear relationship with features
4. **No multicollinearity:** Features are not highly correlated with each other
    