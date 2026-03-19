# Generalization, Overfitting, and Underfitting

The ultimate goal of machine learning is not to perform well on training data, but to perform well on **new, unseen data**. This document explores the critical concepts of generalization, overfitting, and underfitting.

---

## Table of Contents

1. [Generalization](#generalization)
2. [Overfitting](#overfitting)
3. [Underfitting](#underfitting)
4. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
5. [Learning Curves](#learning-curves)

---

## Generalization

**Generalization** is the ability of a machine learning model to perform well on new, unseen data that wasn't part of the training set.

### What is Generalization?

A model that generalizes well has learned the **underlying patterns** in the data, not just memorized the training examples.

**Good generalization:**
```
Training accuracy: 92%
Test accuracy: 90%        ← Similar performance on new data ✓
Gap: 2%                   ← Small gap indicates good generalization
```

**Poor generalization:**
```
Training accuracy: 99%
Test accuracy: 65%        ← Much worse on new data ✗
Gap: 34%                  ← Large gap indicates poor generalization
```

### Intuitive Example: Learning to Recognize Dogs

**Scenario:** You're teaching a child to recognize dogs using photos.

**Good generalization (learned the concept):**
- Child sees: 10 photos of different dogs
- Child learns: "Dogs have four legs, fur, tail, wet nose, bark"
- Test: Show a new dog breed → Child correctly identifies it as a dog ✓
- **Why it works:** Learned general features of dogs

**Poor generalization (memorized examples):**
- Child sees: 10 photos of specific dogs
- Child memorizes: "This exact brown dog is a dog, this exact white dog is a dog..."
- Test: Show a new dog → Child says "I haven't seen this one, so it's not a dog" ✗
- **Why it fails:** Memorized specific examples, not the concept

### Mathematical Perspective

The goal is to minimize the **expected error** on the true data distribution, not just the training set.

**Training error:**
$$E_{train} = \frac{1}{m} \sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)})$$

Where:
- $m$ = number of training examples
- $L$ = loss function
- $f(x^{(i)})$ = model prediction
- $y^{(i)}$ = true label

**Generalization error (what we really care about):**
$$E_{gen} = \mathbb{E}_{(x,y) \sim P}[L(f(x), y)]$$

Where $P$ is the true data distribution.

**The problem:** We don't know $P$! We only have a finite training set sampled from $P$.

**The goal:** Minimize $E_{gen}$, but we can only measure $E_{train}$.

**Generalization gap:**
$$\text{Gap} = E_{gen} - E_{train}$$

A small gap means the model generalizes well.

### Factors Affecting Generalization

#### 1. Model Complexity

**Too simple:**
- Can't capture patterns
- High training error
- High test error
- **Underfitting**

**Just right:**
- Captures true patterns
- Low training error
- Low test error
- **Good generalization**

**Too complex:**
- Captures noise as patterns
- Very low training error
- High test error
- **Overfitting**

```
Error
  ↑
  │     Test Error
  │        ∿∿∿
  │      ∿     ∿
  │    ∿         ∿
  │   ∿           ∿∿∿∿∿∿∿
  │  ∿       ↑
  │ ∿    Sweet spot
  │∿
  │ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
  │     Training Error
  └────────────────────→ Model Complexity
  Simple              Complex
```

#### 2. Training Data Size

**Small dataset:**
- Model might memorize all examples
- Poor generalization
- High variance

**Large dataset:**
- Model must learn general patterns
- Better generalization
- Lower variance

**Rule of thumb:** You need at least 10 times as many training examples as you have parameters.

```python
# Example: Linear regression with 10 features
n_features = 10
n_parameters = n_features + 1  # weights + bias = 11

# Minimum recommended training samples
min_samples = 10 * n_parameters  # 110 samples

# Better: 100x more
recommended_samples = 100 * n_parameters  # 1100 samples
```

#### 3. Data Quality

**Clean, representative data:**
- Model learns true patterns
- Good generalization

**Noisy, biased data:**
- Model learns noise and biases
- Poor generalization

**Example:**

```python
# Clean data: True relationship is y = 2x + 1
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])  # Perfect relationship

# Noisy data: Same relationship + random noise
y_noisy = np.array([3.2, 4.8, 7.3, 8.9, 11.1])

# Model trained on clean data: Learns y = 2x + 1 (perfect)
# Model trained on noisy data: Learns y = 1.98x + 1.04 (close, but not perfect)
```

#### 4. Feature Selection

**Relevant features:**
- Help model learn true patterns
- Improve generalization

**Irrelevant features:**
- Add noise
- Hurt generalization
- Increase overfitting risk

**Example:**

```python
# Predicting house prices

# Relevant features:
# - Square footage ✓
# - Number of bedrooms ✓
# - Location ✓
# - Age of house ✓

# Irrelevant features:
# - Owner's favorite color ✗
# - Day of the week listed ✗
# - Listing agent's birthday ✗

# Model with only relevant features: Better generalization
# Model with irrelevant features: Worse generalization (might learn spurious correlations)
```

---

## Overfitting

**Overfitting** occurs when a model learns the training data **too well**, including noise and random fluctuations, failing to generalize to new data.

### Visual Understanding

```
Underfitting:              Good Fit:              Overfitting:
    •                         •                       •
  •   •                     •   •                   •   •
 •     •                   •     •                 •     •
•       •                 •       •               •       •
  •   •                     •   •                   •   •
    •                         •                       •
    
Simple line              Smooth curve           Wiggly curve
Misses pattern          Captures pattern       Memorizes noise
High bias               Balanced               High variance
```

**Underfitting:** Model is too simple
- Straight line can't capture curved relationship
- High training error, high test error

**Good fit:** Model captures true pattern
- Smooth curve follows general trend
- Low training error, low test error

**Overfitting:** Model is too complex
- Wiggly curve passes through every point
- Very low training error, high test error

### Real Example: Polynomial Regression

Let's predict house prices based on square footage.

**True relationship:** $\text{price} = 100 \times \text{sqft} + 50000 + \text{noise}$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
sqft = np.linspace(1000, 3000, 20)
price = 100 * sqft + 50000 + np.random.normal(0, 10000, 20)

X_train = sqft[:15].reshape(-1, 1)
y_train = price[:15]
X_test = sqft[15:].reshape(-1, 1)
y_test = price[15:]

# Degree 1: Linear (may underfit)
poly_1 = PolynomialFeatures(degree=1)
X_train_poly_1 = poly_1.fit_transform(X_train)
X_test_poly_1 = poly_1.transform(X_test)

model_1 = LinearRegression()
model_1.fit(X_train_poly_1, y_train)

train_error_1 = mean_squared_error(y_train, model_1.predict(X_train_poly_1))
test_error_1 = mean_squared_error(y_test, model_1.predict(X_test_poly_1))

print(f"Degree 1 (Linear):")
print(f"  Train MSE: {train_error_1:,.0f}")
print(f"  Test MSE: {test_error_1:,.0f}")
print(f"  Gap: {test_error_1 - train_error_1:,.0f}")

# Degree 2: Quadratic (good fit)
poly_2 = PolynomialFeatures(degree=2)
X_train_poly_2 = poly_2.fit_transform(X_train)
X_test_poly_2 = poly_2.transform(X_test)

model_2 = LinearRegression()
model_2.fit(X_train_poly_2, y_train)

train_error_2 = mean_squared_error(y_train, model_2.predict(X_train_poly_2))
test_error_2 = mean_squared_error(y_test, model_2.predict(X_test_poly_2))

print(f"\nDegree 2 (Quadratic):")
print(f"  Train MSE: {train_error_2:,.0f}")
print(f"  Test MSE: {test_error_2:,.0f}")
print(f"  Gap: {test_error_2 - train_error_2:,.0f}")

# Degree 10: High polynomial (overfits!)
poly_10 = PolynomialFeatures(degree=10)
X_train_poly_10 = poly_10.fit_transform(X_train)
X_test_poly_10 = poly_10.transform(X_test)

model_10 = LinearRegression()
model_10.fit(X_train_poly_10, y_train)

train_error_10 = mean_squared_error(y_train, model_10.predict(X_train_poly_10))
test_error_10 = mean_squared_error(y_test, model_10.predict(X_test_poly_10))

print(f"\nDegree 10 (High Polynomial):")
print(f"  Train MSE: {train_error_10:,.0f}")
print(f"  Test MSE: {test_error_10:,.0f}")
print(f"  Gap: {test_error_10 - train_error_10:,.0f}")

# Typical output:
# Degree 1 (Linear):
#   Train MSE: 95,234,567
#   Test MSE: 98,123,456
#   Gap: 2,888,889
#
# Degree 2 (Quadratic):
#   Train MSE: 87,654,321
#   Test MSE: 89,234,567
#   Gap: 1,580,246
#
# Degree 10 (High Polynomial):
#   Train MSE: 123  ← Nearly perfect on training!
#   Test MSE: 456,789,012  ← Terrible on test!
#   Gap: 456,788,889  ← Huge gap = overfitting!
```

**Analysis:**

| Model | Train Error | Test Error | Gap | Diagnosis |
|-------|-------------|------------|-----|-----------|
| Degree 1 | High | High | Small | Underfitting |
| Degree 2 | Medium | Medium | Small | Good fit ✓ |
| Degree 10 | Very low | Very high | Huge | Overfitting |

### Signs of Overfitting

**1. Large gap between training and test performance**

```python
# Example metrics
train_accuracy = 0.99  # 99%
test_accuracy = 0.65   # 65%
gap = train_accuracy - test_accuracy  # 0.34 (34%)

if gap > 0.1:  # Gap > 10%
    print("Warning: Possible overfitting!")
```

**2. Perfect or near-perfect training performance**

```python
if train_accuracy > 0.99:  # 99%+ accuracy
    print("Warning: Model might be memorizing training data!")
```

**3. Model is very complex relative to data size**

```python
n_samples = len(X_train)
n_parameters = model.count_parameters()

if n_parameters > n_samples:
    print("Warning: More parameters than samples!")
    print("High risk of overfitting!")
```

**4. Performance degrades with more training**

```
Epoch 1: Train loss = 0.5, Val loss = 0.6
Epoch 2: Train loss = 0.3, Val loss = 0.5
Epoch 3: Train loss = 0.1, Val loss = 0.4
Epoch 4: Train loss = 0.05, Val loss = 0.5  ← Val loss increased!
Epoch 5: Train loss = 0.01, Val loss = 0.6  ← Getting worse!

Training loss keeps decreasing, but validation loss starts increasing.
This is a clear sign of overfitting.
```

### Causes of Overfitting

#### 1. Model Too Complex

**Problem:** Too many parameters relative to the amount of data.

```python
# Example: Predicting house prices with 100 samples

# Simple model: 3 features
# Parameters: 3 weights + 1 bias = 4 parameters
# Ratio: 100 samples / 4 parameters = 25 samples per parameter ✓

# Complex model: 50 features
# Parameters: 50 weights + 1 bias = 51 parameters
# Ratio: 100 samples / 51 parameters = 2 samples per parameter ✗
# Not enough data to reliably estimate 51 parameters!
```

**Solution:** Simplify the model or get more data.

#### 2. Too Little Training Data

**Problem:** Not enough examples to learn general patterns.

```python
# Extreme example: 5 training samples
X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

# A degree-4 polynomial can fit these 5 points perfectly!
# But it won't generalize to new data.
```

**Solution:** Collect more data or use data augmentation.

#### 3. Training Too Long

**Problem:** Model starts memorizing training data after learning general patterns.

```
Early epochs: Learning general patterns ✓
Later epochs: Memorizing specific examples ✗
```

**Solution:** Use early stopping (stop when validation performance stops improving).

#### 4. Noisy Data

**Problem:** Model learns noise as if it were signal.

```python
# True relationship: y = 2x + 1
# But data has noise:
X = [1, 2, 3, 4, 5]
y = [3.2, 4.8, 7.3, 8.9, 11.1]  # True values + noise

# Complex model might learn:
# "When x=1, y=3.2 exactly"
# "When x=2, y=4.8 exactly"
# This is learning noise, not the true pattern!
```

**Solution:** Clean data, use regularization, or use simpler models.

#### 5. Too Many Irrelevant Features

**Problem:** Model finds spurious correlations in irrelevant features.

```python
# Predicting house prices
features = [
    'square_feet',      # Relevant ✓
    'bedrooms',         # Relevant ✓
    'owner_age',        # Irrelevant ✗
    'listing_day',      # Irrelevant ✗
    'agent_birthday'    # Irrelevant ✗
]

# Model might learn: "Houses listed on Tuesdays are cheaper"
# This is a spurious correlation in the training data!
```

**Solution:** Feature selection, domain knowledge, or regularization.

---

### Preventing Overfitting

#### 1. Get More Training Data

**Why it works:** More data forces the model to learn general patterns instead of memorizing.

```python
# Small dataset: 100 samples
# Model can memorize all 100 examples

# Large dataset: 10,000 samples
# Model can't memorize 10,000 examples
# Must learn general patterns
```

**Practical approaches:**

```python
# A. Collect more data (best, but often expensive)
# B. Data augmentation (create variations of existing data)

# Example: Image data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,        # Rotate images up to 20 degrees
    width_shift_range=0.2,    # Shift horizontally
    height_shift_range=0.2,   # Shift vertically
    horizontal_flip=True,     # Flip images
    zoom_range=0.2            # Zoom in/out
)

# One image → Many variations
# Effectively increases dataset size
```

#### 2. Regularization

**Regularization** adds a penalty for model complexity, discouraging the model from fitting noise.

**L2 Regularization (Ridge):**

Adds penalty proportional to the square of weights:

$$J(\vec{w}) = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2$$

Where:
- $\lambda$ (lambda) = regularization strength
- Higher $\lambda$ = more regularization = simpler model

```python
from sklearn.linear_model import Ridge

# No regularization
model_no_reg = LinearRegression()
model_no_reg.fit(X_train, y_train)
print(f"Weights: {model_no_reg.coef_}")
# Weights: [123.4, -567.8, 901.2, -345.6, ...]  ← Large weights

# With L2 regularization
model_ridge = Ridge(alpha=1.0)  # alpha = λ
model_ridge.fit(X_train, y_train)
print(f"Weights: {model_ridge.coef_}")
# Weights: [12.3, -5.6, 9.0, -3.4, ...]  ← Smaller weights
```

**Effect:** Forces weights to be smaller, preventing the model from relying too heavily on any single feature.

**L1 Regularization (Lasso):**

Adds penalty proportional to the absolute value of weights:

$$J(\vec{w}) = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$$

```python
from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=1.0)
model_lasso.fit(X_train, y_train)
print(f"Weights: {model_lasso.coef_}")
# Weights: [12.3, 0.0, 9.0, 0.0, ...]  ← Some weights become exactly 0!
```

**Effect:** Forces some weights to exactly zero, effectively performing feature selection.

**Choosing regularization strength:**

```python
# Try different values of alpha (λ)
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Alpha={alpha}: Train={train_score:.3f}, Val={val_score:.3f}")

# Typical output:
# Alpha=0.001: Train=0.95, Val=0.70  ← Overfitting
# Alpha=0.01:  Train=0.93, Val=0.75
# Alpha=0.1:   Train=0.90, Val=0.85  ← Good balance
# Alpha=1.0:   Train=0.85, Val=0.84  ← Still good
# Alpha=10.0:  Train=0.75, Val=0.75  ← Underfitting
# Alpha=100.0: Train=0.60, Val=0.60  ← Severe underfitting
```

#### 3. Feature Selection

**Remove irrelevant or redundant features.**

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top 10 most important features
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# See which features were selected
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")
```

**Manual feature selection based on domain knowledge:**

```python
# Before: 50 features (many irrelevant)
# After: 10 most relevant features

# Example: Predicting house prices
relevant_features = [
    'square_feet',
    'bedrooms',
    'bathrooms',
    'age',
    'location_score'
]

X_train_selected = X_train[relevant_features]
```

#### 4. Cross-Validation

**Use multiple train/validation splits to detect overfitting.**

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")

# If std is high, model is unstable (overfitting to specific folds)
if scores.std() > 0.1:
    print("Warning: High variance across folds!")
```

#### 5. Early Stopping

**Stop training when validation performance stops improving.**

```python
# Pseudo-code for training with early stopping
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait
patience_counter = 0

for epoch in range(1000):
    # Train for one epoch
    train_loss = train_one_epoch(model, X_train, y_train)
    
    # Evaluate on validation set
    val_loss = evaluate(model, X_val, y_val)
    
    print(f"Epoch {epoch}: Train loss={train_loss:.3f}, Val loss={val_loss:.3f}")
    
    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model(model)  # Save best model
    else:
        patience_counter += 1
    
    # Stop if no improvement for 'patience' epochs
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        load_model()  # Restore best model
        break

# Example output:
# Epoch 0: Train loss=0.500, Val loss=0.600
# Epoch 1: Train loss=0.300, Val loss=0.500
# Epoch 2: Train loss=0.200, Val loss=0.450
# Epoch 3: Train loss=0.150, Val loss=0.440  ← Best
# Epoch 4: Train loss=0.100, Val loss=0.450
# Epoch 5: Train loss=0.080, Val loss=0.460
# Epoch 6: Train loss=0.060, Val loss=0.470
# Epoch 7: Train loss=0.040, Val loss=0.480
# Epoch 8: Train loss=0.020, Val loss=0.490
# Early stopping at epoch 8
# Restoring model from epoch 3
```

#### 6. Dropout (Neural Networks)

**Randomly disable neurons during training.**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.5),  # Randomly drop 50% of neurons
    Dense(64, activation='relu'),
    Dropout(0.3),  # Randomly drop 30% of neurons
    Dense(1)
])

# During training: Random neurons are disabled
# During testing: All neurons are active
```

**Why it works:** Forces the network to learn redundant representations, preventing it from relying too heavily on specific neurons.

#### 7. Ensemble Methods

**Combine multiple models to reduce overfitting.**

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Train multiple models
model1 = LinearRegression()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Average predictions
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

ensemble_pred = (pred1 + pred2 + pred3) / 3

# Ensemble often generalizes better than individual models
```

---

## Underfitting

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. It performs poorly on both training and test data.

### Signs of Underfitting

**1. Poor performance on both training and test data**

```python
train_accuracy = 0.65  # 65%
test_accuracy = 0.63   # 63%

# Both are low!
# Model isn't learning the patterns
```

**2. Training and test errors are similar (both high)**

```python
train_error = 0.35
test_error = 0.37
gap = test_error - train_error  # 0.02 (small gap)

# Small gap is good, but both errors are high!
# This indicates underfitting, not good generalization
```

**3. Model is too simple**

```python
# Example: Using linear model for non-linear data
# Data has quadratic relationship: y = x²
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Linear model: y = mx + b
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)

# Predictions: [3, 7, 11, 15, 19]
# Actual: [1, 4, 9, 16, 25]
# Poor fit! Linear model can't capture quadratic relationship
```

### Real Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 3, (100, 1))

# Split data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Model 1: Linear (underfits)
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

train_error = mean_squared_error(y_train, model_linear.predict(X_train))
test_error = mean_squared_error(y_test, model_linear.predict(X_test))

print("Linear Model (Underfitting):")
print(f"  Train MSE: {train_error:.2f}")
print(f"  Test MSE: {test_error:.2f}")
print(f"  Gap: {test_error - train_error:.2f}")

# Model 2: Quadratic (good fit)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

train_error_poly = mean_squared_error(y_train, model_poly.predict(X_train_poly))
test_error_poly = mean_squared_error(y_test, model_poly.predict(X_test_poly))

print("\nQuadratic Model (Good Fit):")
print(f"  Train MSE: {train_error_poly:.2f}")
print(f"  Test MSE: {test_error_poly:.2f}")
print(f"  Gap: {test_error_poly - train_error_poly:.2f}")

# Typical output:
# Linear Model (Underfitting):
#   Train MSE: 45.23
#   Test MSE: 47.89
#   Gap: 2.66  ← Small gap, but both errors are high!
#
# Quadratic Model (Good Fit):
#   Train MSE: 8.12
#   Test MSE: 9.45
#   Gap: 1.33  ← Small gap, and both errors are low ✓
```

### Causes of Underfitting

#### 1. Model Too Simple

**Problem:** Model doesn't have enough capacity to learn the patterns.

```python
# Example: Predicting house prices with complex relationships

# Too simple: Only use square footage
# price = w * square_feet + b
# Ignores: location, bedrooms, age, etc.

# Better: Use multiple features
# price = w1*sqft + w2*bedrooms + w3*location + w4*age + b
```

#### 2. Too Few Features

**Problem:** Missing important information.

```python
# Predicting salary with only one feature
X = df[['years_experience']]  # Only one feature

# Missing important features:
# - Education level
# - Industry
# - Location
# - Job title
# - Company size
```

#### 3. Too Much Regularization

**Problem:** Regularization penalty is too strong, preventing the model from learning.

```python
from sklearn.linear_model import Ridge

# Regularization too strong
model = Ridge(alpha=1000.0)  # Very high alpha
model.fit(X_train, y_train)

# Weights are forced to be very small
print(model.coef_)
# [0.001, 0.002, 0.001, ...]  ← Nearly zero!

# Model essentially predicts the mean
# Can't learn the patterns
```

#### 4. Insufficient Training

**Problem:** Model hasn't trained long enough to learn patterns.

```python
# Neural network trained for only 1 epoch
model.fit(X_train, y_train, epochs=1)

# Not enough iterations to learn
# Weights are still close to random initialization
```

### Fixing Underfitting

#### 1. Increase Model Complexity

**Add more capacity to the model.**

```python
# Before: Linear model
model = LinearRegression()

# After: Polynomial model
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
```

**Or use a more complex model:**

```python
# Before: Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# After: Neural network
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
```

#### 2. Add More Features

**Include more relevant information.**

```python
# Before: Only basic features
features = ['square_feet', 'bedrooms']

# After: Add more features
features = [
    'square_feet',
    'bedrooms',
    'bathrooms',
    'age',
    'location_score',
    'school_rating',
    'crime_rate'
]
```

#### 3. Feature Engineering

**Create new features from existing ones.**

```python
# Original features
df['square_feet'] = [1200, 1500, 1800]
df['bedrooms'] = [2, 3, 3]

# Engineered features
df['sqft_per_bedroom'] = df['square_feet'] / df['bedrooms']
df['sqft_squared'] = df['square_feet'] ** 2
df['sqft_bedrooms_interaction'] = df['square_feet'] * df['bedrooms']

# These new features might help the model learn better
```

#### 4. Reduce Regularization

**Lower the regularization strength.**

```python
# Before: Strong regularization
model = Ridge(alpha=100.0)

# After: Weaker regularization
model = Ridge(alpha=0.1)

# Or remove regularization entirely
model = LinearRegression()
```

#### 5. Train Longer

**Give the model more time to learn.**

```python
# Before: Few epochs
model.fit(X_train, y_train, epochs=10)

# After: More epochs
model.fit(X_train, y_train, epochs=100)
```

#### 6. Use a Different Algorithm

**Some algorithms are better suited for certain problems.**

```python
# Linear data: Linear regression works well
# Non-linear data: Try polynomial regression, decision trees, or neural networks

# Before: Linear regression on non-linear data
model = LinearRegression()

# After: Decision tree (handles non-linearity)
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5)

# Or: Random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
```

---

## Bias-Variance Tradeoff

The **bias-variance tradeoff** is the fundamental tension in machine learning between two sources of error.

### Definitions

**Bias:** Error from overly simplistic assumptions in the learning algorithm.
- **High bias** → Model makes strong assumptions → Underfitting
- Model consistently misses the true pattern
- Example: Using a straight line to fit a curved relationship

**Variance:** Error from sensitivity to small fluctuations in the training data.
- **High variance** → Model is too sensitive to training data → Overfitting
- Model changes drastically with different training sets
- Example: A wiggly curve that passes through every training point

**Irreducible Error:** Error that cannot be reduced regardless of the model.
- Comes from noise in the data
- Represents the best possible error we can achieve

### The Mathematical Decomposition

The expected prediction error can be decomposed into three components:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Bias:**
$$\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x)$$

Where:
- $\hat{f}(x)$ = our model's prediction
- $f(x)$ = true function
- $\mathbb{E}[\hat{f}(x)]$ = expected prediction across different training sets

**Variance:**
$$\text{Variance} = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

Measures how much $\hat{f}(x)$ varies across different training sets.

### Visual Understanding

```
High Bias (Underfitting):
Target: ∿∿∿∿∿∿∿∿ (curved)
Model:  ————————— (straight line)

Problem: Model is too simple
- Consistently misses the curve
- Same error regardless of training data
- High bias, low variance

High Variance (Overfitting):
Target: ∿∿∿∿∿∿∿∿ (smooth curve)
Model:  ∿∿∿∿∿∿∿∿ (wiggly curve)

Problem: Model is too complex
- Fits training data perfectly
- Changes drastically with different training data
- Low bias, high variance

Balanced:
Target: ∿∿∿∿∿∿∿∿ (smooth curve)
Model:  ∿∿∿∿∿∿∿∿ (similar smooth curve)

Result: Captures true pattern
- Low bias, low variance
```

### The Tradeoff

As model complexity increases:
- **Bias decreases** (model can fit more complex patterns)
- **Variance increases** (model becomes more sensitive to training data)

```
Error
  ↑
  │
  │     Total Error
  │        ∿∿∿
  │      ∿     ∿
  │    ∿         ∿
  │   ∿           ∿∿∿∿∿∿∿
  │  ∿       ↑
  │ ∿    Sweet spot
  │∿
  │ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  Bias²
  │
  │                 ∿∿∿∿∿∿∿  Variance
  │               ∿∿
  │             ∿∿
  │           ∿∿
  │         ∿∿
  │       ∿∿
  │     ∿∿
  │   ∿∿
  │ ∿∿
  │∿
  └────────────────────→ Model Complexity
  Simple              Complex
```

**Key insight:** You can't minimize both bias and variance simultaneously. Reducing one increases the other.

### Practical Example

Let's demonstrate with polynomial regression:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# True function: y = sin(x) + noise
np.random.seed(42)
X_true = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
y_true = np.sin(X_true).ravel()

# Generate multiple training sets
n_datasets = 100
n_samples = 20

results = []

for degree in [1, 2, 5, 15]:  # Different model complexities
    predictions = []
    
    for _ in range(n_datasets):
        # Generate training set
        X_train = np.random.uniform(0, 2*np.pi, n_samples).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, n_samples)
        
        # Train model
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_true_poly = poly.transform(X_true)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predict on true X
        y_pred = model.predict(X_true_poly)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # Calculate bias and variance
    mean_prediction = predictions.mean(axis=0)
    bias_squared = ((mean_prediction - y_true) ** 2).mean()
    variance = predictions.var(axis=0).mean()
    
    print(f"Degree {degree}:")
    print(f"  Bias²: {bias_squared:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Total: {bias_squared + variance:.4f}")
    print()

# Typical output:
# Degree 1 (Linear):
#   Bias²: 0.2500      ← High bias (can't fit sine wave)
#   Variance: 0.0050   ← Low variance (stable)
#   Total: 0.2550
#
# Degree 2 (Quadratic):
#   Bias²: 0.1200      ← Medium bias
#   Variance: 0.0100   ← Low variance
#   Total: 0.1300      ← Best total error!
#
# Degree 5:
#   Bias²: 0.0200      ← Low bias (can fit sine wave)
#   Variance: 0.0800   ← Medium variance
#   Total: 0.1000
#
# Degree 15:
#   Bias²: 0.0010      ← Very low bias
#   Variance: 0.5000   ← Very high variance (unstable!)
#   Total: 0.5010      ← Worse total error!
```

**Analysis:**

| Degree | Bias² | Variance | Total | Diagnosis |
|--------|-------|----------|-------|-----------|
| 1 | High | Low | High | Underfitting (high bias) |
| 2 | Medium | Low | **Low** | **Optimal balance** ✓ |
| 5 | Low | Medium | Medium | Slight overfitting |
| 15 | Very low | Very high | High | Overfitting (high variance) |

### Diagnosing Bias vs Variance Problems

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| High train error, high test error | **High bias** (underfitting) | Increase complexity, add features |
| Low train error, high test error | **High variance** (overfitting) | Decrease complexity, regularization, more data |
| Low train error, low test error | **Good balance** ✓ | Keep it! |
| High train error, low test error | **Unusual** (check for bugs) | Verify data and code |

**Decision tree:**

```
Check training error:
├─ High training error?
│  └─ High bias problem
│     ├─ Increase model complexity
│     ├─ Add more features
│     ├─ Reduce regularization
│     └─ Train longer
│
└─ Low training error?
   └─ Check test error:
      ├─ High test error?
      │  └─ High variance problem
      │     ├─ Get more training data
      │     ├─ Reduce model complexity
      │     ├─ Add regularization
      │     ├─ Feature selection
      │     └─ Early stopping
      │
      └─ Low test error?
         └─ Good! Model is working well ✓
```

### Practical Strategies

**To reduce bias (underfitting):**
1. Use more complex model
2. Add more features
3. Reduce regularization
4. Train longer
5. Remove feature selection

**To reduce variance (overfitting):**
1. Get more training data
2. Use simpler model
3. Add regularization
4. Feature selection
5. Early stopping
6. Ensemble methods

**To find the sweet spot:**
1. Start simple, gradually increase complexity
2. Use cross-validation to evaluate
3. Plot learning curves
4. Monitor train/validation gap
5. Use regularization to fine-tune

---

## Learning Curves

**Learning curves** plot model performance as a function of training set size. They're invaluable for diagnosing bias and variance problems.

### What Are Learning Curves?

Learning curves show how training and validation errors change as you add more training data.

**X-axis:** Training set size (number of samples)
**Y-axis:** Error (or accuracy)
**Two lines:** Training error and validation error

### Interpreting Learning Curves

#### 1. High Bias (Underfitting)

```
Error
  ↑
  │ ————————————————  ← Training error (high)
  │ ————————————————  ← Validation error (high)
  │                     Both converge to high error
  │                     Small gap between them
  │                     More data won't help!
  └────────────────→ Training Set Size
```

**Characteristics:**
- Both errors are high
- Both errors converge quickly
- Small gap between training and validation error
- Curves plateau early

**Diagnosis:** High bias (underfitting)

**Solution:** More data won't help! Need a more complex model.

#### 2. High Variance (Overfitting)

```
Error
  ↑
  │         ————————  ← Validation error (high)
  │        ∕
  │       ∕
  │      ∕
  │     ∕
  │    ∕
  │   ∕
  │  ∕
  │ ∕———————————————  ← Training error (low)
  │                     Large gap between curves
  │                     More data will help!
  └────────────────→ Training Set Size
```

**Characteristics:**
- Training error is low
- Validation error is high
- Large gap between the two
- Validation error decreases as data increases
- Curves haven't converged yet

**Diagnosis:** High variance (overfitting)

**Solution:** More data will help! Or use regularization.

#### 3. Good Fit

```
Error
  ↑
  │     ————————————  ← Validation error (low)
  │    ∕
  │   ∕
  │  ∕
  │ ∕———————————————  ← Training error (low)
  │                     Small gap
  │                     Both errors are low
  │                     Curves have converged
  └────────────────→ Training Set Size
```

**Characteristics:**
- Both errors are low
- Small gap between them
- Curves have converged
- Adding more data won't significantly improve performance

**Diagnosis:** Good fit ✓

**Solution:** Model is working well! Deploy it.

### Creating Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y, cv=5):
    """
    Plot learning curves for a model
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # Calculate mean and std
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training error', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation error', marker='o')
    
    # Add shaded regions for std
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# High bias model (too simple)
model_simple = LinearRegression()
plot_learning_curve(model_simple, X, y)

# High variance model (too complex)
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)
model_complex = LinearRegression()
plot_learning_curve(model_complex, X_poly, y)

# Good fit model (balanced)
model_balanced = Ridge(alpha=1.0)
plot_learning_curve(model_balanced, X, y)
```

### Using Learning Curves for Decisions

**Scenario 1: Both curves are high and flat**
```
Action: Increase model complexity
Reason: Model is too simple (high bias)
More data won't help
```

**Scenario 2: Large gap, validation curve is decreasing**
```
Action: Get more training data
Reason: Model is overfitting (high variance)
More data will reduce the gap
```

**Scenario 3: Both curves are low and converged**
```
Action: Deploy the model!
Reason: Model is working well
More data won't significantly improve performance
```

**Scenario 4: Training error is increasing**
```
Action: Check for bugs or data issues
Reason: Training error should decrease or stay flat
Increasing training error is unusual
```

---

## Summary

### Key Concepts

**Generalization:**
- Goal: Perform well on new, unseen data
- Measure: Test set performance
- Indicator: Small gap between train and test performance

**Overfitting:**
- Problem: Model memorizes training data
- Signs: Low train error, high test error, large gap
- Solutions: More data, regularization, simpler model, early stopping

**Underfitting:**
- Problem: Model is too simple
- Signs: High train error, high test error, small gap
- Solutions: More complex model, more features, less regularization

**Bias-Variance Tradeoff:**
- Bias: Error from simplistic assumptions (underfitting)
- Variance: Error from sensitivity to data (overfitting)
- Goal: Balance both for minimum total error

**Learning Curves:**
- Tool: Diagnose bias and variance problems
- High bias: Both curves high and flat
- High variance: Large gap, validation curve decreasing
- Good fit: Both curves low and converged

### Decision Framework

```
1. Train model on training set
2. Evaluate on validation set

3. Check training performance:
   ├─ High training error?
   │  └─ High bias (underfitting)
   │     → Increase complexity
   │
   └─ Low training error?
      └─ Check validation performance:
         ├─ High validation error?
         │  └─ High variance (overfitting)
         │     → Add regularization or more data
         │
         └─ Low validation error?
            └─ Good fit!
               → Evaluate on test set
               → Deploy if test performance is good

4. Use learning curves to confirm diagnosis
5. Iterate until satisfied
6. Final evaluation on test set (once!)
```

### Quick Reference Table

| Symptom | Train Error | Test Error | Gap | Problem | Solution |
|---------|-------------|------------|-----|---------|----------|
| Underfitting | High | High | Small | High bias | Increase complexity |
| Overfitting | Low | High | Large | High variance | Regularization, more data |
| Good fit | Low | Low | Small | None | Deploy! ✓ |

---

## Next Steps

- [Working with Data](./notes.md) — data preparation foundations
- [Gradient Descent](../gradient-descent/notes.md) — optimization algorithms
- [Neural Networks](../../advanced%20ml%20conceptss/neural-networks.md) — modern deep learning
