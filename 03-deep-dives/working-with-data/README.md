# Working with Data

A comprehensive guide to understanding and preparing data for machine learning.

---

## Overview

Before building models, you must understand your data. This guide covers everything from basic data types to advanced concepts like overfitting and the bias-variance tradeoff.

**What you'll learn:**
- Types of data (numerical vs categorical)
- How to encode and scale features
- Dataset splitting strategies
- Generalization and why it matters
- Overfitting and underfitting
- The bias-variance tradeoff
- How to diagnose and fix model problems

---

## Files in This Section

### [notes.md](./notes.md) - Main Guide
**Topics covered:**
- **Data Types**
  - Numerical data (continuous vs discrete)
  - Feature scaling (normalization vs standardization)
  - Categorical data (nominal vs ordinal)
  - Encoding methods (label, one-hot, target)
- **Datasets and Splits**
  - Train/test splitting
  - Train/validation/test splitting
  - Cross-validation
  - Data leakage prevention
- **Practical Workflow**
  - Complete data preparation pipeline
  - Common mistakes to avoid
  - Checklist for data preparation

### [generalization-overfitting.md](./generalization-overfitting.md) - Deep Dive
**Topics covered:**
- **Generalization**
  - What it means to generalize
  - Factors affecting generalization
  - Mathematical perspective
- **Overfitting**
  - Visual understanding
  - Real examples with code
  - Signs and causes
  - Prevention strategies (7 methods)
- **Underfitting**
  - Signs and causes
  - How to fix it
- **Bias-Variance Tradeoff**
  - Mathematical decomposition
  - Visual understanding
  - Practical examples
  - Diagnostic strategies
- **Learning Curves**
  - How to create and interpret them
  - Diagnosing bias vs variance
  - Using them for decisions

---

## Quick Start

### 1. Understanding Data Types

```python
import pandas as pd
import numpy as np

# Numerical data: Can be used directly in calculations
numerical_data = [1200, 1500, 1800, 2200, 2500]  # Square feet

# Categorical data: Must be encoded
categorical_data = ['Red', 'Blue', 'Green', 'Red', 'Blue']  # Colors
```

### 2. Encoding Categorical Data

```python
# One-hot encoding for nominal data (no order)
colors_encoded = pd.get_dummies(categorical_data)

# Label encoding for ordinal data (has order)
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
```

### 3. Scaling Numerical Data

```python
from sklearn.preprocessing import StandardScaler

# Standardization (most common)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. Splitting Data

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 5. Diagnosing Model Problems

```python
# Train and evaluate
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Diagnose
if train_score < 0.7 and test_score < 0.7:
    print("Underfitting: Model too simple")
elif train_score > 0.95 and test_score < 0.7:
    print("Overfitting: Model too complex")
else:
    print("Good fit!")
```

---

## When to Read What

| Your Question | Read This |
|---------------|-----------|
| "What's the difference between continuous and discrete data?" | [notes.md - Numerical Data](./notes.md#numerical-data) |
| "How do I encode categorical features?" | [notes.md - Categorical Data](./notes.md#categorical-data) |
| "Should I use normalization or standardization?" | [notes.md - Feature Scaling](./notes.md#feature-scaling-for-numerical-data) |
| "How do I split my data properly?" | [notes.md - Datasets and Splits](./notes.md#datasets-and-splits) |
| "What is data leakage?" | [notes.md - Data Leakage](./notes.md#data-leakage) |
| "My model has 99% training accuracy but 60% test accuracy" | [generalization-overfitting.md - Overfitting](./generalization-overfitting.md#overfitting) |
| "My model performs poorly on both train and test" | [generalization-overfitting.md - Underfitting](./generalization-overfitting.md#underfitting) |
| "What's the bias-variance tradeoff?" | [generalization-overfitting.md - Bias-Variance](./generalization-overfitting.md#bias-variance-tradeoff) |
| "How do I interpret learning curves?" | [generalization-overfitting.md - Learning Curves](./generalization-overfitting.md#learning-curves) |
| "Complete workflow from raw data to model?" | [notes.md - Practical Workflow](./notes.md#practical-workflow) |

---

## Key Concepts at a Glance

### Data Types

| Type | Subtypes | Examples | Encoding |
|------|----------|----------|----------|
| **Numerical** | Continuous | Height, price, temperature | Scale (standardize/normalize) |
| | Discrete | Count, age in years | Scale (standardize/normalize) |
| **Categorical** | Nominal | Colors, countries | One-hot or target encoding |
| | Ordinal | Education level, ratings | Label encoding |

### Model Problems

| Problem | Train Error | Test Error | Gap | Solution |
|---------|-------------|------------|-----|----------|
| **Underfitting** | High | High | Small | Increase complexity |
| **Overfitting** | Low | High | Large | Regularization, more data |
| **Good Fit** | Low | Low | Small | Deploy! ✓ |

### Encoding Methods

| Method | Use For | Pros | Cons |
|--------|---------|------|------|
| **Label Encoding** | Ordinal data | Simple, preserves order | Creates false ordering for nominal data |
| **One-Hot Encoding** | Nominal data (low cardinality) | No false ordering | High dimensionality |
| **Target Encoding** | High-cardinality nominal data | Handles many categories | Overfitting risk |

### Scaling Methods

| Method | Formula | Range | Use When |
|--------|---------|-------|----------|
| **Normalization** | $(x - x_{min}) / (x_{max} - x_{min})$ | [0, 1] | Need bounded values |
| **Standardization** | $(x - \mu) / \sigma$ | Unbounded | Data has outliers (most common) |

---

## Prerequisites

**Required knowledge:**
- Basic Python programming
- NumPy and Pandas basics
- Basic statistics (mean, standard deviation)

**Helpful but not required:**
- Linear algebra basics
- Calculus (for understanding bias-variance mathematically)

---

## Recommended Reading Order

1. **Start here:** [notes.md](./notes.md)
   - Read sections on data types
   - Understand encoding and scaling
   - Learn about dataset splitting

2. **Then read:** [generalization-overfitting.md](./generalization-overfitting.md)
   - Understand generalization
   - Learn about overfitting and underfitting
   - Master the bias-variance tradeoff

3. **Practice:** Use the practical workflow
   - Apply to your own datasets
   - Experiment with different encodings
   - Diagnose and fix model problems

4. **Reference:** Come back when you encounter issues
   - Use as a troubleshooting guide
   - Refer to specific sections as needed

---

## Practical Examples

All examples use real code that you can run:

```python
# Complete example: Preparing data for machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Create complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
print(f"Train score: {pipeline.score(X_train, y_train):.3f}")
print(f"Test score: {pipeline.score(X_test, y_test):.3f}")
```

---

## Common Questions

**Q: Should I always standardize my data?**  
A: For most algorithms (linear models, neural networks, SVM), yes. Tree-based models (decision trees, random forests) don't require scaling.

**Q: When should I use one-hot encoding vs label encoding?**  
A: Use one-hot for nominal data (no order), label encoding for ordinal data (has order).

**Q: How do I know if my model is overfitting?**  
A: Check if training accuracy is much higher than test accuracy (gap > 10%).

**Q: What's the difference between validation and test sets?**  
A: Validation set is for tuning hyperparameters (use multiple times). Test set is for final evaluation (use only once).

**Q: How much data should I use for testing?**  
A: Typically 20% for test set. For small datasets (<1000 samples), use 30%. For large datasets (>100k), 10% is sufficient.

**Q: What is data leakage?**  
A: When information from the test set "leaks" into training, giving unrealistic performance estimates. Always split data before preprocessing!

---

## Next Steps

After mastering this material:

1. **Apply to real datasets** — Kaggle competitions, UCI Repository, your own projects

2. **Explore algorithms**:
   - [Linear Regression](../../01-supervised-learning/linear-regression/notes.md)
   - [Logistic Regression](../../01-supervised-learning/logistic-regression/notes.md)
   - [Clustering](../../02-unsupervised-learning/clustering/notes.md)

3. **Study optimization**:
   - [Gradient Descent](../gradient-descent/notes.md)
   - [Multiple Linear Regression](../multiple-linear-regression/notes.md)

---

**Remember:** Good data preparation is more important than a fancy algorithm. Spend time understanding your data, and your models will perform better!
