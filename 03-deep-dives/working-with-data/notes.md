# Working with Data

Understanding your data is the foundation of successful machine learning. Before you can build models, you must understand what data you have, how to prepare it, and how to ensure your model will work on new, unseen data.

This guide covers everything from basic data types to advanced concepts like overfitting and the bias-variance tradeoff.

---

## Table of Contents

1. [Types of Data](#types-of-data)
   - [Numerical Data](#numerical-data)
   - [Categorical Data](#categorical-data)
2. [Datasets and Splits](#datasets-and-splits)
3. [Practical Workflow](#practical-workflow)

**For Generalization, Overfitting, and Bias-Variance:** See [generalization-overfitting.md](./generalization-overfitting.md)

---

## Types of Data

Machine learning algorithms fundamentally work with numbers. However, real-world data comes in many forms. Understanding the **type** of data you have determines how you should prepare it for your model.

There are two fundamental categories:
1. **Numerical data** — Already numbers, can be used in mathematical operations
2. **Categorical data** — Labels or categories that must be converted to numbers

---

## Numerical Data

**Numerical data** (also called **quantitative data**) consists of numbers that represent measurable quantities. These values have mathematical meaning—you can add them, subtract them, find averages, etc.

### Why Numerical Data Matters

Machine learning models perform mathematical operations on inputs. When you have numerical data, you can:
- Compute distances between points
- Calculate averages and standard deviations
- Apply mathematical transformations
- Use the data directly in equations

### Types of Numerical Data

#### 1. Continuous Data

**Continuous data** can take **any value** within a range. Between any two values, there are infinitely many possible values.

**Characteristics:**
- Can be measured with arbitrary precision
- Often involves decimals
- Represents measurements on a continuous scale

**Examples:**
- **Height:** 175.3 cm, 180.7 cm, 162.45 cm
- **Temperature:** 23.7°C, -5.2°C, 98.6°F
- **Price:** $45.99, $1,234.56, $0.01
- **Time:** 3.14 seconds, 2.5 hours
- **Weight:** 68.5 kg, 150.3 lbs
- **Distance:** 42.195 km (marathon distance)

**Mathematical representation:**
$$x \in \mathbb{R} \text{ where } x \in [a, b]$$

For example, height might be: $x \in [0, 300]$ cm

**Why "continuous"?**
Between 175 cm and 176 cm, there are infinite possible heights: 175.1, 175.01, 175.001, 175.0001, etc.

#### 2. Discrete Data

**Discrete data** can only take **specific, separate values**. Usually integers. You can count discrete values, but you can't have values "in between."

**Characteristics:**
- Countable
- Usually whole numbers
- Represents counts or distinct categories with order

**Examples:**
- **Number of students:** 25, 30, 42 (not 25.5 students!)
- **Age in years:** 18, 25, 67 (we don't usually say 25.3 years old)
- **Number of purchases:** 0, 1, 2, 3, ...
- **Number of bedrooms:** 1, 2, 3, 4
- **Dice roll:** 1, 2, 3, 4, 5, 6
- **Number of clicks:** 0, 1, 2, 3, ...

**Mathematical representation:**
$$x \in \mathbb{Z} \text{ where } x \in \{0, 1, 2, 3, ...\}$$

**Why "discrete"?**
You can't have 2.5 bedrooms or 3.7 students. The values are separated with gaps between them.

### Working with Numerical Data in Python

```python
import numpy as np
import pandas as pd

# Example: House prices dataset
data = {
    'square_feet': [1200, 1500, 1800, 2200, 2500],  # Continuous
    'bedrooms': [2, 3, 3, 4, 4],                     # Discrete
    'bathrooms': [1.5, 2.0, 2.5, 3.0, 3.5],         # Continuous (can have half-baths)
    'age_years': [5, 12, 8, 3, 15],                 # Discrete
    'price': [250000, 300000, 350000, 425000, 475000]  # Continuous
}
df = pd.DataFrame(data)

print(df)
#    square_feet  bedrooms  bathrooms  age_years   price
# 0         1200         2        1.5          5  250000
# 1         1500         3        2.0         12  300000
# 2         1800         3        2.5          8  350000
# 3         2200         4        3.0          3  425000
# 4         2500         4        3.5         15  475000

# Numerical data can be directly used in calculations
mean_price = df['price'].mean()
print(f"Average price: ${mean_price:,.0f}")  # Average price: $360,000

# Compute correlation between features
correlation = df['square_feet'].corr(df['price'])
print(f"Correlation: {correlation:.3f}")  # Correlation: 0.995
```

### The Problem: Different Scales

Consider this dataset:

| House | Square Feet | Bedrooms | Price |
|-------|-------------|----------|-------|
| 1 | 1200 | 2 | 250000 |
| 2 | 1500 | 3 | 300000 |
| 3 | 1800 | 3 | 350000 |

Notice the **scale differences**:
- Square feet: 1200 - 1800 (range of 600)
- Bedrooms: 2 - 3 (range of 1)
- Price: 250000 - 350000 (range of 100000)

**Why this is a problem:**

Many machine learning algorithms compute **distances** between data points. When features have different scales, features with larger values dominate the distance calculation.

**Example: Computing distance between House 1 and House 2**

Using Euclidean distance:
$$d = \sqrt{(1500-1200)^2 + (3-2)^2 + (300000-250000)^2}$$
$$d = \sqrt{300^2 + 1^2 + 50000^2}$$
$$d = \sqrt{90000 + 1 + 2500000000}$$
$$d \approx 50000$$

The price difference (50000) completely dominates! The bedroom difference (1) is essentially ignored.

**Solution:** Feature scaling

---

### Feature Scaling for Numerical Data

**Feature scaling** transforms features to similar scales so no single feature dominates.

#### Method 1: Normalization (Min-Max Scaling)

**Normalization** scales features to a fixed range, typically [0, 1].

**Formula:**
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Intuition:**
- Subtract the minimum → Shifts the smallest value to 0
- Divide by the range → Scales the largest value to 1
- Everything else falls proportionally between 0 and 1

**Step-by-step example:**

```python
import numpy as np

# Original data: house square footage
square_feet = np.array([1200, 1500, 1800, 2200, 2500])

# Step 1: Find min and max
x_min = square_feet.min()  # 1200
x_max = square_feet.max()  # 2500

# Step 2: Apply formula
normalized = (square_feet - x_min) / (x_max - x_min)

print("Original:", square_feet)
print("Normalized:", normalized)
# Original: [1200 1500 1800 2200 2500]
# Normalized: [0.0  0.23 0.46 0.77 1.0]
```

**Manual calculation for first value:**
$$x_{norm} = \frac{1200 - 1200}{2500 - 1200} = \frac{0}{1300} = 0.0$$

**Manual calculation for second value:**
$$x_{norm} = \frac{1500 - 1200}{2500 - 1200} = \frac{300}{1300} \approx 0.23$$

**Properties:**
- Minimum value becomes 0
- Maximum value becomes 1
- Preserves the shape of the distribution
- Bounded: all values in [0, 1]

**When to use:**
- When you need bounded values (e.g., neural networks with sigmoid activation)
- When you know the min and max values are meaningful
- When the distribution doesn't have extreme outliers

**Disadvantage:**
- Sensitive to outliers (one extreme value can compress all others)

#### Method 2: Standardization (Z-score Normalization)

**Standardization** transforms data to have mean = 0 and standard deviation = 1.

**Formula:**
$$x_{std} = \frac{x - \mu}{\sigma}$$

Where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

**Intuition:**
- Subtract the mean → Centers the data around 0
- Divide by standard deviation → Scales to unit variance
- Result: "How many standard deviations away from the mean?"

**Step-by-step example:**

```python
import numpy as np

# Original data: house prices
prices = np.array([250000, 300000, 350000, 425000, 475000])

# Step 1: Calculate mean
mean = prices.mean()  # 360000

# Step 2: Calculate standard deviation
std = prices.std()  # 79056.94

# Step 3: Apply formula
standardized = (prices - mean) / std

print("Original:", prices)
print("Standardized:", standardized)
# Original: [250000 300000 350000 425000 475000]
# Standardized: [-1.39 -0.76 -0.13  0.82  1.45]
```

**Manual calculation for first value:**
$$x_{std} = \frac{250000 - 360000}{79056.94} = \frac{-110000}{79056.94} \approx -1.39$$

This means 250000 is 1.39 standard deviations **below** the mean.

**Properties:**
- Mean becomes 0
- Standard deviation becomes 1
- Not bounded (can be any value)
- Less sensitive to outliers than normalization

**When to use:**
- When features have different units (meters, kilograms, dollars)
- When data has outliers
- When you don't know the theoretical min/max
- Default choice for most algorithms (SVM, logistic regression, neural networks)

**Comparison:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Original data with an outlier
data = np.array([10, 12, 13, 14, 15, 16, 18, 100])  # 100 is an outlier

# Normalization
normalized = (data - data.min()) / (data.max() - data.min())

# Standardization
standardized = (data - data.mean()) / data.std())

print("Original:     ", data)
print("Normalized:   ", normalized)
print("Standardized: ", standardized)

# Original:      [ 10  12  13  14  15  16  18 100]
# Normalized:    [0.0  0.02 0.03 0.04 0.06 0.07 0.09 1.0]
#                 ↑ All compressed near 0 because of outlier!
# Standardized:  [-1.18 -1.11 -1.07 -1.04 -1.0 -0.96 -0.89 2.26]
#                 ↑ Better spread, outlier is identified
```

**Key insight:** Normalization compresses most values near 0 when there's an outlier. Standardization handles outliers better.

---

### Practical Implementation

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Sample data
data = {
    'square_feet': [1200, 1500, 1800, 2200, 2500],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [250000, 300000, 350000, 425000, 475000]
}
df = pd.DataFrame(data)

# Method 1: Normalization
scaler_norm = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_norm.fit_transform(df),
    columns=df.columns
)

print("Normalized:")
print(df_normalized)
#    square_feet  bedrooms  price
# 0         0.00      0.00   0.00
# 1         0.23      0.50   0.22
# 2         0.46      0.50   0.44
# 3         0.77      1.00   0.78
# 4         1.00      1.00   1.00

# Method 2: Standardization
scaler_std = StandardScaler()
df_standardized = pd.DataFrame(
    scaler_std.fit_transform(df),
    columns=df.columns
)

print("\nStandardized:")
print(df_standardized)
#    square_feet  bedrooms     price
# 0        -1.41     -1.41     -1.39
# 1        -0.71      0.00     -0.76
# 2         0.00      0.00     -0.13
# 3         1.06      1.41      0.82
# 4         1.41      1.41      1.45
```

**Critical rule:** Always fit the scaler on training data only, then transform both training and test data:

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit scaler on training data ONLY
scaler = StandardScaler()
scaler.fit(X_train)  # Learn mean and std from training data

# Transform both sets using the same scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# WRONG: Don't do this!
# X_train_scaled = StandardScaler().fit_transform(X_train)
# X_test_scaled = StandardScaler().fit_transform(X_test)
# ↑ This uses different means/stds for train and test!
```

**Why?** If you fit the scaler on test data, you're "leaking" information from the test set into your model. The test set should represent truly unseen data.

---

## Categorical Data

**Categorical data** (also called **qualitative data**) represents discrete categories, groups, or labels. Unlike numerical data, categorical values don't have inherent mathematical meaning—you can't add "red" + "blue" or compute the average of "cat" and "dog."

### Why Categorical Data Matters

Real-world datasets are full of categorical information:
- Customer demographics (gender, country, occupation)
- Product categories (electronics, clothing, food)
- Medical diagnoses (disease types)
- Text labels (spam/not spam, positive/negative sentiment)

Machine learning models need numbers as input, so we must **encode** categorical data into numerical form. However, we must do this carefully to preserve the meaning (or lack thereof) in the categories.

### Types of Categorical Data

#### 1. Nominal Data (No Natural Order)

**Nominal data** consists of categories with **no inherent ranking or order**. One category isn't "greater than" or "better than" another—they're just different.

**Examples:**
- **Color:** Red, Blue, Green, Yellow
  - Is Red > Blue? No! They're just different colors.
- **Country:** USA, UK, Japan, Brazil
  - No natural ordering
- **Product Type:** Electronics, Clothing, Food, Toys
  - Different categories, no hierarchy
- **Animal Species:** Cat, Dog, Bird, Fish
- **Email Domain:** gmail.com, yahoo.com, outlook.com
- **Payment Method:** Credit Card, PayPal, Cash, Bitcoin

**Key characteristic:** Swapping the order doesn't change the meaning.

#### 2. Ordinal Data (Natural Order)

**Ordinal data** consists of categories with a **meaningful order or ranking**. You can say one category is "higher" or "better" than another, but the differences between categories aren't necessarily equal.

**Examples:**
- **Education Level:** High School < Bachelor's < Master's < PhD
  - Clear ordering, but the "distance" between levels isn't uniform
- **T-shirt Size:** XS < S < M < L < XL < XXL
  - Ordered, but XL isn't "twice as large" as M
- **Customer Rating:** Poor < Fair < Good < Very Good < Excellent
  - Ordered, but the difference between Poor and Fair might not equal the difference between Good and Very Good
- **Income Bracket:** <$30k < $30k-$60k < $60k-$100k < >$100k
- **Priority Level:** Low < Medium < High < Critical
- **Grade:** F < D < C < B < A

**Key characteristic:** The order matters, but the intervals between categories may not be equal.

### The Encoding Challenge

Consider this simple example:

```python
# Dataset: Car information
data = {
    'color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'price': [15000, 20000, 25000, 18000, 16000]
}
```

**Problem:** How do we convert 'Red', 'Blue', 'Green' into numbers?

**Bad approach:**
```python
# Assign arbitrary numbers
color_encoding = {'Red': 0, 'Blue': 1, 'Green': 2}
```

**Why this is bad:**
- Implies Red < Blue < Green (false ordering!)
- Implies Blue is "halfway" between Red and Green
- Model might learn: "Green cars are more expensive because Green=2 and Red=0"
- This is mathematically meaningless!

**The solution depends on whether the data is nominal or ordinal.**

---

### Encoding Categorical Data

#### Method 1: Label Encoding

**Label encoding** assigns each unique category a unique integer: 0, 1, 2, 3, ...

**How it works:**

```python
from sklearn.preprocessing import LabelEncoder

# Example: Education levels (ordinal)
education = ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'High School']

encoder = LabelEncoder()
encoded = encoder.fit_transform(education)

print("Original: ", education)
print("Encoded:  ", encoded)
# Original:  ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'High School']
# Encoded:   [2 0 3 4 0 2]

# See the mapping
print(dict(zip(encoder.classes_, range(len(encoder.classes_)))))
# {'Bachelor': 0, 'High School': 2, 'Master': 3, 'PhD': 4}
```

**Manual implementation:**

```python
# Create mapping manually to control the order
education_mapping = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

education = ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor']
encoded = [education_mapping[level] for level in education]
print(encoded)  # [0, 1, 2, 3, 1]
```

**When to use label encoding:**

✅ **Use for ordinal data** where order matters:
- Education levels: High School (0) < Bachelor (1) < Master (2) < PhD (3)
- Ratings: Poor (0) < Fair (1) < Good (2) < Excellent (3)
- Sizes: Small (0) < Medium (1) < Large (2)

❌ **Don't use for nominal data** where order doesn't matter:
- Colors: Red (0), Blue (1), Green (2) ← Implies Red < Blue < Green!
- Countries: USA (0), UK (1), Japan (2) ← Implies USA < UK < Japan!

**Why it works for ordinal data:**

The model can learn: "Higher education level → Higher salary"
- PhD (3) > Master (2) → Model learns PhD earns more
- The numerical relationship matches the real-world relationship

**Why it fails for nominal data:**

The model might learn: "Green (2) cars are more expensive than Red (0) cars"
- But this is just because we arbitrarily assigned 2 to Green!
- The numbers don't represent any real relationship

---

#### Method 2: One-Hot Encoding

**One-hot encoding** creates a new binary column for each category. Each sample gets a 1 in the column for its category and 0 in all others.

**How it works:**

```python
import pandas as pd

# Example: Car colors (nominal)
colors = ['Red', 'Blue', 'Green', 'Red', 'Blue']
df = pd.DataFrame({'color': colors})

# One-hot encode
one_hot = pd.get_dummies(df['color'], prefix='color')

print("Original:")
print(df)
#    color
# 0    Red
# 1   Blue
# 2  Green
# 3    Red
# 4   Blue

print("\nOne-hot encoded:")
print(one_hot)
#    color_Blue  color_Green  color_Red
# 0           0            0          1
# 1           1            0          0
# 2           0            1          0
# 3           0            0          1
# 4           1            0          0
```

**Understanding the transformation:**

Original: `['Red', 'Blue', 'Green', 'Red']`

Becomes three columns:
- `color_Red`: [1, 0, 0, 1] ← 1 where color is Red
- `color_Blue`: [0, 1, 0, 0] ← 1 where color is Blue
- `color_Green`: [0, 0, 1, 0] ← 1 where color is Green

**Each row has exactly one 1** (hence "one-hot").

**Using sklearn:**

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

colors = np.array(['Red', 'Blue', 'Green', 'Red', 'Blue']).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(colors)

print(encoded)
# [[0. 0. 1.]  ← Red
#  [1. 0. 0.]  ← Blue
#  [0. 1. 0.]  ← Green
#  [0. 0. 1.]  ← Red
#  [1. 0. 0.]] ← Blue

print(encoder.categories_)
# [array(['Blue', 'Green', 'Red'], dtype=object)]
```

**Advantages:**

✅ **No false ordering:** Blue isn't "greater than" Red
✅ **Works for any nominal data:** Colors, countries, product types
✅ **Model treats each category independently**

**Disadvantages:**

❌ **High dimensionality:** If you have 1000 categories, you get 1000 columns!
❌ **Sparse data:** Most values are 0
❌ **Memory intensive:** Large datasets with high-cardinality features

**The Dummy Variable Trap:**

When using one-hot encoding, you can actually drop one column without losing information:

```python
# With all columns:
# Red: [1, 0, 0]
# Blue: [0, 1, 0]
# Green: [0, 0, 1]

# Drop one column (e.g., Green):
# Red: [1, 0]
# Blue: [0, 1]
# Green: [0, 0]  ← Still uniquely identified!

# Using pandas
one_hot = pd.get_dummies(df['color'], prefix='color', drop_first=True)
```

**Why drop one?**
- Prevents multicollinearity in linear models
- If `color_Red=0` and `color_Blue=0`, then it must be Green
- Saves memory and computation

**When to use one-hot encoding:**

✅ **Use for nominal data** where order doesn't matter:
- Colors, countries, product categories
- Any categorical feature with no inherent ranking

✅ **Use when you have relatively few categories** (< 50):
- Otherwise, you'll have too many columns

❌ **Don't use for high-cardinality features** (many unique values):
- ZIP codes (thousands of values)
- User IDs (millions of values)
- Use target encoding or embeddings instead

---

#### Method 3: Target Encoding (Mean Encoding)

**Target encoding** replaces each category with the **mean of the target variable** for that category.

**How it works:**

```python
import pandas as pd

# Example: City and house prices
data = {
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC', 'Chicago', 'LA'],
    'price': [500000, 400000, 550000, 300000, 420000, 480000, 320000, 390000]
}
df = pd.DataFrame(data)

# Calculate mean price per city
city_means = df.groupby('city')['price'].mean()
print(city_means)
# city
# Chicago    310000.0
# LA         403333.3
# NYC        510000.0

# Replace city with its mean price
df['city_encoded'] = df['city'].map(city_means)

print(df)
#       city   price  city_encoded
# 0      NYC  500000      510000.0
# 1       LA  400000      403333.3
# 2      NYC  550000      510000.0
# 3  Chicago  300000      310000.0
# 4       LA  420000      403333.3
# 5      NYC  480000      510000.0
# 6  Chicago  320000      310000.0
# 7       LA  390000      403333.3
```

**Intuition:**

Instead of arbitrary numbers or many columns, we encode each category with information about its relationship to the target:
- NYC → 510000 (houses in NYC average $510k)
- LA → 403333 (houses in LA average $403k)
- Chicago → 310000 (houses in Chicago average $310k)

The model can directly learn: "Higher city_encoded → Higher price"

**Advantages:**

✅ **Captures relationship with target:** Encoding contains predictive information
✅ **Handles high cardinality:** 1000 cities → 1 column (not 1000!)
✅ **Often improves model performance**

**Disadvantages:**

❌ **Risk of overfitting:** Especially with small datasets
❌ **Data leakage:** If not done carefully with cross-validation
❌ **Requires target variable:** Can't use for unsupervised learning

**The Overfitting Problem:**

```python
# Small dataset
data = {
    'city': ['NYC', 'LA', 'Chicago'],
    'price': [500000, 400000, 300000]
}

# Target encoding
# NYC → 500000 (based on 1 sample!)
# LA → 400000 (based on 1 sample!)
# Chicago → 300000 (based on 1 sample!)

# Model learns: city_encoded = price
# Perfect training accuracy, but won't generalize!
```

**Solution: Cross-validation target encoding**

```python
from sklearn.model_selection import KFold

def target_encode_cv(X, y, column, n_folds=5):
    """
    Target encode with cross-validation to prevent overfitting
    """
    X_encoded = X.copy()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X):
        # Calculate means on training fold
        means = X.iloc[train_idx].groupby(column)[y.iloc[train_idx]].mean()
        
        # Apply to validation fold
        X_encoded.loc[val_idx, f'{column}_encoded'] = X.loc[val_idx, column].map(means)
    
    return X_encoded
```

**When to use target encoding:**

✅ **High-cardinality categorical features:**
- ZIP codes, user IDs, product IDs
- Anything with hundreds or thousands of unique values

✅ **Tree-based models:**
- Random forests, gradient boosting
- These models handle target encoding well

❌ **Small datasets:**
- High risk of overfitting
- Use one-hot encoding instead

❌ **Linear models without regularization:**
- Can lead to overfitting
- Use with strong regularization

---

### Comparison of Encoding Methods

| Method | Best For | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **Label Encoding** | Ordinal data | Simple, preserves order | Creates false ordering for nominal data |
| **One-Hot Encoding** | Nominal data (low cardinality) | No false ordering, interpretable | High dimensionality, sparse |
| **Target Encoding** | High-cardinality nominal data | Handles many categories, captures target relationship | Overfitting risk, requires target |

**Decision tree:**

```
Is the data ordinal (has natural order)?
├─ Yes → Use Label Encoding
│   Example: Education level, ratings, sizes
│
└─ No (nominal data) → How many unique categories?
    ├─ Few (<50) → Use One-Hot Encoding
    │   Example: Colors, countries, product types
    │
    └─ Many (>50) → Use Target Encoding
        Example: ZIP codes, user IDs, product IDs
```

**Practical example combining all methods:**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Sample dataset
data = {
    'education': ['High School', 'Bachelor', 'Master', 'Bachelor', 'PhD'],  # Ordinal
    'color': ['Red', 'Blue', 'Red', 'Green', 'Blue'],  # Nominal (low cardinality)
    'zip_code': ['10001', '90210', '10001', '60601', '90210'],  # Nominal (high cardinality)
    'age': [25, 30, 35, 28, 45],  # Numerical
    'salary': [50000, 70000, 90000, 65000, 120000]  # Target
}
df = pd.DataFrame(data)

# 1. Label encode ordinal data
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_mapping)

# 2. One-hot encode nominal data (low cardinality)
color_dummies = pd.get_dummies(df['color'], prefix='color', drop_first=True)

# 3. Target encode high-cardinality data
zip_means = df.groupby('zip_code')['salary'].mean()
df['zip_encoded'] = df['zip_code'].map(zip_means)

# 4. Combine everything
df_processed = pd.concat([
    df[['education_encoded', 'age', 'zip_encoded']],
    color_dummies
], axis=1)

print(df_processed)
#    education_encoded  age  zip_encoded  color_Green  color_Red
# 0                  0   25        50000            0          1
# 1                  1   30        95000            0          0
# 2                  2   35        50000            0          1
# 3                  1   28        65000            1          0
# 4                  3   45        95000            0          0
```

---

## Datasets and Splits

A **dataset** is a structured collection of data used to train and evaluate machine learning models. Understanding how to properly organize and split your dataset is crucial for building models that generalize well.

### What is a Dataset?

In machine learning, a dataset consists of:
1. **Features (X):** Input variables used to make predictions
2. **Labels/Targets (y):** Output variables we want to predict

**Structure:**

```
Dataset = Features (X) + Labels (y)

Example: Predicting house prices
┌──────────────┬──────────┬────────┬──────────┐
│  square_feet │ bedrooms │  city  │  price   │  ← Each row = one sample/example
├──────────────┼──────────┼────────┼──────────┤
│     1200     │    2     │  NYC   │  250000  │  ← Sample 1
│     1500     │    3     │   LA   │  300000  │  ← Sample 2
│     1800     │    3     │  NYC   │  350000  │  ← Sample 3
│     2200     │    4     │Chicago │  425000  │  ← Sample 4
└──────────────┴──────────┴────────┴──────────┘
       ↑            ↑         ↑         ↑
   Feature 1    Feature 2  Feature 3  Label (target)
   
   ←─────── X (features) ──────→      y (target)
```

**Terminology:**

| Term | Meaning | Example |
|------|---------|---------|
| **Sample/Example** | One row of data | One house |
| **Feature** | Input variable (column) | Square feet, bedrooms |
| **Label/Target** | Output variable to predict | Price |
| **m** | Number of samples | 1000 houses |
| **n** | Number of features | 3 features |
| **$x^{(i)}$** | The i-th sample | House #5 |
| **$x_j^{(i)}$** | Feature j of sample i | Bedrooms of house #5 |

### Why Split Your Data?

**The fundamental problem:** If you test your model on the same data it was trained on, you can't tell if it learned general patterns or just memorized the training data.

**Analogy:** Imagine studying for an exam:
- **Bad approach:** Study with the actual exam questions, then take the same exam
  - You'll get 100%, but did you really learn?
- **Good approach:** Study with practice problems, then take a different exam
  - Your score reflects true understanding

**In machine learning:**
- **Training set:** Data the model learns from (practice problems)
- **Test set:** Data the model is evaluated on (actual exam)

**Golden rule:** Never let your model see the test data during training!

---

### Train/Test Split

The most basic split: divide your data into two parts.

**Typical split:** 80% training, 20% testing

```
Full Dataset (100%)
├─────────────────────────────────┬──────────┐
│     Training Set (80%)          │ Test (20%)│
│  Model learns patterns here     │ Evaluate  │
└─────────────────────────────────┴───────────┘
```

**Implementation:**

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data: 100 houses
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = np.random.randn(100)      # 100 target values

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"Training samples: {len(X_train)}")  # 80
print(f"Test samples: {len(X_test)}")       # 20

# Train model on training data
model.fit(X_train, y_train)

# Evaluate on test data (unseen during training)
test_score = model.score(X_test, y_test)
print(f"Test accuracy: {test_score:.2f}")
```

**Why random_state?**

```python
# Without random_state: different split each time
X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2)
X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2)
# X_train1 ≠ X_train2 (different samples)

# With random_state: same split every time
X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train1 == X_train2 (identical samples)
```

This ensures reproducibility—you get the same results when you rerun your code.

**Common split ratios:**

| Split | When to Use |
|-------|-------------|
| **80/20** | Most common, good default |
| **70/30** | Smaller datasets (< 1000 samples) |
| **90/10** | Very large datasets (> 100,000 samples) |
| **60/20/20** | When you need train/validation/test |

**How to choose:**

```
Dataset size → Test set size

Small (< 1000):
- Need more training data
- Use 70/30 or 60/40

Medium (1000-100,000):
- Use 80/20 (standard)

Large (> 100,000):
- Test set can be smaller percentage
- Use 90/10 or 95/5
- Even 1% of 1,000,000 = 10,000 test samples (plenty!)
```

---

### Train/Validation/Test Split

For more complex workflows (hyperparameter tuning, model selection), use three splits:

```
Full Dataset (100%)
├──────────────────────────┬─────────────┬──────────┐
│   Training Set (60%)     │ Val (20%)   │ Test (20%)│
│   Model learns here      │ Tune here   │ Final eval│
└──────────────────────────┴─────────────┴───────────┘
```

**Purpose of each set:**

1. **Training set (60%):** Model learns patterns
   - Used to update model parameters (weights, biases)
   - Model sees this data during training

2. **Validation set (20%):** Tune hyperparameters and select models
   - Used to evaluate different model configurations
   - Choose learning rate, regularization strength, model architecture
   - Can be used multiple times during development

3. **Test set (20%):** Final evaluation
   - Used only once at the very end
   - Provides unbiased estimate of model performance
   - Simulates real-world deployment

**Implementation:**

```python
from sklearn.model_selection import train_test_split

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: divide remaining into train (75% of 80% = 60%) and validation (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Training: 60 samples (60%)
# Validation: 20 samples (20%)
# Test: 20 samples (20%)
```

**Typical workflow:**

```python
# 1. Train multiple models
models = [
    LinearRegression(),
    Ridge(alpha=0.1),
    Ridge(alpha=1.0),
    Ridge(alpha=10.0)
]

best_model = None
best_val_score = -float('inf')

# 2. Evaluate each on validation set
for model in models:
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    if val_score > best_val_score:
        best_val_score = val_score
        best_model = model

# 3. Final evaluation on test set (only once!)
test_score = best_model.score(X_test, y_test)
print(f"Final test score: {test_score:.3f}")
```

**Why not just use train/test?**

```
Without validation set:
1. Train model A → Test score: 0.85
2. Train model B → Test score: 0.87
3. Train model C → Test score: 0.89 ← Choose this!
4. Report: "Our model achieves 0.89 accuracy"

Problem: You've "tuned" to the test set!
- You tried multiple models and picked the best on test data
- Test score is now optimistically biased
- Real-world performance will likely be lower

With validation set:
1. Train model A → Val score: 0.85
2. Train model B → Val score: 0.87
3. Train model C → Val score: 0.89 ← Choose this!
4. Final test: 0.86 (unbiased estimate)
```

**The test set is sacred:** Touch it only once, at the very end.

---

### Stratified Splitting

For **classification** problems with imbalanced classes, use **stratified splitting** to maintain class proportions.

**Problem:**

```python
# Imbalanced dataset: 90% class 0, 10% class 1
y = np.array([0]*90 + [1]*10)

# Regular split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Might get unlucky:
# Training: 75 class 0, 5 class 1 (93% vs 7%)
# Test: 15 class 0, 5 class 1 (75% vs 25%)
# ↑ Distributions don't match!
```

**Solution: Stratified split**

```python
# Stratified split maintains class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # ← Maintain class distribution
    random_state=42
)

# Guaranteed:
# Training: 72 class 0, 8 class 1 (90% vs 10%)
# Test: 18 class 0, 2 class 1 (90% vs 10%)
# ↑ Same distribution in both sets!
```

**When to use stratified splitting:**

✅ **Classification problems** (always!)
✅ **Imbalanced datasets** (critical!)
✅ **Small datasets** (ensures each class is represented)

❌ **Regression problems** (stratify parameter doesn't apply)

---

### Cross-Validation

**Cross-validation** uses multiple train/validation splits to get a more robust performance estimate.

**K-Fold Cross-Validation:**

```
Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]

Average the 5 test scores → Final score
```

**Implementation:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")

# Scores: [0.82, 0.85, 0.79, 0.88, 0.84]
# Mean: 0.836
# Std: 0.031
```

**Advantages:**

✅ **More reliable:** Uses all data for both training and testing
✅ **Better for small datasets:** Maximizes use of limited data
✅ **Provides uncertainty estimate:** Standard deviation shows variability

**Disadvantages:**

❌ **Computationally expensive:** Trains model k times
❌ **Slower:** Takes k times longer than single train/test split

**When to use:**

- Small datasets (< 10,000 samples)
- Model selection and hyperparameter tuning
- When you need robust performance estimates

**When not to use:**

- Very large datasets (too slow)
- Deep learning (training is already expensive)
- Time-series data (requires special handling)

---

### Time-Series Splits

For **time-series data**, random splitting breaks temporal order. Use **time-based splits** instead.

**Problem with random split:**

```
Time series: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

Random split:
Train: [Jan, Mar, May, Jun, Aug, Oct, Dec]
Test:  [Feb, Apr, Jul, Sep, Nov]

Problem: Training on future data (Dec) to predict past data (Feb)!
This is "data leakage" and gives unrealistic results.
```

**Solution: Time-based split**

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate

# Visualization:
# Fold 1: [Train][Test]
# Fold 2: [Train][Train][Test]
# Fold 3: [Train][Train][Train][Test]
# Fold 4: [Train][Train][Train][Train][Test]
# Fold 5: [Train][Train][Train][Train][Train][Test]
```

**Key principle:** Always train on past data, test on future data.

---

### Data Leakage

**Data leakage** occurs when information from the test set "leaks" into the training process, leading to overly optimistic performance estimates.

**Common sources of leakage:**

**1. Scaling on full dataset**

```python
# WRONG: Fit scaler on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← Uses test data statistics!
X_train, X_test = train_test_split(X_scaled)

# RIGHT: Fit scaler on training data only
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ← Learn from train only
X_test_scaled = scaler.transform(X_test)        # ← Apply to test
```

**2. Feature selection on full dataset**

```python
# WRONG: Select features using all data
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # ← Uses test labels!
X_train, X_test = train_test_split(X_selected)

# RIGHT: Select features using training data only
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

**3. Target encoding without cross-validation**

```python
# WRONG: Target encode using all data
city_means = df.groupby('city')['price'].mean()  # ← Uses test prices!
df['city_encoded'] = df['city'].map(city_means)

# RIGHT: Target encode with cross-validation
# (See target encoding section above)
```

**4. Using future information in time series**

```python
# WRONG: Create lag features using future data
df['next_day_price'] = df['price'].shift(-1)  # ← Future information!

# RIGHT: Only use past information
df['prev_day_price'] = df['price'].shift(1)  # ← Past information
```

**How to avoid leakage:**

1. **Always split first**, then preprocess
2. **Fit transformers on training data only**
3. **Never use test labels** during training
4. **Be careful with time-series data**
5. **Use pipelines** to ensure correct order

**Using sklearn pipelines to prevent leakage:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Pipeline ensures correct order
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step 1: Scale
    ('model', LinearRegression())      # Step 2: Train
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit pipeline (scaler fits on X_train only)
pipeline.fit(X_train, y_train)

# Predict (scaler transforms X_test using X_train statistics)
predictions = pipeline.predict(X_test)
```

The pipeline ensures the scaler is fit on training data and applied to test data, preventing leakage.

---

## Practical Workflow

A step-by-step guide to preparing data for machine learning, from raw data to model-ready features.

### Complete Data Preparation Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Load data
df = pd.read_csv('data.csv')
print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

# Step 2: Initial exploration
print("\n=== Data Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Target Distribution ===")
print(df['target'].value_counts())

# Step 3: Handle missing values
# Option A: Drop rows with missing values
df_clean = df.dropna()

# Option B: Fill with mean/median/mode
df['age'].fillna(df['age'].median(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Option C: Use imputer (better for pipelines)
# We'll do this in the pipeline below

# Step 4: Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Step 5: Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Step 6: Further categorize categorical columns
# Identify ordinal vs nominal (requires domain knowledge)
ordinal_cols = ['education_level', 'rating']  # Have natural order
nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

# Step 7: Split data FIRST (before any preprocessing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.dtype == 'object' or len(y.unique()) < 20 else None
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Step 8: Create preprocessing pipelines

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values
    ('scaler', StandardScaler())                     # Standardize
])

# Categorical pipeline (nominal)
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encode
])

# Ordinal pipeline
# For ordinal data, we need to specify the order
from sklearn.preprocessing import OrdinalEncoder

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[
        ['High School', 'Bachelor', 'Master', 'PhD'],  # education_level order
        ['Poor', 'Fair', 'Good', 'Excellent']          # rating order
    ]))
])

# Combine all pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, nominal_cols),
        ('ord', ordinal_pipeline, ordinal_cols)
    ],
    remainder='drop'  # Drop any columns not specified
)

# Step 9: Fit preprocessor on training data ONLY
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")

# Step 10: Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)

# Step 11: Evaluate
train_score = model.score(X_train_processed, y_train)
test_score = model.score(X_test_processed, y_test)

print(f"\n=== Model Performance ===")
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
print(f"Gap: {train_score - test_score:.3f}")

# Step 12: Diagnose
if train_score < 0.7 and test_score < 0.7:
    print("\n⚠️  High bias (underfitting)")
    print("Solutions:")
    print("  - Use more complex model")
    print("  - Add more features")
    print("  - Reduce regularization")
elif train_score > 0.95 and test_score < 0.7:
    print("\n⚠️  High variance (overfitting)")
    print("Solutions:")
    print("  - Get more training data")
    print("  - Add regularization")
    print("  - Reduce model complexity")
    print("  - Feature selection")
else:
    print("\n✓ Good fit!")

# Step 13: Create a complete pipeline (optional but recommended)
from sklearn.pipeline import Pipeline

complete_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Now you can fit and predict in one step
complete_pipeline.fit(X_train, y_train)
predictions = complete_pipeline.predict(X_test)

# Save the pipeline
import joblib
joblib.dump(complete_pipeline, 'model_pipeline.pkl')

# Load and use later
loaded_pipeline = joblib.load('model_pipeline.pkl')
new_predictions = loaded_pipeline.predict(X_new)
```

### Checklist for Data Preparation

**Before splitting:**
- [ ] Load data
- [ ] Explore data (info, describe, missing values)
- [ ] Understand target variable
- [ ] Identify column types (numerical, categorical, ordinal)

**Splitting:**
- [ ] Separate features (X) and target (y)
- [ ] Split into train/test (or train/val/test)
- [ ] Use stratify for classification with imbalanced classes
- [ ] Set random_state for reproducibility

**After splitting:**
- [ ] Handle missing values (fit on train, transform both)
- [ ] Scale numerical features (fit on train, transform both)
- [ ] Encode categorical features (fit on train, transform both)
- [ ] Create pipelines to prevent data leakage

**Training:**
- [ ] Train model on training data
- [ ] Evaluate on validation/test data
- [ ] Check for overfitting/underfitting
- [ ] Tune hyperparameters using validation set

**Final evaluation:**
- [ ] Evaluate on test set (only once!)
- [ ] Analyze errors
- [ ] Document performance
- [ ] Save model and preprocessing pipeline

### Common Mistakes to Avoid

**1. Preprocessing before splitting**
```python
# WRONG: Fit scaler on all data
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# RIGHT: Split first, then fit on train only
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**2. Using test set for hyperparameter tuning**
```python
# WRONG: Tune on test set
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ← Using test set!

# RIGHT: Use validation set or cross-validation
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Alpha {alpha}: {scores.mean():.3f}")
```

**3. Label encoding nominal data**
```python
# WRONG: Label encode colors (nominal)
colors = ['Red', 'Blue', 'Green']
encoded = [0, 1, 2]  # Implies Red < Blue < Green!

# RIGHT: One-hot encode
one_hot = pd.get_dummies(colors)
```

**4. Forgetting to handle missing values**
```python
# WRONG: Ignore missing values
model.fit(X_train, y_train)  # ← Will error if X_train has NaN!

# RIGHT: Handle missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
```

**5. Not using pipelines**
```python
# WRONG: Manual preprocessing (error-prone)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# RIGHT: Use pipeline (automatic, prevents errors)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## Summary

### Key Takeaways

**1. Data Types:**
- **Numerical:** Continuous (any value) or discrete (specific values)
  - Scale appropriately: normalization or standardization
- **Categorical:** Nominal (no order) or ordinal (has order)
  - Encode appropriately: one-hot, label, or target encoding

**2. Datasets:**
- Always split: train/test or train/val/test
- Split BEFORE preprocessing
- Never test on training data
- Use stratified splitting for classification

**3. Generalization:**
- Goal: Perform well on unseen data
- Measure: Test set performance
- Indicator: Small gap between train and test

**4. Overfitting:**
- Problem: Memorizes training data
- Signs: Low train error, high test error
- Solutions: More data, regularization, simpler model

**5. Underfitting:**
- Problem: Too simple to capture patterns
- Signs: High train error, high test error
- Solutions: More complex model, more features

**6. Bias-Variance:**
- Bias: Error from simplistic assumptions
- Variance: Error from sensitivity to data
- Goal: Balance both

### Decision Tree

```
Start with your data
│
├─ What type of data?
│  ├─ Numerical → Scale it (standardization or normalization)
│  └─ Categorical → Encode it
│     ├─ Ordinal → Label encoding
│     ├─ Nominal (few categories) → One-hot encoding
│     └─ Nominal (many categories) → Target encoding
│
├─ Split data (train/test or train/val/test)
│  └─ Classification with imbalanced classes? → Use stratify
│
├─ Preprocess (fit on train, transform both)
│  ├─ Handle missing values
│  ├─ Scale numerical features
│  └─ Encode categorical features
│
├─ Train model
│
├─ Evaluate
│  ├─ High train error, high test error → Underfitting
│  │  └─ Increase complexity, add features
│  ├─ Low train error, high test error → Overfitting
│  │  └─ Regularization, more data, simpler model
│  └─ Low train error, low test error → Good fit!
│     └─ Deploy
│
└─ Use learning curves to confirm diagnosis
```

---

## Next Steps

- [Generalization & Overfitting](./generalization-overfitting.md) — understand model generalization
- [Gradient Descent](../gradient-descent/notes.md) — how models learn
- [Neural Networks](../../advanced%20ml%20conceptss/neural-networks.md) — modern ML techniques

---

**Remember:** Understanding your data is the foundation of successful machine learning. Take time to explore, visualize, and prepare your data properly. A well-prepared dataset with a simple model often outperforms a complex model with poorly prepared data.
