# Classification
Classification is the task of predicting which of a set of classes (categories) an example belongs to. In binary classification, there are two classes (e.g., spam/not spam, positive/negative). In multi-class classification, there are more than two classes.

## Classification Thresholds
Logistic regression models output **probability scores** between 0 and 1, not direct class labels. To convert these probabilities into actual predictions, we must choose a **classification threshold**. This threshold divides the probability scale into two regions: predictions above the threshold are classified as the positive class, and predictions below are classified as the negative class.

**Example:** An email classifier outputs:
- Email A: 0.99 probability (99% spam)
- Email B: 0.51 probability (51% spam)

With threshold = 0.5: Both emails classified as spam
With threshold = 0.95: Only Email A classified as spam

**Key insight:** The choice of threshold depends on your business needs. A lower threshold catches more spam but increases false alarms. A higher threshold is more conservative and misses some spam.


## Confusion Matrix
The confusion matrix summarizes the performance of a binary classifier by showing all possible prediction outcomes compared to actual ground truth labels. For any binary classification, there are four possible outcomes:

1. **True Positive (TP):** Model predicted positive, and the actual label is positive ✓
2. **True Negative (TN):** Model predicted negative, and the actual label is negative ✓
3. **False Positive (FP):** Model predicted positive, but the actual label is negative ✗ (Type I error)
4. **False Negative (FN):** Model predicted negative, but the actual label is positive ✗ (Type II error)

**Visualization:**
```
                    Predicted Positive | Predicted Negative
Actual Positive          TP             |        FN
Actual Negative          FP             |        TN
```

This matrix is fundamental for calculating many evaluation metrics.

## Effect of Thresholds on True and False Positives/Negatives
The choice of classification threshold directly affects the counts in the confusion matrix:

- **Lower threshold (e.g., 0.3):** Predicts more samples as positive → Higher TP but also higher FP (more false alarms)
- **Higher threshold (e.g., 0.9):** Predicts fewer samples as positive → Lower FP but also lower TP (more missed positives)
- **Default threshold (0.5):** A balanced midpoint

The optimal threshold depends on the cost of different errors in your specific problem domain. For example:
- **Medical diagnosis:** You might tolerate more false positives (unnecessary tests) to avoid false negatives (missed diseases)
- **Spam detection:** You might tolerate some spam getting through to avoid false positives (blocking legitimate emails)

## Classification: Accuracy, recall, precision, and related metrics
### Accuracy
It is the proportion of all classifications that were correct whether positive or negative.

Accuracy = correct classifications/total 
        = (TP + TN) / (TP + TN + FP + FN)

However, when the dataset is imbalanced, or where one kind of mistake (FN or FP) is more costly than the other, which is the case in most real-world applications, it's better to optimize for one of the other metrics instead.


### Recall (True Positive Rate)
Recall answers: **"Of all the actual positives, how many did we correctly identify?"**

$$\text{Recall} = \frac{\text{TP}}{\text{TP + FN}}$$

- **High recall:** The model catches most positive cases (few false negatives)
- **Use when:** Missing positives is costly (e.g., disease diagnosis, fraud detection)
- **Example:** A cancer detection model with 95% recall catches 95 out of 100 actual cancers 

### False Positive Rate
False Positive Rate answers: **"Of all the actual negatives, how many did we incorrectly classify as positive?"**

$$\text{FPR} = \frac{\text{FP}}{\text{FP + TN}}$$

- **Low FPR:** The model has few false alarms (few incorrect positive predictions)
- **Use when:** False positives are costly (e.g., spam filters blocking legitimate emails, false criminal accusations)
- **Note:** FPR is complementary to specificity. Specificity = 1 - FPR

### Precision
Precision answers: **"Of all the samples we predicted as positive, how many were actually positive?"**

$$\text{Precision} = \frac{\text{TP}}{\text{TP + FP}}$$

- **High precision:** When the model predicts positive, it's usually correct (few false positives)
- **Use when:** False positives are costly (e.g., recommending a product you're confident about)
- **Example:** A spam filter with 99% precision means 99% of emails flagged as spam are actually spam

### Accuracy
Accuracy answers: **"What proportion of all predictions were correct?"**

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}$$

- Measures overall correctness across both positive and negative predictions
- **Limitation:** Misleading with imbalanced datasets. For example, if 99% of emails are not spam, a naive model that always predicts "not spam" achieves 99% accuracy but is useless!
- **Better for:** Balanced datasets where all classes are equally important

### Precision vs. Recall Trade-off
These metrics are inversely related:
- **Lowering the threshold:** Increases Recall (catches more positives) but decreases Precision (more false alarms)
- **Raising the threshold:** Increases Precision (fewer false alarms) but decreases Recall (misses more positives)

You must choose based on your problem: Do you care more about finding positives (Recall) or being confident when you predict positive (Precision)?

### ROC and AUC
**ROC (Receiver Operating Characteristic) Curve** visualizes the trade-off between Recall and False Positive Rate across all possible thresholds.

- **X-axis:** False Positive Rate (1 - Specificity)
- **Y-axis:** True Positive Rate (Recall)
- **Diagonal line:** Random classifier (no predictive power)
- **Curve above diagonal:** Good classifier

**AUC (Area Under the Curve)** is a single metric summarizing ROC performance:
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.5:** Random classifier (same as guessing)
- **AUC = 0.0:** Worst classifier (inverse predictions)

**When to use:** AUC is excellent for comparing models and handling imbalanced datasets. It's threshold-independent and shows overall model performance across all thresholds.

### Prediction Bias
Prediction bias measures whether your model systematically predicts higher or lower values than actual outcomes.

$$\text{Prediction Bias} = \frac{\sum \text{Predicted Values}}{\sum \text{Actual Values}}$$

- **Bias ≈ 1.0:** Model predictions match actual distribution (good)
- **Bias > 1.0:** Model predicts too high on average (overestimation)
- **Bias < 1.0:** Model predicts too low on average (underestimation)

**Example:** If your model predicts an average spam probability of 0.8 but the actual spam rate is 0.2, you have a bias of 4.0 (significant overestimation).

**How to fix:** If bias exists, you can adjust the decision threshold, retrain the model with class weighting, or collect more balanced training data.

## F1-Score

The **F1-Score** is the harmonic mean of precision and recall, providing a single metric that balances both:

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why harmonic mean?** It penalizes extreme imbalances. If either precision or recall is very low, F1 will also be low.

**Example calculation:**
- Precision = 0.8, Recall = 0.6
- F1 = 2 × (0.8 × 0.6) / (0.8 + 0.6) = 2 × 0.48 / 1.4 = **0.686**

**When to use F1:**
- When you need a single metric that balances precision and recall
- When dealing with imbalanced datasets
- When false positives and false negatives are equally costly

---

## Multi-Class Classification

When dealing with more than two classes, we extend binary classification techniques:

### One-vs-All (One-vs-Rest)

Train **K separate binary classifiers** (one for each class):
- Class 1 vs All Others
- Class 2 vs All Others
- ... etc.

**Prediction:** Choose the class with highest confidence score.

```
Input: [features]
    │
    ├── Classifier 1: Is it Class A? → Score: 0.7
    ├── Classifier 2: Is it Class B? → Score: 0.2
    └── Classifier 3: Is it Class C? → Score: 0.5
    
    Prediction: Class A (highest score)
```

### One-vs-One

Train a classifier for **every pair of classes**: $\frac{K(K-1)}{2}$ classifiers.

**Prediction:** Voting—each classifier votes for one class, majority wins.

### Multi-Class Metrics

For multi-class problems, metrics can be calculated in different ways:

| Averaging | Description |
|-----------|-------------|
| **Macro** | Average metric across all classes (treats all classes equally) |
| **Weighted** | Average weighted by class frequency |
| **Micro** | Calculate globally (aggregate TPs, FPs, FNs from all classes) |

```python
from sklearn.metrics import precision_recall_fscore_support

# Macro average (all classes equal weight)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

# Weighted average (by class frequency)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
```

### Confusion Matrix for Multi-Class

```
              Predicted
            A    B    C
Actual  A [ 50   2    3 ]
        B [  1  45    4 ]
        C [  2   3   40 ]

Diagonal = Correct predictions
Off-diagonal = Errors (shows which classes get confused)
```

---

## Worked Example: Calculating All Metrics

**Given confusion matrix:**
```
                Predicted
              Positive  Negative
Actual  Positive    80       20    (TP=80, FN=20)
        Negative    10       90    (FP=10, TN=90)
```

**Calculations:**

| Metric | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| **Accuracy** | (TP+TN)/(All) | (80+90)/200 | **0.85** |
| **Precision** | TP/(TP+FP) | 80/(80+10) | **0.89** |
| **Recall** | TP/(TP+FN) | 80/(80+20) | **0.80** |
| **F1-Score** | 2×P×R/(P+R) | 2×0.89×0.80/1.69 | **0.84** |
| **FPR** | FP/(FP+TN) | 10/(10+90) | **0.10** |

---

## Quick Reference

| Metric | What It Measures | Optimize When... |
|--------|------------------|------------------|
| **Accuracy** | Overall correctness | Classes are balanced |
| **Precision** | Quality of positive predictions | FP is costly (spam filter) |
| **Recall** | Coverage of positives | FN is costly (disease detection) |
| **F1-Score** | Balance of P and R | Need single balanced metric |
| **ROC-AUC** | Overall model quality | Comparing models |