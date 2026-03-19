# Fairness in ML

A practical introduction to building responsible ML systems.

---

## Why Fairness Matters

- Models can unintentionally treat groups differently (by gender, race, age, etc.) because of biased data, proxies, or modeling choices
- Fairness is important for ethics, legal compliance, and user trust

---

## Sources of Bias

| Bias Type | Description |
|-----------|-------------|
| **Historical bias** | Historical data reflects unfair practices |
| **Sampling bias** | Training data isn't representative of the population |
| **Label bias** | Labels reflect human or process bias |
| **Proxy features** | Innocuous features correlate with protected attributes |

---

## Fairness Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Demographic Parity** | Positive prediction rate should be equal across groups |
| **Equalized Odds** | True positive and false positive rates should be similar across groups |
| **Predictive Parity** | Positive predictive value should be similar across groups |

---

## Trade-offs

- You often cannot optimize all metrics at once; improving one can worsen another
- Fairness may trade off with overall accuracy — quantify and make decisions explicit

---

## Mitigation Strategies

| Stage | Technique |
|-------|-----------|
| **Pre-processing** | Modify data (reweighting, resampling) to reduce bias before training |
| **In-processing** | Change learning algorithm to include fairness constraints or regularizers |
| **Post-processing** | Adjust model outputs (thresholds or calibration) to improve fairness |

---

## Practical Example

```python
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
import numpy as np

# Synthetic data: feature X, label y, sensitive attribute s (0/1)
X = np.random.randn(1000, 2)
y = (X[:,0] + X[:,1] > 0).astype(int)
s = (np.random.rand(1000) < 0.5).astype(int)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.2, random_state=42)

base_clf = LogisticRegression(solver='liblinear')
constraint = DemographicParity()
mitigator = ExponentiatedGradient(base_clf, constraint)
mitigator.fit(X_train, y_train, sensitive_features=s_train)

preds = mitigator.predict(X_test)

# Evaluate accuracy and group rates
from sklearn.metrics import accuracy_score
print('Overall accuracy', accuracy_score(y_test, preds))
for group in [0, 1]:
    idx = s_test == group
    print(f'Group {group} positive rate', preds[idx].mean())
```

---

## Practical Steps for Teams

1. Identify protected groups relevant to your product and legal/regulatory context
2. Define measurable fairness goals and metrics
3. Test models on stratified holdout sets and compute chosen metrics
4. If issues arise, try a mitigation technique and re-evaluate for unintended harms
5. Monitor fairness over time — data drift can reintroduce bias

---

## Tools

- **AIF360** — IBM's comprehensive toolkit
- **Fairlearn** — Microsoft's fairness library
- **What-If Tool** — Google's visualization tool
- **Cloud provider tools** — Built-in fairness monitoring

---

## Ethics & Governance

- Keep human oversight, document choices, and maintain clear audit logs
- Engage stakeholders and affected communities when possible

---

## Next Steps

**Course Complete!** You've learned:

- [Supervised Learning](../../01-supervised-learning/) — regression and classification
- [Unsupervised Learning](../../02-unsupervised-learning/) — clustering and anomaly detection
- [Deep Dives](../../03-deep-dives/) — math and fundamentals
- [Advanced Concepts](../../advanced%20ml%20conceptss/) — neural networks and LLMs
- [Real World ML](./README.md) — production, AutoML, and fairness

Keep building and learning!
