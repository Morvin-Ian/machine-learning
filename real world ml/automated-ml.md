# Automated ML (AutoML)

Automate model selection and hyperparameter tuning.

---

## What is AutoML?

AutoML automates parts of the machine learning workflow:
- Model selection
- Hyperparameter tuning
- Sometimes feature engineering or neural architecture search

**Goal:** Make ML accessible and faster to iterate.

---

## Core Components

| Component | Description |
|-----------|-------------|
| **Search space** | Set of models, preprocessing steps, and hyperparameters to try |
| **Search strategy** | How to explore the space (random, Bayesian, evolutionary) |
| **Evaluation strategy** | How candidates are measured (cross-validation, holdout sets) |

---

## Types of Automation

| Type | Description |
|------|-------------|
| **Hyperparameter tuning** | Find good settings for a chosen model |
| **Pipeline search** | Compose preprocessing + model (TPOT, Auto-sklearn) |
| **Neural Architecture Search** | Design network topologies automatically |
| **End-to-end AutoML** | Feature preprocessing, model selection, ensembling |

---

## Pros & Cons

| Pros | Cons |
|------|------|
| Faster prototyping | Can be compute-heavy |
| Quick baselines | May yield complex models |
| Good for non-experts | Still needs human oversight |

---

## When to Use

**Use AutoML for:**
- Quick baselines and proof-of-concepts
- Standard feature engineering tasks

**Avoid for:**
- Highly custom architectures
- When interpretability is required
- Strict constraints on model complexity

---

## Practical Tips

1. Limit search space to reasonable model families
2. Use budgeted runs (max time or trials)
3. Validate final model with domain-specific tests
4. Don't skip human review for fairness and quality

---

## Example: Auto-sklearn

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,  # seconds
    per_run_time_limit=30,
    ensemble_size=50,
)
automl.fit(X_train, y_train)
print(automl.leaderboard())
print('Test accuracy', automl.score(X_test, y_test))
```

---

## Tooling

| Tool | Type |
|------|------|
| Auto-sklearn | Pipeline search |
| TPOT | Genetic programming |
| H2O AutoML | End-to-end |
| AutoGluon | End-to-end |
| Google Cloud AutoML | Cloud-based |
| AWS SageMaker Autopilot | Cloud-based |

---

## Next Steps

- [Fairness in ML](./fairness.md) — build responsible systems
- [Production ML](./production.md) — deploy models to production
- [Back to Course Start](../README.md) — review the full learning path
