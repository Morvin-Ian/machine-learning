# Production ML Systems

A practical guide to designing, shipping, and operating ML systems.

---

## Overview

Production ML is about reliably running models so they deliver value continuously. Unlike research or prototyping, production systems must survive changing data, hardware failures, and real-world usage patterns.

**Key goals:**
- **Correctness** — predictions match intent
- **Reliability** — uptime and graceful degradation
- **Observability** — knowing what the model is doing
- **Cost control** — reasonable compute/storage spend
- **Repeatability** — reproduce exactly how a model was built

> Note: Operational ML sits at the intersection of software engineering, data engineering, and DevOps.

---

## Model Serving Modes

| Mode | Description | Example |
|------|-------------|---------|
| **Batch** | Run predictions on large datasets (cheap, high-latency) | Nightly credit scoring |
| **Online** | Serve individual requests with low latency | Recommendation API |
| **Streaming** | Continuous scoring for event-driven systems | Fraud detection |

---

## Data Pipelines & Feature Engineering

- Separate training pipelines (offline) from inference pipelines (online)
- Use a **feature store** to centralize transformations
- Document data lineage for traceability
- Prevent train/serve skew (features must compute identically in both environments)

---

## Versioning & Reproducibility

- Store code, data snapshots, features, model artifacts, and hyperparameters
- Use ML metadata tracking (MLflow, DVC, MLMD)
- Hash or fingerprint datasets to detect distribution changes
- Treat models as first-class artifacts in your CI system

---

## CI/CD for Models

Automate the end-to-end workflow:
1. Run unit tests on transformation code
2. Validate new training data
3. Retrain the model
4. Run evaluation metrics
5. Package and deploy to staging/production

**Deployment strategies:**
- **Canary releases** — gradual rollout
- **Shadow mode** — new model scores live traffic without acting

---

## Monitoring & Observability

**Key metrics to track:**
- Prediction distributions
- Feature statistics
- Input data drift
- Model latency
- Request/response errors
- Business KPIs

**Set alerts** when values cross thresholds (e.g., 10% drift in key feature, 95th-percentile latency >200ms).

---

## Testing

| Test Type | Purpose |
|----------|---------|
| **Unit tests** | Transformation functions, data validation |
| **Integration tests** | Full pipeline runs end-to-end |
| **Regression tests** | New models vs baseline |
| **Shadow testing** | New model alongside production |

---

## Security & Privacy

- Protect PII by anonymizing or tokenizing
- Encrypt data in transit (TLS) and at rest
- Comply with regulations (GDPR, CCPA)
- Monitor for model extraction attacks
- Rate-limit APIs

---

## Cost & Scaling

- Use autoscaling for online services
- Profile compute during development
- Monitor storage costs for datasets and artifacts
- Compress or archive stale data

---

## Quick Checklist

1. Freeze a training dataset snapshot for reproducibility
2. Package model artifact with preprocessing code and environment
3. Add tests and run them as part of CI
4. Deploy to staging; run canaries or shadow traffic
5. Monitor metrics and set alerts before full rollout

---

## Next Steps

- [AutoML](./automated-ml.md) — automate ML workflows
- [Fairness in ML](./fairness.md) — build responsible systems
- [Back to Course Start](../README.md) — review the full learning path
