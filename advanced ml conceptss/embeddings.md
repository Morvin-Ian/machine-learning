# Embeddings

Dense vector representations that capture semantic relationships.

---

## What Are Embeddings?

An embedding is a dense, low-dimensional vector representation of discrete objects (words, tokens, users, items) that captures semantic relationships. Similar objects have vectors close together under a chosen metric (cosine similarity or Euclidean distance).

---

## Why Embeddings?

- Convert categorical, textual, or high-dimensional inputs into continuous vectors usable by ML models
- Capture semantic relationships (e.g., `vec('king') - vec('man') + vec('woman') ≈ vec('queen')`)

```python
from gensim.downloader import load
model = load('word2vec-google-news-300')
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # [('queen', 0.7118)]
```

---

## Types of Embeddings

| Type | Description | Examples |
|------|-------------|----------|
| **Static** | Single vector per token | Word2Vec, GloVe |
| **Contextual** | Varies by context | BERT, GPT |
| **Task-specific** | Trained end-to-end | Recommendation embeddings |

---

## Distance & Similarity

- **Cosine similarity**: Common for semantic similarity; invariant to vector length
- **Euclidean distance**: When absolute magnitudes matter
- **Dot product**: Often used inside models (e.g., attention scores)

---

## Practical Uses

### Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ['cat video', 'machine learning tutorial', 'buy cheap shoes']
doc_embs = model.encode(docs)

query = 'how to train a neural net'
q_emb = model.encode([query])[0]
sims = np.dot(doc_embs, q_emb) / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb))
print('closest doc:', docs[np.argmax(sims)])
```

### Clustering & Visualization

Reduce embedding dimensionality (PCA/UMAP) and visualize with scatter plots to see semantic groupings.

### Recommendation

User/item embeddings for collaborative filtering. Multiply user and item vectors to predict ratings.

---

## Engineering Considerations

| Aspect | Consideration |
|--------|---------------|
| **Dimensionality** | 50-300 for classical, 512-4096 for modern |
| **Indexing** | Use ANN (HNSW, FAISS) for large collections |
| **Normalization** | Normalize vectors before cosine search |
| **Updates** | Support incremental retraining for dynamic domains |

---

## Quick Reference

| Method | Use Case |
|--------|----------|
| Word2Vec | Static word embeddings |
| BERT | Contextual embeddings |
| Sentence Transformers | Semantic search |
| CLIP | Image-text embeddings |

---

## Next Steps

- [Intro to LLMs](./intro-to-llms.md) — large language models
- [Neural Networks](./neural-networks.md) — deep learning fundamentals
- [Production ML](../real%20world%20ml/production.md) — deploy models to production
