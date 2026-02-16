```markdown
# Embeddings — What They Are and How to Use Them

## Definition
An embedding is a dense, low-dimensional vector representation of a discrete object (word, token, image patch, user, item) that captures semantic relationships by geometry: similar objects have vectors close under a chosen metric (commonly cosine similarity or Euclidean distance).

## Why Embeddings?
- Convert categorical, textual, or high-dimensional inputs into continuous vectors usable by ML models.
- Capture semantic relationships (e.g., `vec('king') - vec('man') + vec('woman') ≈ vec('queen')`).

## How Embeddings Are Learned
- **Supervised learning:** Train embeddings together with a downstream task (classification, recommendation).
- **Self-supervised / contrastive:** Learn to pull positive pairs together and push negatives apart (e.g., SimCLR, triplet loss).
- **Language-model-based:** Contextual embeddings come from internal layers of language models; static embeddings (Word2Vec, GloVe) are trained with co-occurrence or predictive objectives.

## Embedding Types
- **Static embeddings:** Single vector per token (Word2Vec, GloVe).
- **Contextual embeddings:** Vary by context (BERT, GPT); capture polysemy.
- **Learned task-specific embeddings:** Trained end-to-end for recommendation or classification.

## Distance & Similarity
- **Cosine similarity:** Common for measuring semantic similarity; invariant to vector length.
- **Euclidean distance:** Useful when absolute magnitudes matter.
- **Dot product:** Often used inside models (e.g., attention scores). Normalize if necessary.

## Practical Uses
- **Semantic search / retrieval:** Encode queries and documents, retrieve by nearest neighbors (ANN indexes like FAISS, Annoy, Milvus).
- **Clustering & visualization:** Reduce embedding dimensionality (PCA/UMAP) for inspection.
- **Recommendation:** User/item embeddings for collaborative filtering (matrix factorization, deep learning).
- **Transfer learning:** Use pretrained embeddings as features for downstream tasks.

## Engineering Considerations
- **Dimensionality:** 50–300 for classical embeddings; 512–4096 for modern contextual representations. Balance quality vs storage/latency.
- **Indexing:** For large collections, use approximate nearest neighbor (ANN) search to scale (HNSW, IVF, PQ).
- **Normalization:** Normalize vectors (unit length) before cosine search to speed up comparisons.
- **Updating embeddings:** For dynamic domains, support incremental retraining or hybrid approaches (cold-start heuristics).

## Common Pitfalls
- Using raw token vectors from different models without alignment causes incompatibility.
- Ignoring distributional shifts — embeddings trained on one domain may not transfer well to another.
- Storing massive embedding corpora without compression or ANN can be costly.

## Quick Examples
- Word2Vec / Skip-gram: Predict context words from a center word; produces static embeddings.
- Contrastive loss: Learn `sim(a, b)` high for positives and low for negatives using InfoNCE or triplet losses.

```
