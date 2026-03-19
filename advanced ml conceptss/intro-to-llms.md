# Introduction to LLMs

Large language models and the transformer architecture.

---

## What is an LLM?

Large Language Models are neural models (usually transformer-based) trained on large-scale text corpora. They predict tokens given context and produce rich representations for generation, understanding, and downstream tasks.

---

## Core Architecture: Transformer

### Self-Attention

Each token attends to others using queries, keys, and values. Attention weights are computed by scaled dot-product.

```python
import torch

def attention(Q, K, V):
    # Q, K, V: [seq_len, d]
    scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(Q.size(-1)))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Multi-head attention** | Parallel attention blocks with separate Q/K/V projections |
| **Feed-forward layers** | Position-wise MLPs (ReLU or GELU) |
| **Residual connections** | Improves gradient flow and stabilizes training |
| **Layer normalization** | Normalizes activations |

---

## Training Paradigms

| Paradigm | Models | Use Case |
|----------|--------|----------|
| **Causal / Autoregressive** | GPT | Free-form generation |
| **Masked / Denoising** | BERT | Encoding, classification |
| **Seq2Seq** | T5, BART | Translation, summarization |

---

## Tokenization

Subword tokenizers (BPE, WordPiece, SentencePiece) split text into tokens balancing vocabulary size vs sequence length.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, how are you?"
tokens = tokenizer(text)
print(tokenizer.convert_ids_to_tokens(tokens['input_ids']))
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
```

---

## Fine-tuning vs In-Context Learning

### Fine-tuning
Update model weights on a labeled dataset for a specific task.

```python
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments

model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
trainer = Trainer(
    model=model, 
    args=TrainingArguments(output_dir='./out', num_train_epochs=3),
    train_dataset=train_ds
)
trainer.train()
```

### In-Context Learning
Provide examples in the prompt without weight updates.

```python
prompt = """Translate English to French:
English: Hello
French: Bonjour
English: How are you?
French:"""
```

---

## Practical Considerations

| Aspect | Consideration |
|--------|---------------|
| **Context window** | Finite length; use RAG for longer contexts |
| **Hallucinations** | Confident but incorrect outputs; use retrieval grounding |
| **Latency/Cost** | Larger models are slower; consider quantization |
| **Safety/Bias** | Reflect training data; apply filtering and alignment |

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| Transformer | Self-attention + feed-forward layers |
| Attention | Q/K/V projections for contextual understanding |
| Fine-tuning | Update weights for specific tasks |
| In-context learning | Examples in prompt without training |

---

## Next Steps

**Advanced Concepts Complete!**

- [Production ML](../real%20world%20ml/production.md) — deploy ML systems
- [AutoML](../real%20world%20ml/automated-ml.md) — automate ML workflows
- [Fairness](../real%20world%20ml/fairness.md) — build responsible ML
