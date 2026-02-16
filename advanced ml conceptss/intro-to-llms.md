```markdown
# Intro to Large Language Models (LLMs)

## What is an LLM?
Large Language Models are neural models (usually transformer-based) trained on large-scale text corpora to model language. They predict tokens given context and produce rich contextual representations used for generation, understanding, and downstream tasks.

## Core Architecture: Transformer
- **Self-attention:** Each token attends to others using queries, keys, and values. Attention weights are computed by scaled dot-product and allow modeling of long-range dependencies efficiently.
- **Multi-head attention:** Multiple parallel attention heads let the model capture different relationships.
- **Feed-forward layers:** Apply position-wise nonlinear transforms.
- **Residual connections & normalization:** Improve gradient flow and stability.

## Training Paradigms
- **Causal / Autoregressive (e.g., GPT):** Model p(x_t | x_{<t}) and used for free-form generation.
- **Masked / Denoising (e.g., BERT):** Mask tokens and predict them; good for encoding and classification.
- **Seq2Seq pretraining (e.g., T5):** Encoder-decoder models trained on denoising/objective tasks allowing flexible generation.

## Tokenization
- Subword tokenizers (Byte-Pair Encoding, WordPiece, SentencePiece) split text into manageable tokens balancing vocabulary size vs sequence length.
- Tokenization affects model behavior; always use the tokenizer associated with a pretrained model.

## Capabilities & Use Cases
- Text generation, summarization, translation, question answering, code generation, assistants, semantic search (via embeddings), and more.

## Fine-tuning vs In-Context Learning
- **Fine-tuning:** Update model weights on a labeled dataset for a specific task; resource-intensive but often yields best task performance.
- **In-Context Learning / Prompting:** Provide examples in the prompt to guide behavior without weight updates. Powerful for few-shot tasks but has limits (context window size, brittleness).

## Scaling & Emergent Behavior
- Larger models and larger pretraining datasets often yield qualitatively new capabilities (emergent behaviors). Scaling improves few-shot performance and robustness but increases compute and alignment challenges.

## Practical Engineering Notes
- **Context window:** LLMs have finite context lengths; use retrieval-augmented generation (RAG) to provide external knowledge beyond the window.
- **Safety & Bias:** LLMs reflect training data and can produce biased, toxic, or incorrect outputs. Apply filtering, human review, and alignment techniques.
- **Latency & Cost:** Larger models have higher inference costs; consider distillation, quantization (8-bit/4-bit), or smaller specialized models.
- **Evaluation:** Use human evaluation, task-specific metrics, and red-teaming for safety-critical systems.

## Deployment Patterns
- **Server-side inference:** Centralized model serving (GPU/TPU) for high-quality responses.
- **Local / Edge:** Tiny or quantized models for on-device privacy-sensitive use.
- **Hybrid (RAG):** Retrieve relevant documents, then condition generation on retrieved context for up-to-date or factual outputs.

## Limitations
- **Hallucinations:** Confident but incorrect outputs; mitigations include grounding with retrieval and answer verification.
- **Context length limits:** Hard limits require chunking or retrieval strategies.
- **Data privacy:** Pretrained models may memorize sensitive data if present in training corpora.

## Next Steps for Learning
- Study the transformer paper (Vaswani et al., 2017).
- Experiment with small transformer implementations (PyTorch, Hugging Face Transformers).
- Try prompt engineering, RAG, and fine-tuning small models to gain practical experience.

```
