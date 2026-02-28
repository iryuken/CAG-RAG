# Velox — Lightweight RAG + CAG System

A cost-efficient Q&A pipeline combining **Elasticsearch hybrid search** (RAG),
**SQLite semantic caching** (CAG), and a **quantized 3B local LLM** — all
running on an 8 GB RAM, CPU-only machine.

> Built for the **Elastic Blogathon 2026**.

---

## Architecture

```
User Query
   │
   ▼
┌──────────────────────┐
│  CAG (SQLite Cache)  │──── HIT ──► Return cached answer (~0.05s)
└──────────┬───────────┘
           │ MISS
           ▼
┌──────────────────────┐
│  RAG (Elasticsearch) │  BM25 + kNN hybrid search → Top-4 chunks
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3B Local LLM        │  Context-grounded generation (max 256 tokens)
│  (4-bit quantized)   │
└──────────┬───────────┘
           │
           ▼
    Store in Cache → Return answer
```

---

## Elastic Stack Implementation

In this system, Elasticsearch 8.x serves as the high-performance retrieval backbone. It goes beyond simple keyword matching by implementing a **Hybrid Search** strategy:

1.  **Dense Vector Storage**: Documents are embedded into 384-dimensional vectors using `all-MiniLM-L6-v2` and stored in a `dense_vector` field using the **HNSW (Hierarchical Navigable Small World)** algorithm for efficient similarity search.
2.  **BM25 Keyword Search**: Traditional full-text search ensures that specific terms (names, IDs, rare keywords) are not missed by the vector search.
3.  **Reciprocal Rank Fusion (RRF)**: The results from the kNN (vector) and BM25 legs are merged using RRF. This fusion technique provides a unified ranking that consistently outperforms either method alone, ensuring the LLM receives the most relevant context chunks.
4.  **Schema-on-Write**: The index mapping is explicitly defined in `retriever.py` to optimize for `cosine` similarity, which is ideal for semantic matching.

---

## Project Structure

```
velox/
├── config.py        # Configuration with env-var overrides
├── utils.py         # Cosine similarity, timer, coloured output
├── cache.py         # CAG — SQLite semantic cache layer
├── retriever.py     # RAG — Elasticsearch hybrid retriever
├── llm.py           # Local 3B LLM (Phi-3 / LLaMA auto-detect)
├── pipeline.py      # Orchestrator wiring CAG → RAG → LLM
├── main.py          # CLI entry point & demo runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Prerequisites

| Requirement         | Notes                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Python 3.10+        |                                                                       |
| Elasticsearch 8.x   | See Docker command below                                              |
| RAM                 | 8 GB minimum (CPU); 6 GB VRAM for GPU 4-bit                          |

### 2. Start Elasticsearch (Docker)

```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.13.0
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only machines**: Remove `bitsandbytes` from requirements.txt —
> quantization is skipped automatically when no CUDA is detected.

### 4. Choose your 3B model

| Model                                | Size  | Access            |
|--------------------------------------|-------|-------------------|
| `microsoft/Phi-3-mini-4k-instruct`   | 3.8B  | Free, no login    |
| `meta-llama/Llama-3.2-3B-Instruct`   | 3B    | Needs HF token    |

The prompt template auto-detects based on the model name.

### 5. Run

```bash
# Full demo (indexes sample docs + runs 4 queries)
python main.py

# Single-query mode
python main.py --query "How does Elasticsearch handle vector search?"

# Override model
python main.py --model meta-llama/Llama-3.2-3B-Instruct

# Skip re-indexing if already indexed
python main.py --no-index --query "What is RAG?"

# See all options
python main.py --help
```

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables:

| Setting            | Env Var                  | Default                                  |
|--------------------|--------------------------|------------------------------------------|
| ES host            | `VELOX_ES_HOST`          | `http://localhost:9200`                   |
| ES index           | `VELOX_ES_INDEX`         | `rag_docs`                                |
| LLM model          | `VELOX_LLM_MODEL`        | `microsoft/Phi-3-mini-4k-instruct`        |
| Cache threshold    | `VELOX_SIM_THRESHOLD`    | `0.88`                                    |
| Cache TTL (days)   | `VELOX_CACHE_TTL_DAYS`   | `7`                                       |
| Top-K chunks       | `VELOX_TOP_K`            | `4`                                       |
| Max new tokens     | `VELOX_MAX_NEW_TOKENS`   | `256`                                     |
| 4-bit quantization | `VELOX_USE_4BIT`         | `true`                                    |

---

## Expected Performance (CPU, 8 GB RAM)

| Query type             | Latency      |
|------------------------|--------------|
| Cache HIT              | < 0.1s       |
| Cache MISS (retrieval) | 0.5–1s       |
| Cache MISS (+ LLM)    | 30–90s       |

> **Tip**: The last demo query is semantically similar to the first — you should
> see a cache HIT returning in under 100ms.

---

## Extending Velox

- **Index your own docs**: `pipeline.index([{"blog_id": ..., "title": ..., "content": ...}])`
- **Tune cache**: Lower `VELOX_SIM_THRESHOLD` (e.g. `0.80`) for more hits
- **Swap LLM**: Set `VELOX_LLM_MODEL` to any HuggingFace causal LM
- **Scale cache**: Replace SQLite with Redis for multi-process deployments
- **Add feedback**: Store user ratings to purge low-quality cache entries

---

## Troubleshooting

| Problem                              | Solution                                                       |
|--------------------------------------|----------------------------------------------------------------|
| `ConnectionError: Cannot reach ES`   | Ensure Elasticsearch is running on the configured host/port    |
| `OutOfMemoryError`                   | Set `VELOX_USE_4BIT=true` or use a smaller model               |
| `bitsandbytes` import error          | Remove it from requirements.txt (CPU-only machines)            |
| Slow first query                     | Expected — model loads into RAM on first inference              |
| LLaMA auth error                     | Run `huggingface-cli login` and accept model terms             |

---

## License

MIT

---

> **Elastic Blogathon 2026** — Built with Elasticsearch, Sentence Transformers, and a 3B LLM.
