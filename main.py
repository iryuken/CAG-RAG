#!/usr/bin/env python3
"""
Velox — Lightweight RAG + CAG System
=====================================
A cost-efficient Q&A pipeline combining:
  - Elasticsearch hybrid BM25 + kNN retrieval  (RAG)
  - SQLite-backed semantic answer caching      (CAG)
  - 3B-parameter quantized local LLM           (Phi-3 Mini)

Usage:
    python main.py
"""

from config import Config
from pipeline import RAGCAGPipeline


# ── Sample documents ─────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    {
        "blog_id": "es_vector",
        "title":   "How Elasticsearch Handles Vector Search",
        "content": (
            "Elasticsearch supports dense vector fields and kNN (k-nearest "
            "neighbour) search using the HNSW (Hierarchical Navigable Small "
            "World) algorithm. Each document can store a dense_vector field "
            "of fixed dimensions. At query time, the engine computes cosine "
            "or dot-product similarity between the query vector and stored "
            "vectors, returning the top-k matches. Hybrid search combines "
            "BM25 keyword scores with kNN vector scores using Reciprocal "
            "Rank Fusion (RRF) for better relevance."
        ),
    },
    {
        "blog_id": "rag_intro",
        "title":   "Introduction to Retrieval-Augmented Generation",
        "content": (
            "RAG (Retrieval-Augmented Generation) is a technique where a "
            "language model is grounded in external knowledge retrieved at "
            "inference time. Instead of relying solely on parametric memory, "
            "the model receives relevant document chunks as context. This "
            "reduces hallucinations and allows the model to answer questions "
            "about documents it was never trained on. RAG pipelines typically "
            "involve an embedding model, a vector store, and a generative LLM."
        ),
    },
    {
        "blog_id": "cag_intro",
        "title":   "Cache-Augmented Generation for Cost Reduction",
        "content": (
            "CAG (Cache-Augmented Generation) stores LLM-generated answers "
            "against query embeddings. When a new query arrives, its embedding "
            "is compared semantically against cached queries. If the cosine "
            "similarity exceeds a threshold (e.g. 0.88), the cached answer is "
            "returned immediately, skipping both retrieval and LLM inference. "
            "This dramatically cuts latency and compute cost for repeated or "
            "semantically similar queries."
        ),
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pipeline = RAGCAGPipeline()

    # Index sample documents
    pipeline.index(SAMPLE_DOCS)

    # Four demo queries — the last is semantically similar to the first
    questions = [
        "How does Elasticsearch handle vector search?",
        "What is RAG and why is it useful?",
        "How does caching help reduce LLM costs?",
        # Semantically similar to Q1 — should hit cache
        "Explain how Elastic does kNN vector search.",
    ]

    for q in questions:
        result = pipeline.query(q)

        print(f"\n  \u2705 Source  : {result['source']}")
        print(f"  \u23f1 Latency : {result['latency']}s")
        print(f"  \U0001f4ac Answer  : {result['answer'][:300]}")
        if len(result["answer"]) > 300:
            print("               ...")
        print()
        print("\u2500" * 70)


if __name__ == "__main__":
    main()
