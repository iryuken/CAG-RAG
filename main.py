#!/usr/bin/env python3
"""
Velox — Lightweight RAG + CAG System
=====================================
A cost-efficient Q&A pipeline combining:
  • Elasticsearch hybrid BM25 + kNN retrieval  (RAG)
  • SQLite-backed semantic answer caching      (CAG)
  • 3B-parameter quantized local LLM           (Phi-3 / LLaMA 3.2)

Usage:
    python main.py                       # run the built-in demo
    python main.py --query "your question here"
    python main.py --model meta-llama/Llama-3.2-3B-Instruct
    python main.py --es-host http://my-es:9200
"""

import argparse
import sys

from config import Config
from pipeline import RAGCAGPipeline
from utils import cprint, banner


# ── Sample documents for the demo ────────────────────────────────────────────

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

DEMO_QUESTIONS = [
    "How does Elasticsearch handle vector search?",
    "What is RAG and why is it useful?",
    "How does caching help reduce LLM costs?",
    # Semantically similar to Q1 — should trigger a cache hit
    "Explain how Elastic does kNN vector search.",
]


# ── Display helpers ──────────────────────────────────────────────────────────

def _print_result(result: dict):
    """Pretty-print a pipeline result."""
    source_colours = {"cache": "green", "rag+llm": "cyan", "none": "red"}
    colour = source_colours.get(result["source"], "reset")

    cprint(f"\n  {'='*60}", "dim")
    cprint(f"  Source  : {result['source']}", colour)
    cprint(f"  Latency : {result['latency']}s", "bold")
    if result.get("timing"):
        parts = ", ".join(f"{k}={v}" for k, v in result["timing"].items())
        cprint(f"  Timing  : {parts}", "dim")
    cprint(f"  Answer  : {result['answer'][:300]}", "reset")
    if len(result["answer"]) > 300:
        cprint("  …(truncated)", "dim")
    cprint(f"  {'='*60}\n", "dim")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="velox",
        description="Velox — Lightweight RAG + CAG Q&A System",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Ask a single question instead of running the full demo.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override the LLM model name (e.g. meta-llama/Llama-3.2-3B-Instruct).",
    )
    parser.add_argument(
        "--es-host",
        type=str,
        default=None,
        help="Override the Elasticsearch host URL.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Override the cache similarity threshold (default: 0.88).",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip indexing sample documents (use if already indexed).",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace):
    """Apply CLI overrides to Config."""
    if args.model:
        Config.LLM_MODEL = args.model
    if args.es_host:
        Config.ES_HOST = args.es_host
    if args.threshold is not None:
        Config.SIM_THRESHOLD = args.threshold


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    apply_overrides(args)

    pipeline = RAGCAGPipeline()

    # Index sample docs unless skipped
    if not args.no_index:
        pipeline.index(SAMPLE_DOCS)

    # Single-query mode
    if args.query:
        result = pipeline.query(args.query)
        _print_result(result)
        return

    # Full demo mode
    banner("Running Demo — 4 Queries")
    for q in DEMO_QUESTIONS:
        result = pipeline.query(q)
        _print_result(result)

    cprint("\nDemo complete. The last query should have been a cache HIT.", "green")


if __name__ == "__main__":
    main()
