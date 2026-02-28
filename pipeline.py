"""
Velox — RAG + CAG Pipeline Orchestrator
========================================
Wires together the CAG cache, Elasticsearch retriever, and local LLM
into a single query() call with structured timing breakdown.
"""

import json
import time

from sentence_transformers import SentenceTransformer

from config import Config
from cache import CAGCache
from retriever import ElasticRetriever
from llm import LocalLLM


class RAGCAGPipeline:
    """Full RAG + CAG pipeline orchestrator."""

    def __init__(self):
        print("\n[INIT] Loading embedding model …")
        self.embedder = SentenceTransformer(Config.EMBED_MODEL)

        print("[INIT] Connecting to Elasticsearch …")
        self.retriever = ElasticRetriever(self.embedder)

        print("[INIT] Setting up semantic cache …")
        self.cache = CAGCache(Config.CACHE_DB, self.embedder)

        print("[INIT] Loading LLM (this may take a minute on CPU) …")
        self.llm = LocalLLM()

        print("\n[READY] Pipeline initialised.\n")

    def index(self, docs: list[dict]):
        """Index documents into Elasticsearch."""
        self.retriever.index_documents(docs)

    def query(self, question: str) -> dict:
        """
        Answer a question using the CAG → RAG → LLM pipeline.

        Returns a dict with: answer, source, chunks, latency.
        """
        print(f"\n── Query: '{question}'")
        t0 = time.time()

        # ── 1. CAG: check cache first ────────────────────────────────────
        cached = self.cache.lookup(question, Config.SIM_THRESHOLD)

        if cached:
            return {
                "answer":  cached.answer,
                "source":  "cache",
                "chunks":  json.loads(cached.chunks),
                "latency": round(time.time() - t0, 2),
            }

        # ── 2. RAG: retrieve from Elasticsearch ─────────────────────────
        print("  [RAG] Retrieving from Elasticsearch ...")
        chunks = self.retriever.retrieve(question)

        if not chunks:
            return {
                "answer":  "No relevant documents found.",
                "source":  "none",
                "chunks":  [],
                "latency": round(time.time() - t0, 2),
            }

        context     = "\n\n".join(f"[{c.title}]\n{c.content}" for c in chunks)
        chunk_texts = [c.content for c in chunks]

        # ── 3. LLM: generate answer ─────────────────────────────────────
        print("  [LLM] Generating answer ...")
        answer = self.llm.generate(context, question)

        # ── 4. CAG: store result ─────────────────────────────────────────
        self.cache.store(question, answer, chunk_texts)

        elapsed = round(time.time() - t0, 1)
        return {
            "answer":  answer,
            "source":  "rag+llm",
            "chunks":  chunk_texts,
            "latency": elapsed,
        }
