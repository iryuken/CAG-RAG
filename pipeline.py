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
from utils import cprint, banner, timer


class RAGCAGPipeline:
    """Full RAG + CAG pipeline orchestrator."""

    def __init__(self):
        banner("Velox — RAG + CAG Pipeline")

        cprint("\n[INIT] Loading embedding model …", "cyan")
        self.embedder = SentenceTransformer(Config.EMBED_MODEL)

        cprint("[INIT] Connecting to Elasticsearch …", "cyan")
        self.retriever = ElasticRetriever(self.embedder)

        cprint("[INIT] Setting up semantic cache …", "cyan")
        self.cache = CAGCache(Config.CACHE_DB, self.embedder)

        cprint("[INIT] Loading LLM (this may take a minute on CPU) …", "cyan")
        self.llm = LocalLLM()

        cache_info = self.cache.stats()
        cprint(
            f"\n[READY] Pipeline initialised. "
            f"Cache: {cache_info['active_entries']} active entries.\n",
            "green",
        )

    def index(self, docs: list[dict]):
        """Index documents into Elasticsearch."""
        self.retriever.index_documents(docs)

    def query(self, question: str) -> dict:
        """
        Answer a question using the CAG → RAG → LLM pipeline.

        Returns a dict with: answer, source, chunks, latency,
        and a timing breakdown (embed_ms, retrieval_ms, llm_ms).
        """
        cprint(f"\n── Query: {question!r}", "bold")
        t0 = time.perf_counter()

        # ── 1. CAG: check cache first ────────────────────────────────────
        with timer() as cache_t:
            cached = self.cache.lookup(question, Config.SIM_THRESHOLD)

        if cached:
            return {
                "answer":   cached.answer,
                "source":   "cache",
                "chunks":   json.loads(cached.chunks),
                "latency":  round(time.perf_counter() - t0, 3),
                "timing": {
                    "cache_lookup_ms": round(cache_t["elapsed"] * 1000, 1),
                },
            }

        # ── 2. RAG: retrieve from Elasticsearch ─────────────────────────
        cprint("  [RAG] Retrieving from Elasticsearch …", "cyan")
        with timer() as ret_t:
            chunks = self.retriever.retrieve(question)

        if not chunks:
            return {
                "answer":  "No relevant documents found.",
                "source":  "none",
                "chunks":  [],
                "latency": round(time.perf_counter() - t0, 3),
                "timing":  {},
            }

        context     = "\n\n".join(f"[{c.title}]\n{c.content}" for c in chunks)
        chunk_texts = [c.content for c in chunks]

        # ── 3. LLM: generate answer ─────────────────────────────────────
        cprint("  [LLM] Generating answer …", "cyan")
        with timer() as llm_t:
            answer = self.llm.generate(context, question)

        # ── 4. CAG: store result ─────────────────────────────────────────
        self.cache.store(question, answer, chunk_texts)

        elapsed = round(time.perf_counter() - t0, 3)
        return {
            "answer":  answer,
            "source":  "rag+llm",
            "chunks":  chunk_texts,
            "latency": elapsed,
            "timing": {
                "retrieval_ms": round(ret_t["elapsed"] * 1000, 1),
                "llm_ms":       round(llm_t["elapsed"] * 1000, 1),
            },
        }
