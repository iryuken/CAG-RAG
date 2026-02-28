"""
Velox — Elasticsearch Retriever (RAG Layer)
============================================
Hybrid BM25 + dense‑vector retrieval using Reciprocal Rank Fusion.
"""

from dataclasses import dataclass

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from config import Config
from utils import cprint


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk returned by hybrid search."""
    doc_id:  str
    title:   str
    content: str
    score:   float


# ── Retriever ─────────────────────────────────────────────────────────────────

class ElasticRetriever:
    """Hybrid BM25 + dense vector retrieval from Elasticsearch."""

    def __init__(self, embedder: SentenceTransformer):
        self.es       = Elasticsearch(Config.ES_HOST)
        self.embedder = embedder
        self._health_check()
        self._ensure_index()

    # ── Health & setup ───────────────────────────────────────────────────

    def _health_check(self):
        """Verify Elasticsearch is reachable."""
        if not self.es.ping():
            raise ConnectionError(
                f"Cannot reach Elasticsearch at {Config.ES_HOST}. "
                "Is the server running?"
            )
        cprint(f"  [ES] Connected to {Config.ES_HOST}", "green")

    def _ensure_index(self):
        """Create the index with dense_vector mapping if it doesn't exist."""
        if not self.es.indices.exists(index=Config.ES_INDEX):
            self.es.indices.create(
                index=Config.ES_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "title":     {"type": "text"},
                            "content":   {"type": "text"},
                            "embedding": {
                                "type":       "dense_vector",
                                "dims":       Config.EMBED_DIM,
                                "index":      True,
                                "similarity": "cosine",
                            },
                            "blog_id": {"type": "keyword"},
                        }
                    }
                },
            )
            cprint(f"  [ES] Created index '{Config.ES_INDEX}'", "green")

    # ── Indexing ─────────────────────────────────────────────────────────

    def index_documents(self, docs: list[dict]):
        """
        Index documents into Elasticsearch.

        Each doc is split into ~300‑word chunks before indexing so that
        retrieval returns focused passages rather than full articles.

        Args:
            docs: list of {"title": str, "content": str, "blog_id": str}
        """
        chunk_count = 0
        for doc in docs:
            chunks = self._chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):
                emb = self.embedder.encode(chunk).tolist()
                self.es.index(
                    index=Config.ES_INDEX,
                    document={
                        "title":     doc["title"],
                        "content":   chunk,
                        "embedding": emb,
                        "blog_id":   f"{doc['blog_id']}_chunk{i}",
                    },
                )
                chunk_count += 1
        self.es.indices.refresh(index=Config.ES_INDEX)
        cprint(
            f"  [ES] Indexed {len(docs)} doc(s) → {chunk_count} chunk(s).",
            "green",
        )

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Hybrid search: BM25 keyword + kNN vector, fused via RRF.

        Reciprocal Rank Fusion score = Σ 1/(60 + rank) across both legs.
        """
        q_emb = self.embedder.encode(query).tolist()

        # ── BM25 leg ──
        bm25_resp = self.es.search(
            index=Config.ES_INDEX,
            body={
                "size": Config.TOP_K,
                "query": {"match": {"content": query}},
            },
        )

        # ── kNN leg ──
        knn_resp = self.es.search(
            index=Config.ES_INDEX,
            body={
                "size": Config.TOP_K,
                "knn": {
                    "field":          "embedding",
                    "query_vector":   q_emb,
                    "k":              Config.TOP_K,
                    "num_candidates": Config.TOP_K * 5,
                },
            },
        )

        # ── Reciprocal Rank Fusion ──
        rrf_scores: dict[str, float] = {}
        hits_map:   dict[str, dict]  = {}

        for rank, hit in enumerate(bm25_resp["hits"]["hits"]):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)
            hits_map[doc_id]   = hit["_source"]

        for rank, hit in enumerate(knn_resp["hits"]["hits"]):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)
            hits_map[doc_id]   = hit["_source"]

        ranked = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:Config.TOP_K]

        return [
            RetrievedChunk(
                doc_id  = doc_id,
                title   = hits_map[doc_id].get("title", ""),
                content = hits_map[doc_id].get("content", ""),
                score   = score,
            )
            for doc_id, score in ranked
        ]

    # ── Utilities ────────────────────────────────────────────────────────

    def delete_index(self):
        """Delete the Elasticsearch index for a clean reset."""
        if self.es.indices.exists(index=Config.ES_INDEX):
            self.es.indices.delete(index=Config.ES_INDEX)
            cprint(f"  [ES] Deleted index '{Config.ES_INDEX}'", "yellow")

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 300) -> list[str]:
        """Split text into ~chunk_size-word passages."""
        words  = text.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
