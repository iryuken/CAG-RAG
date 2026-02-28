"""
Velox — CAG Cache Layer
========================
SQLite-backed semantic cache that stores LLM-generated answers
against query embeddings. Returns cached answers for semantically
similar queries, skipping both retrieval and LLM inference.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import Config
from utils import cosine_similarity, cprint


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A single cached query–answer pair."""
    query_hash:  str
    query_text:  str
    query_vec:   str   # JSON-encoded float list
    answer:      str
    chunks:      str   # JSON-encoded list of chunk texts
    created_at:  float


# ── Cache ─────────────────────────────────────────────────────────────────────

class CAGCache:
    """SQLite-backed semantic cache with cosine-similarity lookup."""

    def __init__(self, db_path: str, embedder: SentenceTransformer):
        self.conn     = sqlite3.connect(db_path, check_same_thread=False)
        self.embedder = embedder
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash  TEXT PRIMARY KEY,
                query_text  TEXT,
                query_vec   TEXT,
                answer      TEXT,
                chunks      TEXT,
                created_at  REAL
            )
        """)
        self.conn.commit()

    # ── Core operations ──────────────────────────────────────────────────

    def lookup(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """Return a cached entry if a semantically similar query exists."""
        q_vec = self.embedder.encode(query).tolist()

        cutoff = time.time() - Config.CACHE_TTL_DAYS * 86_400
        rows   = self.conn.execute(
            "SELECT * FROM cache WHERE created_at > ?", (cutoff,)
        ).fetchall()

        best_score: float           = -1.0
        best_entry: Optional[CacheEntry] = None

        for row in rows:
            entry    = CacheEntry(*row)
            cached_v = json.loads(entry.query_vec)
            sim      = cosine_similarity(q_vec, cached_v)
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_score >= threshold:
            cprint(f"  [CAG] Cache HIT  (similarity={best_score:.3f})", "green")
            return best_entry

        cprint(f"  [CAG] Cache MISS (best similarity={best_score:.3f})", "yellow")
        return None

    def store(self, query: str, answer: str, chunks: list[str]):
        """Store a query–answer pair in the cache."""
        q_vec = self.embedder.encode(query).tolist()
        entry = CacheEntry(
            query_hash = self._hash(query),
            query_text = query,
            query_vec  = json.dumps(q_vec),
            answer     = answer,
            chunks     = json.dumps(chunks),
            created_at = time.time(),
        )
        self.conn.execute("""
            INSERT OR REPLACE INTO cache
            VALUES (:query_hash, :query_text, :query_vec,
                    :answer, :chunks, :created_at)
        """, asdict(entry))
        self.conn.commit()
        cprint("  [CAG] Answer stored in cache.", "dim")

    # ── Utilities ────────────────────────────────────────────────────────

    def clear(self):
        """Purge all cached entries."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()
        cprint("  [CAG] Cache cleared.", "yellow")

    def stats(self) -> dict:
        """Return basic cache statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        cutoff = time.time() - Config.CACHE_TTL_DAYS * 86_400
        active = self.conn.execute(
            "SELECT COUNT(*) FROM cache WHERE created_at > ?", (cutoff,)
        ).fetchone()[0]
        return {"total_entries": total, "active_entries": active}

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()
