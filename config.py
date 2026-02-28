"""
Velox — Configuration
=====================
Central configuration with environment variable overrides.
"""

import os


class Config:
    """All tuneable parameters in one place."""

    # ── Elasticsearch ─────────────────────────────────────────────────────
    ES_HOST  = os.getenv("VELOX_ES_HOST", "http://localhost:9200")
    ES_INDEX = os.getenv("VELOX_ES_INDEX", "rag_docs")

    # ── Embedding model (384-dim, CPU-friendly) ──────────────────────────
    EMBED_MODEL = os.getenv("VELOX_EMBED_MODEL", "all-MiniLM-L6-v2")
    EMBED_DIM   = 384  # must match the chosen model

    # ── LLM ──────────────────────────────────────────────────────────────
    #   "microsoft/Phi-3-mini-4k-instruct"  → free, no login
    #   "meta-llama/Llama-3.2-3B-Instruct"  → needs HF token
    LLM_MODEL = os.getenv(
        "VELOX_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct"
    )

    # ── RAG ──────────────────────────────────────────────────────────────
    TOP_K          = int(os.getenv("VELOX_TOP_K", "4"))
    MAX_CTX_TOKENS = int(os.getenv("VELOX_MAX_CTX_TOKENS", "1500"))

    # ── CAG (cache) ──────────────────────────────────────────────────────
    CACHE_DB       = os.getenv("VELOX_CACHE_DB", "cag_cache.db")
    SIM_THRESHOLD  = float(os.getenv("VELOX_SIM_THRESHOLD", "0.88"))
    CACHE_TTL_DAYS = int(os.getenv("VELOX_CACHE_TTL_DAYS", "7"))

    # ── LLM generation ───────────────────────────────────────────────────
    MAX_NEW_TOKENS = int(os.getenv("VELOX_MAX_NEW_TOKENS", "256"))
    TEMPERATURE    = float(os.getenv("VELOX_TEMPERATURE", "0.2"))

    # ── Quantization (GPU only) ──────────────────────────────────────────
    USE_4BIT_QUANT = os.getenv("VELOX_USE_4BIT", "true").lower() == "true"
