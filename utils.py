"""
Velox — Utilities
=================
Shared helpers: cosine similarity, timer, coloured terminal output.
"""

import time
from contextlib import contextmanager

import numpy as np


# ── Vector math ───────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity using NumPy."""
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    dot  = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-9
    return float(dot / norm)


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager that measures elapsed wall-clock time."""
    t0 = time.perf_counter()
    result = {"elapsed": 0.0}
    try:
        yield result
    finally:
        result["elapsed"] = round(time.perf_counter() - t0, 4)
        if label:
            cprint(f"  ⏱  {label}: {result['elapsed']}s", "dim")


# ── Coloured terminal output ─────────────────────────────────────────────────

_ANSI = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "cyan":    "\033[96m",
    "red":     "\033[91m",
    "magenta": "\033[95m",
}


def cprint(text: str, style: str = "reset"):
    """Print with ANSI colour. Falls back gracefully if unsupported."""
    code = _ANSI.get(style, _ANSI["reset"])
    print(f"{code}{text}{_ANSI['reset']}")


def banner(text: str):
    """Print a boxed header."""
    width = len(text) + 4
    cprint("┌" + "─" * width + "┐", "cyan")
    cprint(f"│  {text}  │", "cyan")
    cprint("└" + "─" * width + "┘", "cyan")
