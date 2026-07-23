"""chaos — CPU-only native semantic search with a Python API.

The heavy lifting (tokenize → MiniLM embed → vector search) runs entirely in
C++; Python just drives it. The MiniLM model downloads and caches automatically
on first use, so no paths are required.

Synchronous (recommended for scripts/notebooks)::

    from chaos import Client, Document

    client = Client()                      # auto-downloads MiniLM on first use
    index = client.open("notes")           # kind="flat" (default) or "hnsw"
    added, updated = index.add([
        Document(id="1", text="Ship the on-device SDK by Friday."),
        Document(id="2", text="Follow up with LiveKit about latency."),
    ])
    for m in index.search("what's due this week", top_k=3):
        print(m.id, m.score, m.text)        # m: Match(id, score, text, metadata)

Async (same semantics; for event-loop apps)::

    from chaos import AsyncClient, Document
    client = AsyncClient()
    index = await client.open("notes")
    await index.add([...])
    matches = await index.search("...", top_k=3)

``Engine`` and ``Hit`` are the low-level native primitives, exposed for advanced
use; most callers want ``Client``/``AsyncClient``.
"""
from __future__ import annotations

from ._core import Engine, Hit
from .client import (
    AsyncClient,
    AsyncIndex,
    Client,
    Document,
    Index,
    Match,
)
from .models import download_model

__all__ = [
    "Client",
    "AsyncClient",
    "Index",
    "AsyncIndex",
    "Document",
    "Match",
    "download_model",
    "Engine",
    "Hit",
]
__version__ = "0.1.2"
