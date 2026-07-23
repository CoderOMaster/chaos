"""chaos client API over the native engine.

Two clients with identical semantics:

* ``Client`` / ``Index`` — synchronous. The natural fit for a local, CPU-bound
  library; simplest for scripts and notebooks.
* ``AsyncClient`` / ``AsyncIndex`` — async wrappers that offload each call to a
  worker thread (the native engine releases the GIL). Use these inside an event
  loop (FastAPI, agents) or when you want concurrent searches to overlap.

Async here is for integration, **not** speed: search is a local matrix multiply,
so awaiting it adds no throughput on its own — the thread offload is what lets
concurrent calls overlap.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ._core import Engine as _Engine
from .models import DEFAULT_MODEL, resolve_model

DEFAULT_DATA_DIR = "~/.chaos"


@dataclass
class Document:
    """A document to index: a stable string ``id``, its ``text``, and optional
    ``metadata`` (returned with matches but not searched)."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Match:
    """A search match: which document, how similar, and its text/metadata."""
    id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class Index:
    """A synchronous handle to one named (or in-memory) index.

    Opened with a ``name`` and a client ``data_dir``, it persists documents to
    ``<data_dir>/<name>.jsonl`` and reloads them (re-embedding) when reopened by
    that name; without a name it stays purely in-memory.
    """

    def __init__(self, engine: _Engine, name: Optional[str], data_dir: Optional[str]):
        self._engine = engine
        self._name = name
        self._data_dir = data_dir
        self._docs: Dict[str, Document] = {}  # id -> doc (for persistence + metadata)
        self._reload()

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def count(self) -> int:
        return len(self._engine)

    def __len__(self) -> int:
        return len(self._engine)

    @property
    def ef_search(self) -> None:  # write-only knob; no getter on the native side
        raise AttributeError("ef_search is write-only")

    @ef_search.setter
    def ef_search(self, ef: int) -> None:
        """Tune HNSW recall/latency (no-op for a flat index)."""
        self._engine.set_ef_search(ef)

    def add(self, docs: Sequence[Document]) -> Tuple[int, int]:
        """Embed and index documents (upsert by id). Returns (added, updated)."""
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        added, updated = self._engine.upsert_many(ids, texts)
        for d in docs:
            self._docs[d.id] = d
        self._persist()
        return added, updated

    def search(self, query: str, top_k: int = 10) -> List[Match]:
        """Return up to ``top_k`` matches for ``query``, most similar first."""
        hits = self._engine.search(query, top_k)
        return [
            Match(h.id, h.score, h.text, self._docs.get(h.id, Document(h.id, h.text)).metadata)
            for h in hits
        ]

    # --- persistence -----------------------------------------------------
    def _path(self) -> Optional[Path]:
        if not self._name or not self._data_dir:
            return None
        return Path(self._data_dir).expanduser() / f"{self._name}.jsonl"

    def _persist(self) -> None:
        path = self._path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for d in self._docs.values():
                f.write(json.dumps({"id": d.id, "text": d.text, "metadata": d.metadata}) + "\n")
        os.replace(tmp, path)  # atomic

    def _reload(self) -> None:
        path = self._path()
        if path is None or not path.exists():
            return
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                docs.append(Document(r["id"], r["text"], r.get("metadata", {})))
        if docs:
            self.add(docs)  # re-embeds and indexes


class Client:
    """Synchronous entry point.

    With no arguments, the MiniLM model is downloaded (once) and cached
    automatically — no paths required. Pass ``model=`` to pick a different model,
    or ``model_path=``/``vocab_path=`` to use local files (offline).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        data_dir: Optional[str] = DEFAULT_DATA_DIR,
        threads: int = 4,
    ):
        self._model = model
        self._explicit = (model_path, vocab_path) if (model_path and vocab_path) else None
        self._resolved: Optional[Tuple[str, str]] = self._explicit
        self._data_dir = data_dir
        self._threads = threads

    def _resolve(self) -> Tuple[str, str]:
        if self._resolved is None:
            self._resolved = resolve_model(self._model)  # download + cache on first use
        return self._resolved

    def open(
        self,
        name: Optional[str] = None,
        *,
        kind: str = "flat",
        threads: Optional[int] = None,
        hnsw_m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> Index:
        """Open an index: a fresh one, or reload an existing one by ``name``.
        ``kind`` is ``"flat"`` (exact) or ``"hnsw"`` (approximate, for scale)."""
        model_path, vocab_path = self._resolve()
        engine = _Engine(
            model_path, vocab_path,
            index=kind, threads=threads or self._threads,
            hnsw_m=hnsw_m, ef_construction=ef_construction, ef_search=ef_search,
        )
        return Index(engine, name, self._data_dir)


class AsyncIndex:
    """Async facade over :class:`Index`; each call runs in a worker thread."""

    def __init__(self, index: Index):
        self._i = index

    @property
    def name(self) -> Optional[str]:
        return self._i.name

    @property
    def count(self) -> int:
        return self._i.count

    def __len__(self) -> int:
        return len(self._i)

    @property
    def ef_search(self):
        raise AttributeError("ef_search is write-only")

    @ef_search.setter
    def ef_search(self, ef: int) -> None:
        self._i.ef_search = ef

    async def add(self, docs: Sequence[Document]) -> Tuple[int, int]:
        return await asyncio.to_thread(self._i.add, docs)

    async def search(self, query: str, top_k: int = 10) -> List[Match]:
        return await asyncio.to_thread(self._i.search, query, top_k)


class AsyncClient:
    """Async entry point. Model download and index open run off the event loop."""

    def __init__(self, *args, **kwargs):
        self._sync = Client(*args, **kwargs)

    async def open(self, name: Optional[str] = None, **kwargs) -> AsyncIndex:
        index = await asyncio.to_thread(lambda: self._sync.open(name, **kwargs))
        return AsyncIndex(index)
