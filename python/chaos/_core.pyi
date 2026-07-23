"""Type stubs for the native chaos._core module."""
from __future__ import annotations

from typing import List, Tuple

class Hit:
    """A search result."""
    id: str
    score: float
    text: str

class Engine:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        index: str = "flat",
        threads: int = 4,
        hnsw_m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None: ...
    def upsert(self, id: str, text: str) -> bool: ...
    def upsert_many(self, ids: List[str], texts: List[str]) -> Tuple[int, int]: ...
    def search(self, query: str, k: int = 10) -> List[Hit]: ...
    def set_ef_search(self, ef: int) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def is_hnsw(self) -> bool: ...
    def __len__(self) -> int: ...
