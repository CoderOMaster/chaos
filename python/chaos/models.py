"""Model resolution: turn a friendly model name into local (model, vocab) file
paths, downloading and caching from the Hugging Face Hub on first use.

This is what lets ``Client()`` work with no arguments — the MiniLM weights are
fetched once (~86 MB) into the standard HF cache (``~/.cache/huggingface``) and
reused on every later run. Pre-fetch with ``python -m chaos download``.
"""
from __future__ import annotations

import sys
from typing import Dict, Tuple

# Friendly name -> (HF repo, model file, vocab file). These repos ship the
# ONNX export + WordPiece vocab MiniLM was trained with.
KNOWN_MODELS: Dict[str, Tuple[str, str, str]] = {
    "all-MiniLM-L6-v2": ("Xenova/all-MiniLM-L6-v2", "onnx/model.onnx", "vocab.txt"),
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _repo_and_files(model: str, model_file: str | None, vocab_file: str | None):
    if model in KNOWN_MODELS:
        return KNOWN_MODELS[model]
    # Treat an unknown name as a raw HF repo id with conventional filenames.
    return model, model_file or "onnx/model.onnx", vocab_file or "vocab.txt"


def resolve_model(
    model: str = DEFAULT_MODEL,
    *,
    model_file: str | None = None,
    vocab_file: str | None = None,
    quiet: bool = False,
) -> Tuple[str, str]:
    """Return local paths ``(model_path, vocab_path)`` for ``model``.

    Downloads to the HF cache if not already present. ``model`` may be a known
    short name (see ``KNOWN_MODELS``) or any Hugging Face repo id.
    """
    try:
        import logging

        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # quiet token warnings
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Automatic model download needs 'huggingface_hub'. Install it, or pass "
            "model_path=/vocab_path= explicitly to use local files."
        ) from e

    repo, mfile, vfile = _repo_and_files(model, model_file, vocab_file)

    # Fast path: if both files are already cached, use them directly — no network
    # round-trip, no warnings, works offline.
    cm = try_to_load_from_cache(repo, mfile)
    cv = try_to_load_from_cache(repo, vfile)
    if isinstance(cm, str) and isinstance(cv, str):
        return cm, cv

    if not quiet:
        print(f"chaos: downloading model '{model}' from {repo} "
              f"(~86 MB, first run only)…", file=sys.stderr, flush=True)
    model_path = hf_hub_download(repo, mfile)
    vocab_path = hf_hub_download(repo, vfile)
    return model_path, vocab_path


def download_model(model: str = DEFAULT_MODEL) -> Tuple[str, str]:
    """Eagerly fetch a model into the cache; returns its local paths."""
    return resolve_model(model)
