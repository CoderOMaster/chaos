"""Command line: pre-fetch models so the first run of an app is instant/offline.

    python -m chaos download                 # default all-MiniLM-L6-v2
    python -m chaos download <model-or-repo>
    chaos download                           # via the installed console script
"""
from __future__ import annotations

import argparse
import sys

from .models import DEFAULT_MODEL, download_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chaos", description="chaos search utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download and cache a model")
    dl.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                    help=f"model name or HF repo id (default: {DEFAULT_MODEL})")

    args = parser.parse_args(argv)
    if args.command == "download":
        model_path, vocab_path = download_model(args.model)
        print(f"model: {model_path}")
        print(f"vocab: {vocab_path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
