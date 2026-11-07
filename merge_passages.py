#!/usr/bin/env python3
"""Join all batches into one file: project code -> list of passage texts.

Usage:
    python merge_passages.py corpus_clean.json > passages.json

Output (projects in corpus order, passages in field order):
    {"<project code>": ["passage 1", "passage 2", ...], ...}

Projects without passages appear as []. Passages made of non-adjacent
sentences keep the " [...] " marker. Fields re-run after failing come from
``<corpus>.retry_failed.json``. Existing files are only read, never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passage_results import load_final_passages, load_json, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_json", type=Path)
    args = parser.parse_args()

    final, batches, retry = load_final_passages(args.corpus_json)
    log(f"Lendo {args.corpus_json.name} (para a ordem dos projetos)")
    corpus_order = list(load_json(args.corpus_json))

    missing = [name for name in corpus_order if name not in final]
    if missing:
        log(f"ATENÇÃO: {len(missing)} projeto(s) do corpus sem resultado nas batches")

    merged = {
        name.removesuffix(".json"): [
            passage["text"]
            for passages in final.get(name, {}).values()
            for passage in passages
        ]
        for name in corpus_order
    }

    passages = sum(len(texts) for texts in merged.values())
    empty = sum(1 for texts in merged.values() if not texts)
    log(f"\nBatches lidas: {len(batches)} | arquivo de reprocessamento: {retry[0].name if retry else 'nenhum'}")
    log(f"Projetos: {len(merged):,} | passagens: {passages:,} | projetos sem passagens: {empty:,}")

    json.dump(merged, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
