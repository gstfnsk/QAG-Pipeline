#!/usr/bin/env python3
"""Reduce the original corpus to what the passage selection needs.

For every project:
- metadados: keeps only ``codigoprojeto``;
- chunks_sem (semantic chunks): removed;
- chunks_est (structured fields): keeps every field except the keyword list
  (``palavrachave``).

Title fields (tituloprojeto, tituloatividade, tituloplanodeacao,
tituloresultado) are kept here: select_passages.py skips them without calling
the model, because they are short noun phrases with no complete fact.

Usage:
    python clean_corpus.py corpus.json > corpus_clean.json

Input: {"<project file key>": {"metadados": {...}, "chunks_est": {...},
"chunks_sem": {...}}, ...}. Output (stdout): same keys and order, with
{"metadados": {"codigoprojeto": ...}, "chunks_est": {...}}. A summary goes to
stderr. The input file is only read, never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# chunks_est fields to remove.
REMOVED_FIELDS = {"palavrachave"}


def clean_project(project: dict, removed: Counter) -> dict:
    kept_fields = {}
    for key, value in project["chunks_est"].items():
        if key in REMOVED_FIELDS:
            removed[key] += 1
        else:
            kept_fields[key] = value
    return {
        "metadados": {"codigoprojeto": project["metadados"]["codigoprojeto"]},
        "chunks_est": kept_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_json", type=Path)
    args = parser.parse_args()

    with args.corpus_json.open(encoding="utf-8") as file:
        projects = json.load(file)
    if not isinstance(projects, dict):
        sys.exit("The JSON root must be a dictionary of projects.")

    removed: Counter = Counter()
    cleaned = {name: clean_project(project, removed) for name, project in projects.items()}

    json.dump(cleaned, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    print(f"Projects: {len(cleaned)}", file=sys.stderr)
    print(f"Removed chunks_est fields: {sum(removed.values())}", file=sys.stderr)
    for field_name, count in sorted(removed.items()):
        print(f"  {field_name}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
