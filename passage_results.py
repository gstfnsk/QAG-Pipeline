"""Shared helpers to read the final passage-selection results.

Used by merge_passages.py, verify_passages.py and processing_stats.py.
Only reads files; never writes or changes them.

Files used, next to the corpus (e.g. ``corpus_clean.json``):
- ``corpus_clean.batchNNN.json``: reduced results of each batch (run_batches.py);
- ``corpus_clean.retry_failed.json`` (optional): re-run of fields that
  failed; its fields replace the same fields from the batches;
- ``corpus_clean.progress.<timestamp>.json``: full details per run.
Copies such as ``... 2.json`` (made by iCloud/Finder) are ignored.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


OK_STATUSES = {"selected", "no_passages"}


def log(message: str) -> None:
    text = message.lstrip("\n")
    blank_lines = message[: len(message) - len(text)]
    print(
        f"{blank_lines}[{datetime.now():%H:%M:%S}] {text}",
        file=sys.stderr,
        flush=True,
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def batch_files(corpus_path: Path) -> list[Path]:
    pattern = re.compile(rf"{re.escape(corpus_path.stem)}\.batch\d+\.json")
    return sorted(
        path
        for path in corpus_path.parent.glob(f"{corpus_path.stem}.batch*.json")
        if pattern.fullmatch(path.name)
    )


def progress_files(corpus_path: Path) -> list[Path]:
    """Progress files, oldest first (the name holds the start time)."""
    pattern = re.compile(
        rf"{re.escape(corpus_path.stem)}\.progress\.\d{{8}}-\d{{6}}\.json"
    )
    return sorted(
        path
        for path in corpus_path.parent.glob(f"{corpus_path.stem}.progress.*.json")
        if pattern.fullmatch(path.name)
    )


def retry_file(corpus_path: Path) -> Path | None:
    path = corpus_path.with_suffix(".retry_failed.json")
    return path if path.is_file() else None


def run_start(progress_path: Path) -> datetime:
    stamp = re.search(r"\d{8}-\d{6}", progress_path.name).group(0)
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S")


def load_final_passages(
    corpus_path: Path,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    list[tuple[Path, dict[str, Any]]],
    tuple[Path, dict[str, Any]] | None,
]:
    """Return {project: {field: [passage, ...]}} from batches + retry file.

    Also returns each file's path with its other top-level data (settings,
    run times, summary), as (path, info) pairs.
    """
    paths = batch_files(corpus_path)
    if not paths:
        raise FileNotFoundError(f"No batch files found for {corpus_path}")
    final: dict[str, dict[str, list[dict[str, Any]]]] = {}
    batches: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        log(f"Lendo {path.name}")
        data = load_json(path)
        final.update(data.pop("projects"))
        batches.append((path, data))

    retry = None
    retry_path = retry_file(corpus_path)
    if retry_path:
        log(f"Lendo {retry_path.name} (substitui os campos reprocessados)")
        data = load_json(retry_path)
        for project, fields in data.pop("projects").items():
            final.setdefault(project, {}).update(fields)
        retry = (retry_path, data)
    return final, batches, retry


def load_final_field_records(
    corpus_path: Path,
    wanted: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[dict[str, Any], Path]], list[Path]]:
    """Full progress record for each (project, field) in ``wanted``.

    For each field the newest successful record wins (so the retry run
    replaces the failed one); if a field never succeeded, its newest error
    record is used.
    """
    paths = progress_files(corpus_path)
    best: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for path in paths:
        log(f"Lendo {path.name}")
        for project, data in load_json(path).get("projects", {}).items():
            fields = wanted.get(project)
            if not fields:
                continue
            for field, record in data.get("passage_selection", {}).items():
                if field not in fields:
                    continue
                key = (project, field)
                previous = best.get(key)
                if (
                    previous is None
                    or record.get("status") in OK_STATUSES
                    or previous[0].get("status") not in OK_STATUSES
                ):
                    best[key] = (record, path)
    return best, paths
