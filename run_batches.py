#!/usr/bin/env python3
"""Run select_passages.py over the input in batches.

Usage:
    python run_batches.py documents.json --examples selection_examples.txt
    python run_batches.py documents.json --batch-size 50 --examples selection_examples.txt

Batch k holds projects (k-1)*N+1 .. k*N, in the input file's order. Each
batch's result goes next to the input as ``documents.batch001.json``,
``documents.batch002.json``, ... A batch whose file already exists is skipped,
so re-running only does the missing (or failed) batches. Any other options
(e.g. ``--examples``, ``--model``, ``--think``) are passed through to the selection script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SELECT_SCRIPT = Path(__file__).with_name("select_passages.py")


def log(message: str) -> None:
    text = message.lstrip("\n")
    blank_lines = message[: len(message) - len(text)]
    print(
        f"{blank_lines}[{datetime.now():%H:%M:%S}] {text}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the passage selection script in batches of projects."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    args, passthrough = parser.parse_known_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    with args.input_json.open(encoding="utf-8") as file:
        project_names = list(json.load(file))

    batches = [
        project_names[i : i + args.batch_size]
        for i in range(0, len(project_names), args.batch_size)
    ]
    width = max(3, len(str(len(batches))))
    log(
        f"{len(project_names)} projects -> {len(batches)} batch(es) "
        f"of up to {args.batch_size}"
    )

    done, skipped, failed = [], [], []
    for number, batch in enumerate(batches, start=1):
        batch_path = args.input_json.with_suffix(
            f".batch{number:0{width}d}.json"
        )
        first = (number - 1) * args.batch_size + 1
        label = (
            f"batch {number}/{len(batches)} "
            f"(projects {first}-{first + len(batch) - 1})"
        )
        if batch_path.exists():
            log(f"{label}: {batch_path.name} exists - skipped")
            skipped.append(number)
            continue

        # Progress files are named by the second; avoid a name clash with
        # the previous batch.
        time.sleep(1)
        log(f"\n=== {label} -> {batch_path.name} ===")
        # Write to a temporary file so a crashed batch never looks done.
        temporary_path = batch_path.with_suffix(batch_path.suffix + ".tmp")
        command = [
            sys.executable,
            str(SELECT_SCRIPT),
            str(args.input_json),
            "--projects",
            *batch,
            *passthrough,
        ]
        with temporary_path.open("w", encoding="utf-8") as output:
            exit_code = subprocess.run(command, stdout=output).returncode

        if exit_code == 0:
            temporary_path.replace(batch_path)
            log(f"=== {label}: saved {batch_path.name} ===")
            done.append(number)
        else:
            temporary_path.unlink(missing_ok=True)
            log(f"=== {label}: FAILED (exit code {exit_code}) - re-run to retry ===")
            failed.append(number)

    log("\nAll batches finished.")
    log(f"  done:    {len(done)} {done}")
    log(f"  skipped: {len(skipped)} (already existed)")
    log(f"  failed:  {len(failed)} {failed}")
    if failed:
        log("Re-run the same command to retry the failed batches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
