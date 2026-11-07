#!/usr/bin/env python3
"""Check that every selected passage really comes from the original document.

Usage:
    python verify_passages.py corpus_clean.json > verification.json

For each passage (batches + retried fields) it checks:
1. texto: the text is in the original field. First an exact match; if that
   fails, a match ignoring spaces/line breaks. Passages made of non-adjacent
   sentences (" [...] ") must have each part in the field, in order.
2. ids: rebuilding the passage from its sentence_ids gives the same text.
3. sobreposição: no sentence is used by two passages of the same field.

The terminal shows a summary; the JSON (stdout) has the summary and every
problem found. Existing files are only read, never changed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from passage_results import load_final_passages, load_json, log
from select_passages import sentence_spans


GAP = " [...] "


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parts_in_order(parts: list[str], source: str) -> bool:
    position = 0
    for part in parts:
        found = source.find(part, position)
        if found == -1:
            return False
        position = found + len(part)
    return True


def text_match(text: str, source: str) -> str:
    """'exata', 'ignorando_espacos' or 'nao_encontrada'."""
    if text in source:
        return "exata"
    parts = text.split(GAP)
    if len(parts) > 1 and parts_in_order(parts, source):
        return "exata"
    squashed_source = squash(source)
    if parts_in_order([squash(part) for part in parts], squashed_source):
        return "ignorando_espacos"
    return "nao_encontrada"


def rebuild_from_ids(ids: list[int], sentences: list[dict[str, Any]], source: str) -> str | None:
    if not ids or ids[0] < 1 or ids[-1] > len(sentences):
        return None
    if ids == list(range(ids[0], ids[-1] + 1)):
        return source[sentences[ids[0] - 1]["start_char"] : sentences[ids[-1] - 1]["end_char"]]
    return GAP.join(sentences[i - 1]["text"] for i in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_json", type=Path)
    args = parser.parse_args()

    final, batches, retry = load_final_passages(args.corpus_json)
    log(f"Lendo {args.corpus_json.name}")
    corpus = load_json(args.corpus_json)

    matches: Counter[str] = Counter()
    problems: list[dict[str, Any]] = []
    passages_total = ids_ok = 0
    fields_total = fields_with_overlap = 0

    def problem(project: str, field: str, index: int | None, issue: str, **extra: Any) -> None:
        problems.append(
            {"projeto": project, "campo": field, "passagem": index, "problema": issue, **extra}
        )

    log("Verificando...")
    for project, fields in final.items():
        source_fields = (corpus.get(project) or {}).get("chunks_est") or {}
        for field, passages in fields.items():
            fields_total += 1
            source = source_fields.get(field)
            if not isinstance(source, str):
                problem(project, field, None, "campo_nao_existe_no_corpus")
                continue
            sentences = sentence_spans(source)
            used: Counter[int] = Counter()
            for index, passage in enumerate(passages, start=1):
                passages_total += 1
                text, ids = passage["text"], passage["sentence_ids"]

                match = text_match(text, source)
                matches[match] += 1
                if match == "nao_encontrada":
                    problem(project, field, index, "texto_nao_encontrado", texto=text)

                if rebuild_from_ids(ids, sentences, source) == text:
                    ids_ok += 1
                else:
                    problem(project, field, index, "ids_nao_correspondem_ao_texto", sentence_ids=ids, texto=text)
                used.update(ids)

            repeated = sorted(i for i, count in used.items() if count > 1)
            if repeated:
                fields_with_overlap += 1
                problem(project, field, None, "frase_em_mais_de_uma_passagem", sentence_ids=repeated)

    summary = {
        "batches_lidas": len(batches),
        "arquivo_reprocessamento": retry[0].name if retry else None,
        "projetos": len(final),
        "campos": fields_total,
        "passagens": passages_total,
        "texto_encontrado_exato": matches["exata"],
        "texto_encontrado_ignorando_espacos": matches["ignorando_espacos"],
        "texto_nao_encontrado": matches["nao_encontrada"],
        "ids_correspondem_ao_texto": ids_ok,
        "campos_com_frase_repetida": fields_with_overlap,
        "total_de_problemas": len(problems),
    }

    def pct(value: int) -> str:
        return f"{value / passages_total:.2%}" if passages_total else "-"

    log("\nRESUMO DA VERIFICAÇÃO")
    log(f"  Projetos: {len(final):,} | campos: {fields_total:,} | passagens: {passages_total:,}")
    log(f"  Texto encontrado (exato):             {matches['exata']:,} ({pct(matches['exata'])})")
    log(f"  Texto encontrado (ignorando espaços): {matches['ignorando_espacos']:,} ({pct(matches['ignorando_espacos'])})")
    log(f"  Texto NÃO encontrado:                 {matches['nao_encontrada']:,} ({pct(matches['nao_encontrada'])})")
    log(f"  IDs correspondem ao texto:            {ids_ok:,} ({pct(ids_ok)})")
    log(f"  Campos com frase em 2+ passagens:     {fields_with_overlap:,}")
    log(f"  Total de problemas: {len(problems):,}")

    json.dump({"resumo": summary, "problemas": problems}, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
