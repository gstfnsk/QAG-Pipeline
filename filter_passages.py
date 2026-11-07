#!/usr/bin/env python3
"""Remove short passages that only name an action or announce a future one.

The passage type comes from spaCy grammar analysis (pt_core_news_md):
- infinitivo: first word is a verb in the infinitive ("Avaliar...");
- substantivo_de_acao: first word is a noun (or capitalized noun tagged as
  proper noun) and the passage has no conjugated verb ("Montagem dos equipamentos.");
- futuro: the main verb of the first sentence is in the future
  ("Serão avaliados...", "Os dados serão analisados...").

A justification marker ("para", "por meio de", "visando", "porque"...) means
the passage says why or how something is done. A passage is removed when it
has no justification marker and:
- infinitivo / substantivo_de_acao: at most N words (--max-palavras / --ver-removidas);
- futuro: at most --max-palavras-futuro words (default 10).

Usage:
    # Random samples per type and length band:
    python filter_passages.py passages.json --amostras > samples.txt
    # Every passage that would be removed with N=10 (nothing is filtered):
    python filter_passages.py passages.json --ver-removidas 10 > removed_N10.txt
    # Filter:
    python filter_passages.py passages.json --max-palavras 10 \
        --removidas passages_removed_N10.json > passages_filtered_N10.json

Input: {"project_code": ["passage", ...]} (output of merge_passages.py).
Output (stdout): the same structure without the removed passages; the removed
ones go to --removidas, with project, text, type and word count.
The input file is only read, never changed.

Requires: pip install spacy && python -m spacy download pt_core_news_md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from passage_results import log

SPACY_MODEL = "pt_core_news_md"
TYPES = ("infinitivo", "substantivo_de_acao", "futuro")
DEFAULT_FUTURE_LIMIT = 10

# A short passage containing any of these is kept.
JUSTIFICATION_MARKERS = (
    "para",
    "por meio de",
    "através de",
    "visando",
    "a fim de",
    "com o objetivo de",
    "com a finalidade de",
    "de modo a",
    "de forma a",
    "pois",
    "porque",
    "uma vez que",
    "já que",
    "devido",
    "servindo",
    "garantindo",
    "permitindo",
    "possibilitando",
)
MARKER = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in JUSTIFICATION_MARKERS) + r")\b",
    re.IGNORECASE,
)

# Word-count bands used by --amostras, and samples drawn per band.
SAMPLE_BANDS = ((0, 15), (16, 20), (21, 25))
SAMPLES_PER_GROUP = 20
SEED = 42

# Marker between non-adjacent sentences of one passage; not counted as a word.
GAP = "[...]"


def word_count(text: str) -> int:
    return sum(1 for word in text.split() if word != GAP)


def load_spacy():
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy não instalado. Rode: pip install spacy")
    try:
        # The parser finds the main verb of the first sentence (future detection).
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError:
        sys.exit(f"Modelo não instalado. Rode: python -m spacy download {SPACY_MODEL}")


def is_future(token) -> bool:
    return "Fut" in token.morph.get("Tense")


def passage_type(doc) -> str | None:
    """'infinitivo', 'substantivo_de_acao', 'futuro' or None for a spaCy doc."""
    words = [token for token in doc if token.is_alpha]
    if not words:
        return None
    first = words[0]

    if first.pos_ in {"VERB", "AUX"} and "Inf" in first.morph.get("VerbForm"):
        return "infinitivo"

    has_finite_verb = any(
        token.pos_ in {"VERB", "AUX"} and "Fin" in token.morph.get("VerbForm")
        for token in doc
    )
    # A capitalized first noun may be tagged as a proper noun ("Montagem").
    if first.pos_ in {"NOUN", "PROPN"} and not has_finite_verb:
        return "substantivo_de_acao"

    root = next(iter(doc.sents)).root
    if is_future(root) or any(
        child.dep_.startswith("aux") and is_future(child) for child in root.children
    ):
        return "futuro"
    return None


def classify_all(nlp, texts: list[str], max_words: int) -> dict[str, dict[str, Any]]:
    """Type info for every distinct passage with at most max_words words."""
    short = sorted({t for t in texts if word_count(t) <= max_words})
    log(f"Analisando {len(short):,} passagens curtas (até {max_words} palavras) com spaCy...")
    result: dict[str, dict[str, Any]] = {}
    for done, (text, doc) in enumerate(zip(short, nlp.pipe(short, batch_size=256)), start=1):
        kind = passage_type(doc)
        if kind:
            marker = MARKER.search(text)
            result[text] = {
                "tipo": kind,
                "palavras": word_count(text),
                "marcador": marker.group(0).lower() if marker else None,
            }
        if done % 20000 == 0:
            log(f"  {done:,}/{len(short):,}")
    return result


def is_removed(item: dict[str, Any] | None, max_words: int, future_limit: int) -> bool:
    if item is None or item["marcador"]:
        return False
    limit = future_limit if item["tipo"] == "futuro" else max_words
    return item["palavras"] <= limit


def bands_for(kind: str, max_words: int, future_limit: int) -> list[tuple[int, int]]:
    limit = future_limit if kind == "futuro" else max_words
    if kind == "futuro":
        return [(0, limit)]
    bands = [(low, min(high, limit)) for low, high in SAMPLE_BANDS if low <= limit]
    if limit > SAMPLE_BANDS[-1][1]:
        bands.append((SAMPLE_BANDS[-1][1] + 1, limit))
    return bands


def band_label(low: int, high: int) -> str:
    return f"até {high}" if low == 0 else f"{low}–{high}"


def write_samples(texts: list[str], info: dict[str, dict[str, Any]], future_limit: int) -> None:
    rng = random.Random(SEED)
    max_words = SAMPLE_BANDS[-1][1]
    out = sys.stdout
    out.write(f"AMOSTRAS DO FILTRO COM SPACY + FUTURO ({len(texts):,} passagens, modelo {SPACY_MODEL})\n")
    out.write(f"Futuro: removido só com até {future_limit} palavras.\n\n")

    out.write("TOTAL QUE SERIA REMOVIDO, POR LIMITE N (futuro fixo)\n")
    out.write(f"  {'tipo':22s}" + "".join(f"{'N=' + str(h):>10s}" for _, h in SAMPLE_BANDS) + "\n")
    totals = [0] * len(SAMPLE_BANDS)
    for kind in TYPES:
        row = ""
        for i, (_, n) in enumerate(SAMPLE_BANDS):
            count = sum(1 for t in texts if is_removed(info.get(t), n, future_limit) and info[t]["tipo"] == kind)
            totals[i] += count
            row += f"{count:>10,}"
        out.write(f"  {kind:22s}{row}\n")
    out.write(f"  {'TOTAL':22s}" + "".join(f"{t:>10,}" for t in totals) + "\n")

    for kind in TYPES:
        for low, high in bands_for(kind, max_words, future_limit):
            group = [t for t in texts if is_removed(info.get(t), max_words, future_limit)
                     and info[t]["tipo"] == kind and low <= info[t]["palavras"] <= high]
            out.write(f"\n=== {kind} | {band_label(low, high)} palavras | {len(group):,} passagens ===\n")
            for text in rng.sample(group, min(SAMPLES_PER_GROUP, len(group))):
                out.write(f"  - {text}\n")
    for kind in TYPES:
        limit = future_limit if kind == "futuro" else max_words
        group = [(info[t]["marcador"], t) for t in texts
                 if info.get(t) and info[t]["tipo"] == kind and info[t]["marcador"]
                 and info[t]["palavras"] <= limit]
        out.write(f"\n=== {kind} | SALVAS por marcador (até {limit} palavras) | {len(group):,} passagens ===\n")
        for marker, text in rng.sample(group, min(SAMPLES_PER_GROUP, len(group))):
            out.write(f"  - [{marker}] {text}\n")
    log(f"Amostras geradas. Removíveis com N=15/20/25: {totals[0]:,} / {totals[1]:,} / {totals[2]:,}")


def write_removed(data: dict[str, list[str]], info: dict[str, dict[str, Any]], max_words: int, future_limit: int) -> None:
    removed: dict[str, list[tuple[int, str, str]]] = {kind: [] for kind in TYPES}
    for project, texts in data.items():
        for text in texts:
            item = info.get(text)
            if is_removed(item, max_words, future_limit):
                removed[item["tipo"]].append((item["palavras"], project, text))

    out = sys.stdout
    total = sum(len(rows) for rows in removed.values())
    out.write(f"PASSAGENS REMOVIDAS COM N={max_words}, FUTURO ATÉ {future_limit} PALAVRAS (sem marcador de justificativa)\n")
    out.write(f"Total: {total:,} | " + " | ".join(f"{k}: {len(v):,}" for k, v in removed.items()) + "\n")
    for kind in TYPES:
        rows = sorted(removed[kind])
        for low, high in bands_for(kind, max_words, future_limit):
            group = [row for row in rows if low <= row[0] <= high]
            out.write(f"\n=== {kind} | {band_label(low, high)} palavras | {len(group):,} passagens ===\n")
            for words, project, text in group:
                out.write(f"  [{words:2d}] {project} | {text}\n")
    log(f"Listadas {total:,} passagens que seriam removidas (N={max_words}, futuro ≤ {future_limit}).")


def run_filter(data: dict[str, list[str]], info: dict[str, dict[str, Any]], removed_path: Path, max_words: int, future_limit: int) -> None:
    kept_data: dict[str, list[str]] = {}
    removed: list[dict[str, Any]] = []
    removed_by_type: Counter[str] = Counter()
    for project, texts in data.items():
        kept = []
        for text in texts:
            item = info.get(text)
            if is_removed(item, max_words, future_limit):
                removed.append({"projeto": project, "texto": text,
                                "tipo": item["tipo"], "palavras": item["palavras"]})
                removed_by_type[item["tipo"]] += 1
                continue
            kept.append(text)
        kept_data[project] = kept

    with removed_path.open("x", encoding="utf-8") as file:
        json.dump(removed, file, ensure_ascii=False, indent=2)
        file.write("\n")
    json.dump(kept_data, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    total = sum(len(texts) for texts in data.values())
    kept_total = sum(len(texts) for texts in kept_data.values())
    log(f"\nFILTRO COM SPACY + FUTURO (N={max_words}, futuro ≤ {future_limit}, sem marcador de justificativa)")
    log(f"  Projetos: {len(kept_data):,}")
    log(f"  Passagens: {total:,} → mantidas {kept_total:,} | removidas {len(removed):,} ({len(removed) / total:.1%})")
    for kind, count in removed_by_type.most_common():
        log(f"    removidas {kind}: {count:,}")
    log(f"  Removidas salvas em: {removed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("passages_json", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--amostras", action="store_true", help="Gera amostras para escolher o limite.")
    mode.add_argument("--ver-removidas", type=int, metavar="N", help="Lista (txt) todas as passagens que seriam removidas com este N; não filtra nada.")
    mode.add_argument("--max-palavras", type=int, help="Limite de palavras (infinitivo/substantivo) para remover.")
    parser.add_argument("--max-palavras-futuro", type=int, default=DEFAULT_FUTURE_LIMIT,
                        help=f"Limite de palavras para o futuro (padrão {DEFAULT_FUTURE_LIMIT}).")
    parser.add_argument("--removidas", type=Path, help="Arquivo para as passagens removidas (novo).")
    args = parser.parse_args()

    if args.max_palavras is not None:
        if args.removidas is None:
            parser.error("--max-palavras precisa de --removidas.")
        if args.removidas.resolve() == args.passages_json.resolve():
            parser.error("--removidas não pode ser o arquivo de entrada.")
        if args.removidas.exists():
            parser.error(f"{args.removidas} já existe; escolha outro nome ou apague antes.")

    nlp = load_spacy()
    log(f"Lendo {args.passages_json.name}")
    with args.passages_json.open(encoding="utf-8") as file:
        data = json.load(file)
    texts = [text for texts in data.values() for text in texts]

    future_limit = args.max_palavras_futuro
    if args.amostras:
        max_words = SAMPLE_BANDS[-1][1]
    elif args.ver_removidas is not None:
        max_words = args.ver_removidas
    else:
        max_words = args.max_palavras
    info = classify_all(nlp, texts, max(max_words, future_limit))

    if args.amostras:
        write_samples(texts, info, future_limit)
    elif args.ver_removidas is not None:
        write_removed(data, info, max_words, future_limit)
    else:
        run_filter(data, info, args.removidas, max_words, future_limit)


if __name__ == "__main__":
    main()
