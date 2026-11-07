#!/usr/bin/env python3
"""Describe the passage-selection processing: setup, time, calls and output.

Usage:
    python processing_stats.py corpus_clean.json > stats.json

Uses the final results (batches + retried fields) and, for time and tokens,
the detailed records in the progress files. Times per project/field are the
model's own time (Ollama ``total_duration``); clock time is only known per
batch run. The terminal shows a summary in Portuguese; the JSON (stdout) has
every number. Existing files are only read, never changed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from passage_results import (
    load_final_field_records,
    load_final_passages,
    load_json,
    log,
    run_start,
)
from select_passages import MAX_OUTPUT_TOKENS


SETTING_KEYS = (
    "model",
    "think",
    "temperature",
    "seed",
    "num_ctx",
    "chunk_chars",
    "overlap_chars",
    "skipped_fields",
)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "media": round(statistics.fmean(ordered), 2),
        "mediana": round(statistics.median(ordered), 2),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 2),
        "minimo": round(ordered[0], 2),
        "maximo": round(ordered[-1], 2),
    }


def hours(seconds: float) -> float:
    return round(seconds / 3600, 2)


def clock(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_json", type=Path)
    args = parser.parse_args()

    final, batches, retry = load_final_passages(args.corpus_json)
    log(f"Lendo {args.corpus_json.name}")
    corpus = load_json(args.corpus_json)
    wanted = {project: fields for project, fields in final.items()}
    records, progress_paths = load_final_field_records(args.corpus_json, wanted)

    # --- configuration -------------------------------------------------
    runs = [info for _, info in batches] + ([retry[1]] if retry else [])
    settings = {
        key: sorted({json.dumps(info.get(key), ensure_ascii=False) for info in runs})
        for key in SETTING_KEYS
    }
    settings = {
        key: [json.loads(value) for value in values] for key, values in settings.items()
    }
    memory = sysctl("hw.memsize")
    hardware = {
        "processador": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memoria_gb": round(int(memory) / 1e9, 1) if memory else None,
        "sistema": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
    }

    # --- time ----------------------------------------------------------
    batch_runs = [info["run"] for _, info in batches]
    batch_run_seconds = sum(run["duration_seconds"] for run in batch_runs)
    retry_seconds = retry[1]["run"]["duration_seconds"] if retry else 0
    first_start = min(run_start(path) for path in progress_paths)
    last_end = max(
        datetime.fromisoformat(run["finished_at"])
        for run in batch_runs + ([retry[1]["run"]] if retry else [])
    )
    elapsed = (last_end - first_start).total_seconds()

    # --- per field / per project ----------------------------------------
    project_model_seconds: dict[str, float] = defaultdict(float)
    field_model_seconds: list[float] = []
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    calls = chunked_fields = prompt_tokens = output_tokens = 0
    sentences_total = selected_sentences = 0
    rejections: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    chars_total = 0
    missing_records = 0

    for project, fields in final.items():
        source_fields = (corpus.get(project) or {}).get("chunks_est") or {}
        for field, passages in fields.items():
            source = source_fields.get(field)
            chars = len(source) if isinstance(source, str) else 0
            chars_total += chars
            kind = by_type[field]
            kind["campos"] += 1
            kind["caracteres"] += chars
            kind["passagens"] += len(passages)
            kind["campos_sem_passagens"] += 0 if passages else 1
            selected_sentences += sum(len(p["sentence_ids"]) for p in passages)

            entry = records.get((project, field))
            if entry is None:
                missing_records += 1
                continue
            record = entry[0]
            statuses[record.get("status", "?")] += 1
            metrics = record.get("ollama_metrics", {})
            seconds = metrics.get("total_duration", 0) / 1e9
            project_model_seconds[project] += seconds
            field_model_seconds.append(seconds)
            kind["segundos_modelo"] += seconds
            calls += record.get("chunk_count", 0)
            chunked_fields += 1 if record.get("chunk_count", 0) > 1 else 0
            prompt_tokens += metrics.get("prompt_eval_count", 0)
            output_tokens += metrics.get("eval_count", 0)
            sentences_total += record.get("sentence_count", 0)
            for rejected in record.get("rejected_selections", []):
                rejections[re.sub(r"\d+", "N", rejected.get("reason", "?"))] += 1

    model_seconds = sum(field_model_seconds)
    all_passages = [p for fields in final.values() for ps in fields.values() for p in ps]
    passages_per_project = [
        sum(len(ps) for ps in fields.values()) for fields in final.values()
    ]
    passage_chars = [len(p["text"]) for p in all_passages]
    passage_sentences = [len(p["sentence_ids"]) for p in all_passages]
    non_contiguous = sum(
        1
        for p in all_passages
        if p["sentence_ids"] != list(range(p["sentence_ids"][0], p["sentence_ids"][-1] + 1))
    )
    n_passages = len(all_passages)
    n_fields = sum(len(fields) for fields in final.values())

    field_types = {
        field: {
            "campos": int(v["campos"]),
            "caracteres": int(v["caracteres"]),
            "horas_modelo": hours(v["segundos_modelo"]),
            "segundos_por_campo": round(v["segundos_modelo"] / v["campos"], 1),
            "passagens": int(v["passagens"]),
            "passagens_por_campo": round(v["passagens"] / v["campos"], 2),
            "campos_sem_passagens": int(v["campos_sem_passagens"]),
        }
        for field, v in sorted(by_type.items(), key=lambda item: -item[1]["segundos_modelo"])
    }

    failed_in_batches = sum(info["summary"].get("failed_fields", 0) for _, info in batches)
    retried_fields = (
        [
            {"projeto": project, "campo": field}
            for project, fields in load_json(retry[0])["projects"].items()
            for field in fields
        ]
        if retry
        else []
    )

    stats = {
        "configuracao": {
            **settings,
            "limite_tokens_saida": MAX_OUTPUT_TOKENS,
            "observacao_limite": "limite de saída adicionado durante a execução (15/10/2026)",
            "hardware": hardware,
        },
        "corpus": {
            "projetos": len(final),
            "campos_processados": n_fields,
            "caracteres": chars_total,
            "frases": sentences_total,
            "campos_sem_registro_detalhado": missing_records,
        },
        "tempo": {
            "inicio": first_start.isoformat(timespec="seconds"),
            "fim": last_end.isoformat(timespec="seconds"),
            "periodo_total_horas": hours(elapsed),
            "execucao_das_batches_horas": hours(batch_run_seconds),
            "reprocessamento_horas": hours(retry_seconds),
            "tempo_do_modelo_horas": hours(model_seconds),
            "uso_do_modelo_nas_batches": round(model_seconds / batch_run_seconds, 3) if batch_run_seconds else None,
            "projetos_por_hora_de_execucao": round(len(final) / (batch_run_seconds / 3600), 1) if batch_run_seconds else None,
            "segundos_modelo_por_projeto": describe(list(project_model_seconds.values())),
            "segundos_modelo_por_campo": describe(field_model_seconds),
            "segundos_modelo_por_1000_caracteres": round(model_seconds / chars_total * 1000, 2) if chars_total else None,
        },
        "chamadas_ao_modelo": {
            "chamadas": calls,
            "campos_divididos_em_partes": chunked_fields,
            "tokens_entrada": prompt_tokens,
            "tokens_saida": output_tokens,
            "tokens_entrada_por_chamada": round(prompt_tokens / calls, 1) if calls else None,
            "tokens_saida_por_chamada": round(output_tokens / calls, 1) if calls else None,
        },
        "saida": {
            "passagens": n_passages,
            "passagens_por_projeto": describe(passages_per_project),
            "projetos_sem_passagens": sum(1 for n in passages_per_project if n == 0),
            "campos_sem_passagens": sum(1 for fields in final.values() for ps in fields.values() if not ps),
            "caracteres_por_passagem": describe(passage_chars),
            "frases_por_passagem": describe(passage_sentences),
            "passagens_de_uma_frase": round(passage_sentences.count(1) / n_passages, 4) if n_passages else None,
            "passagens_nao_contiguas": round(non_contiguous / n_passages, 4) if n_passages else None,
            "cobertura_de_frases": round(selected_sentences / sentences_total, 4) if sentences_total else None,
        },
        "por_tipo_de_campo": field_types,
        "selecoes_rejeitadas": dict(rejections.most_common()),
        "falhas": {
            "campos_com_falha_nas_batches": failed_in_batches,
            "campos_reprocessados": retried_fields,
            "status_final_dos_campos": dict(statuses),
        },
    }

    # --- terminal summary ---------------------------------------------
    t, s, c = stats["tempo"], stats["saida"], stats["chamadas_ao_modelo"]
    log("\nCONFIGURAÇÃO")
    for key in SETTING_KEYS:
        log(f"  {key}: {', '.join(map(str, settings[key]))}")
    log(f"  hardware: {hardware['processador']}, {hardware['memoria_gb']} GB, {hardware['sistema']}")
    log("\nCORPUS")
    log(f"  {len(final):,} projetos | {n_fields:,} campos | {chars_total:,} caracteres | {sentences_total:,} frases")
    log("\nTEMPO")
    log(f"  Período: {t['inicio']} → {t['fim']} ({clock(elapsed)})")
    log(f"  Execução das batches: {clock(batch_run_seconds)} | reprocessamento: {clock(retry_seconds)}")
    log(f"  Tempo do modelo: {clock(model_seconds)} ({t['uso_do_modelo_nas_batches']:.0%} do tempo das batches)")
    log(f"  Projetos por hora: {t['projetos_por_hora_de_execucao']}")
    pp, pf = t["segundos_modelo_por_projeto"], t["segundos_modelo_por_campo"]
    log(f"  Por projeto (s): média {pp['media']} | mediana {pp['mediana']} | p90 {pp['p90']} | máx {pp['maximo']}")
    log(f"  Por campo (s):   média {pf['media']} | mediana {pf['mediana']} | p90 {pf['p90']} | máx {pf['maximo']}")
    log(f"  Segundos por 1000 caracteres: {t['segundos_modelo_por_1000_caracteres']}")
    log("\nCHAMADAS AO MODELO")
    log(f"  {calls:,} chamadas | {chunked_fields:,} campos divididos em partes")
    log(f"  Tokens: {prompt_tokens:,} entrada ({c['tokens_entrada_por_chamada']}/chamada) | {output_tokens:,} saída ({c['tokens_saida_por_chamada']}/chamada)")
    log("\nSAÍDA")
    log(f"  {n_passages:,} passagens | por projeto: média {s['passagens_por_projeto']['media']}, mediana {s['passagens_por_projeto']['mediana']}")
    log(f"  Projetos sem passagens: {s['projetos_sem_passagens']:,} | campos sem passagens: {s['campos_sem_passagens']:,}")
    log(f"  Caracteres por passagem: média {s['caracteres_por_passagem']['media']}, mediana {s['caracteres_por_passagem']['mediana']}")
    log(f"  Uma frase: {s['passagens_de_uma_frase']:.1%} | não contíguas: {s['passagens_nao_contiguas']:.1%} | cobertura de frases: {s['cobertura_de_frases']:.1%}")
    log("\nPOR TIPO DE CAMPO (ordenado por tempo do modelo)")
    log(f"  {'campo':24s} {'campos':>7s} {'horas':>7s} {'s/campo':>8s} {'passagens':>10s} {'sem pass.':>9s}")
    for field, v in field_types.items():
        log(f"  {field:24s} {v['campos']:7,} {v['horas_modelo']:7.1f} {v['segundos_por_campo']:8.1f} {v['passagens']:10,} {v['campos_sem_passagens']:9,}")
    log("\nSELEÇÕES REJEITADAS")
    for reason, count in rejections.most_common():
        log(f"  {count:7,}  {reason}")
    log("\nFALHAS")
    log(f"  Campos com falha nas batches: {failed_in_batches} | reprocessados: {len(retried_fields)}")
    log(f"  Status final dos campos: {dict(statuses)}")
    if missing_records:
        log(f"  ATENÇÃO: {missing_records} campo(s) sem registro detalhado nos arquivos de progresso")

    json.dump(stats, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
