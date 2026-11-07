#!/usr/bin/env python3
"""Select minimal, non-overlapping answer-worthy passages with local Ollama.

Each complete ``chunks_est`` field is sent to the model. The field is
chunked by this script when necessary. Sentences are numbered only so the model
can return IDs; selected passages are reconstructed verbatim from the source.

No third-party Python packages are required.

Usage:
    python select_passages.py corpus_clean.json --examples selection_examples.txt > out.json
    # Only some projects / fields (e.g. to re-run fields that failed):
    python select_passages.py corpus_clean.json --examples selection_examples.txt \
        --projects a.json b.json --fields metodologia > out.json

Input: output of clean_corpus.py. --examples: text inserted in the prompt
before the output format (see examples/selection_examples_fictional.txt).

The reduced result goes to stdout; progress messages go to stderr. Full
details are saved after every field to a new file per run,
``<input>.progress.<timestamp>.json`` (e.g.
``documents.progress.20261006-140312.json``). Ollama/network failures are
retried every 10 s until they succeed. Timeouts and runaway answers (over
MAX_OUTPUT_TOKENS) are not retried: the field is marked as an error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """Você seleciona passagens para a criação de datasets de perguntas e respostas em português brasileiro. Siga os critérios literalmente e retorne somente o JSON solicitado."""


PROMPT_TEMPLATE = """Analise TODAS as frases do campo completo abaixo e selecione todas as passagens adequadas para servir como respostas a perguntas.

Uma passagem adequada deve:
- expressar uma informação completa e factual;
- ser compreensível sem consultar frases não selecionadas;
- permitir a criação de uma ou mais perguntas específicas e significativas;
- conter somente texto presente no campo;
- conter uma frase ou o menor conjunto necessário de frases.

Regras de seleção:
- selecione passagens mínimas: não inclua uma frase se a passagem continuar completa sem ela;
- uma frase nunca pode aparecer em duas passagens diferentes;
- se uma frase só faz sentido com outra (por pronome, referência ou explicação incompleta), inclua as duas na mesma passagem, com toda a cadeia de dependência; é preferível juntar frases a descartar uma frase que não faz sentido sozinha;
- as frases de uma passagem não precisam ser vizinhas: inclua apenas as frases das quais a passagem realmente depende (por exemplo [2, 5], se a frase 5 depende apenas da 2);
- preserve a ordem original;
- uma única frase pode conter mais de um fato e continuar sendo uma única passagem;
- não selecione títulos, cabeçalhos numerados (por exemplo "2.1. Gestão do Projeto"), rótulos, listas de palavras-chave, enumerações sem explicação, fragmentos ou conteúdo meramente nominal;
- não reescreva as frases;
- para cada passagem, pense em quais perguntas ela responde, mas não as escreva na saída;
- se nada for adequado, retorne uma lista vazia.

{examples}FORMATO OBRIGATÓRIO
Retorne somente um objeto JSON neste formato:
{{"passages": [{{"sentence_ids": [2, 5]}}, {{"sentence_ids": [7]}}]}}

TRECHO DO CAMPO COMPLETO A SER ANALISADO
Nome do campo: {field_name}

Frases:
{numbered_sentences}
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["sentence_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["passages"],
    "additionalProperties": False,
}


ABBREVIATIONS = {
    "aprox.",
    "art.",
    "al.",
    "dra.",
    "dr.",
    "etc.",
    "ex.",
    "fig.",
    "n.",
    "no.",
    "p.",
    "pp.",
    "prof.",
    "profa.",
    "sr.",
    "sra.",
    "vs.",
}


SECTION_NUMBER = re.compile(r"\d+(?:\.\d+)*\.")

# Title fields are not sent to the model.
SKIPPED_FIELDS = (
    "tituloprojeto",
    "tituloplanodeacao",
    "tituloresultado",
    "tituloatividade",
)

CONFLICT_REASON = "Partial overlap: selections share sentences but neither contains the other."


RETRY_WAIT_SECONDS = 10

# Output token limit (Ollama num_predict). An answer that reaches it is
# treated as runaway: the field is marked as an error and not retried.
MAX_OUTPUT_TOKENS = 8192


class OllamaUnavailable(RuntimeError):
    """Transient Ollama/network failure; worth retrying."""


class OllamaTimeout(RuntimeError):
    """The model did not answer in time; retrying would repeat it."""


def log(message: str) -> None:
    """Print to stderr with the clock time; leading blank lines stay first."""
    text = message.lstrip("\n")
    blank_lines = message[: len(message) - len(text)]
    print(
        f"{blank_lines}[{datetime.now():%H:%M:%S}] {text}",
        file=sys.stderr,
        flush=True,
    )


def format_seconds(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))



def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def is_abbreviation(candidate: str) -> bool:
    lowered = candidate.rstrip().lower()
    last_token_match = re.search(r"([^\s]+)$", lowered)
    if not last_token_match:
        return False
    last_token = last_token_match.group(1).strip('"“”()[]{}')
    if last_token in ABBREVIATIONS:
        return True
    return bool(re.fullmatch(r"(?:[a-záàâãéêíóôõúç]\.){2,}", last_token))


def sentence_spans(text: str) -> list[dict[str, Any]]:
    """Return sentence-like units with exact offsets in the original field."""
    spans: list[dict[str, Any]] = []
    start = 0
    boundary_pattern = re.compile(r"[.!?]+(?:[\"”’\)\]]*)?(?=\s+|$)")

    for match in boundary_pattern.finditer(text):
        end = match.end()
        candidate = text[start:end]
        if match.group(0).startswith(".") and is_abbreviation(candidate):
            continue
        if SECTION_NUMBER.fullmatch(candidate.strip()):
            continue

        sentence_start, sentence_end = trim_span(text, start, end)
        if sentence_start < sentence_end:
            spans.append(
                {
                    "id": len(spans) + 1,
                    "start_char": sentence_start,
                    "end_char": sentence_end,
                    "text": text[sentence_start:sentence_end],
                }
            )
        start = end

    sentence_start, sentence_end = trim_span(text, start, len(text))
    if sentence_start < sentence_end:
        spans.append(
            {
                "id": len(spans) + 1,
                "start_char": sentence_start,
                "end_char": sentence_end,
                "text": text[sentence_start:sentence_end],
            }
        )

    return spans


def display_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(field_name: str, sentences: list[dict[str, Any]], examples: str) -> str:
    numbered = "\n".join(
        f"[{sentence['id']}] {display_sentence(sentence['text'])}"
        for sentence in sentences
    )
    return PROMPT_TEMPLATE.format(
        examples=examples,
        field_name=field_name,
        numbered_sentences=numbered,
    )


def ensure_local_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama URL must use http or https.")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "For corpus privacy, --ollama-url must point to localhost."
        )
    return base_url.rstrip("/")


def http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        # 5xx is a server-side problem that may go away; 4xx will not.
        error_class = OllamaUnavailable if error.code >= 500 else RuntimeError
        raise error_class(f"Ollama HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise OllamaTimeout(f"No answer within {timeout}s.") from error
        raise OllamaUnavailable(
            "Could not connect to local Ollama. Start it with `ollama serve`."
        ) from error
    except TimeoutError as error:
        raise OllamaTimeout(f"No answer within {timeout}s.") from error
    except ConnectionError as error:
        raise OllamaUnavailable(f"Ollama connection failed: {error}") from error


def installed_models(base_url: str, timeout: int) -> list[str]:
    response = http_json(f"{base_url}/api/tags", timeout=timeout)
    return sorted(
        model.get("name", "")
        for model in response.get("models", [])
        if model.get("name")
    )


def call_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    num_ctx: int,
    seed: int,
    keep_alive: str,
    think: str | None = None,
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }
    if think:
        payload["think"] = think
    attempt = 1
    while True:
        try:
            response = http_json(
                f"{base_url}/api/chat",
                payload=payload,
                timeout=timeout,
            )
            break
        except OllamaUnavailable as error:
            log(
                f"    attempt {attempt} failed ({error}); "
                f"retrying in {RETRY_WAIT_SECONDS}s"
            )
            time.sleep(RETRY_WAIT_SECONDS)
            attempt += 1
    if response.get("done_reason") == "length":
        raise RuntimeError(
            f"Model output hit the {MAX_OUTPUT_TOKENS}-token cap (runaway answer)."
        )
    content = response.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty message content.")
    return response, content.strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model output did not contain a JSON object.")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model output JSON must be an object.")
    return parsed


def extract_candidates(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    passages = parsed.get("passages")
    if not isinstance(passages, list):
        raise ValueError("The output must contain a `passages` list.")

    candidates: list[dict[str, Any]] = []
    for passage in passages:
        ids = passage.get("sentence_ids") if isinstance(passage, dict) else passage
        if not isinstance(ids, list):
            ids = []
        candidates.append({"sentence_ids": ids})
    return candidates


def validate_and_reconstruct(
    candidates: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    source_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sentence_count = len(sentences)

    for candidate in candidates:
        original_ids = candidate["sentence_ids"]
        if not original_ids or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in original_ids
        ):
            rejected.append(
                {"sentence_ids": original_ids, "reason": "IDs must be integers."}
            )
            continue

        ids = sorted(set(original_ids))
        if ids[0] < 1 or ids[-1] > sentence_count:
            rejected.append(
                {"sentence_ids": original_ids, "reason": "ID outside field range."}
            )
            continue
        valid.append({"ids": ids})

    # When two selected passages overlap, the longer one is kept.
    valid.sort(key=lambda item: (-len(item["ids"]), item["ids"][0]))
    used_ids: set[int] = set()
    accepted: list[dict[str, Any]] = []
    for item in valid:
        if used_ids.intersection(item["ids"]):
            rejected.append(
                {
                    "sentence_ids": item["ids"],
                    "reason": "Overlaps a longer accepted passage.",
                }
            )
            continue
        accepted.append(item)
        used_ids.update(item["ids"])

    accepted.sort(key=lambda item: item["ids"][0])
    reconstructed: list[dict[str, Any]] = []
    for passage_index, item in enumerate(accepted, start=1):
        ids = item["ids"]
        contiguous = ids == list(range(ids[0], ids[-1] + 1))
        if contiguous:
            passage_text = source_text[
                sentences[ids[0] - 1]["start_char"] : sentences[ids[-1] - 1]["end_char"]
            ]
        else:
            passage_text = " [...] ".join(sentences[i - 1]["text"] for i in ids)
        reconstructed.append(
            {
                "passage_index": passage_index,
                "sentence_ids": ids,
                "contiguous": contiguous,
                "text": passage_text,
                "char_count": len(passage_text),
            }
        )

    return reconstructed, rejected


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def recompute_summary(results: dict[str, Any]) -> None:
    summary = {
        "projects_saved": len(results.get("projects", {})),
        "processed_fields": 0,
        "fields_with_passages": 0,
        "fields_without_passages": 0,
        "failed_fields": 0,
        "selected_passages": 0,
    }
    for project in results.get("projects", {}).values():
        for field in project.get("passage_selection", {}).values():
            status = field.get("status")
            if status in {"selected", "no_passages"}:
                summary["processed_fields"] += 1
            if status == "selected":
                summary["fields_with_passages"] += 1
                summary["selected_passages"] += len(field.get("passages", []))
            elif status == "no_passages":
                summary["fields_without_passages"] += 1
            elif status == "error":
                summary["failed_fields"] += 1
    results["summary"] = summary


def new_results(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    return {
        "model": args.model,
        "task": "minimal non-overlapping answer-worthy passage selection (non-contiguous allowed, longer wins on overlap)",
        "prompting": "few-shot",
        "input_policy": "whole field if it fits; otherwise sentence-aligned chunks with overlap; no truncation",
        "think": args.think,
        "chunk_chars": args.chunk_chars,
        "overlap_chars": args.overlap_chars,
        "sentence_ids_are_one_based": True,
        "ollama_url": args.ollama_url,
        "num_ctx": args.num_ctx,
        "temperature": 0,
        "seed": args.seed,
        "skipped_fields": list(SKIPPED_FIELDS),
        "input_file": str(args.input_json),
        "progress_file": str(output_path),
        "projects": {},
        "summary": {},
    }


SETTING_KEYS = (
    "model",
    "task",
    "prompting",
    "input_policy",
    "think",
    "chunk_chars",
    "overlap_chars",
    "sentence_ids_are_one_based",
    "ollama_url",
    "num_ctx",
    "temperature",
    "seed",
    "skipped_fields",
    "input_file",
)


def reduce_results(results: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Keep settings, run time, summary and each passage's IDs and text."""
    reduced: dict[str, Any] = {key: results.get(key) for key in SETTING_KEYS}
    reduced["run"] = run
    reduced["projects"] = {
        project_name: {
            field_name: [
                {"sentence_ids": p["sentence_ids"], "text": p["text"]}
                for p in field.get("passages", [])
            ]
            for field_name, field in project.get("passage_selection", {}).items()
        }
        for project_name, project in results.get("projects", {}).items()
    }
    reduced["summary"] = results.get("summary", {})
    return reduced


def load_finished_projects(
    input_json: Path,
    model: str,
    expected_fields: dict[str, list[str]],
) -> dict[str, tuple[dict[str, Any], str]]:
    """Projects with every expected field done in an earlier progress file.

    Files are read oldest to newest, so the newest complete copy wins.
    Only files made with the same model are used.
    """
    finished: dict[str, tuple[dict[str, Any], str]] = {}
    pattern = f"{input_json.stem}.progress.*.json"
    for path in sorted(input_json.parent.glob(pattern)):
        try:
            with path.open(encoding="utf-8") as file:
                old = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            log(f"Could not read {path.name} ({error}); ignoring it")
            continue
        if old.get("model") != model:
            continue
        old_projects = old.get("projects", {})
        for name, fields in expected_fields.items():
            selection = old_projects.get(name, {}).get("passage_selection", {})
            if fields and all(
                selection.get(field, {}).get("status") in {"selected", "no_passages"}
                for field in fields
            ):
                finished[name] = ({f: selection[f] for f in fields}, path.name)
    return finished


def make_chunks(
    sentences: list[dict[str, Any]],
    max_chars: int,
    overlap_chars: int,
) -> list[list[int]]:
    """Group sentence indexes into chunks; consecutive chunks overlap."""
    sizes = [len(display_sentence(s["text"])) + 8 for s in sentences]
    if sum(sizes) <= max_chars:
        return [list(range(len(sentences)))]

    chunks: list[list[int]] = []
    start = 0
    while start < len(sentences):
        end = start
        total = 0
        while end < len(sentences) and (
            end == start or total + sizes[end] <= max_chars
        ):
            total += sizes[end]
            end += 1
        chunks.append(list(range(start, end)))
        if end >= len(sentences):
            break
        # Next chunk starts with whole sentences totalling >= overlap_chars.
        next_start = end
        overlap = 0
        while next_start > start + 1 and overlap < overlap_chars:
            next_start -= 1
            overlap += sizes[next_start]
        start = next_start if next_start > start else end
    return chunks


def process_field(
    *,
    field_name: str,
    source_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    log(f"  {field_name}: splitting {len(source_text)} chars into sentences...")
    sentences = sentence_spans(source_text)
    if not sentences:
        return {
            "status": "no_passages",
            "sentence_count": 0,
            "passages": [],
            "rejected_selections": [],
        }

    chunks = make_chunks(sentences, args.chunk_chars, args.overlap_chars)
    log(
        f"  {field_name}: {len(sentences)} sentences, {len(chunks)} chunk(s)"
        " - sending to model..."
    )
    all_candidates: dict[tuple, dict[str, Any]] = {}
    rejected_extra: list[dict[str, Any]] = []
    model_outputs: list[dict[str, Any]] = []
    metrics: dict[str, int] = {}

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk_sentences = [sentences[i] for i in chunk]
        first_id, last_id = chunk_sentences[0]["id"], chunk_sentences[-1]["id"]
        prompt = build_prompt(field_name, chunk_sentences, args.examples_text)
        # The prompt's byte count is used as an upper bound for its token
        # count; a chunk that may not fit in num_ctx raises an error.
        bound = len((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")) + 512
        if bound > args.num_ctx:
            raise ValueError(
                f"Chunk {chunk_number} (sentences {first_id}-{last_id}) may "
                f"exceed --num-ctx. Bound: {bound}; context: {args.num_ctx}. "
                "Lower --chunk-chars or raise --num-ctx."
            )

        log(
            f"    chunk {chunk_number}/{len(chunks)} "
            f"(sentences {first_id}-{last_id}): calling model..."
        )
        started = time.monotonic()
        response, raw_output = call_ollama(
            base_url=args.ollama_url,
            model=args.model,
            prompt=prompt,
            timeout=args.timeout,
            num_ctx=args.num_ctx,
            seed=args.seed,
            keep_alive=args.keep_alive,
            think=args.think,
        )
        log(
            f"    chunk {chunk_number}/{len(chunks)} "
            f"(sentences {first_id}-{last_id}): "
            f"answered in {time.monotonic() - started:.0f}s"
        )
        parsed = parse_json_object(raw_output)
        model_outputs.append(
            {
                "chunk": chunk_number,
                "sentence_range": [first_id, last_id],
                "output": parsed,
            }
        )
        for key in ("total_duration", "prompt_eval_count", "eval_count"):
            if isinstance(response.get(key), int):
                metrics[key] = metrics.get(key, 0) + response[key]

        chunk_candidates = extract_candidates(parsed)
        log(
            f"    chunk {chunk_number}/{len(chunks)}: "
            f"model picked {len(chunk_candidates)} selection(s)"
        )
        for candidate in chunk_candidates:
            ids = candidate["sentence_ids"]
            if not all(
                isinstance(i, int)
                and not isinstance(i, bool)
                and first_id <= i <= last_id
                for i in ids
            ):
                rejected_extra.append(
                    {
                        "sentence_ids": ids,
                        "reason": f"ID outside chunk {chunk_number} range.",
                    }
                )
                continue
            # Same passage found in two chunks (overlap zone): keep one.
            all_candidates.setdefault(tuple(sorted(set(ids))), candidate)

    log(f"  {field_name}: validating selections (IDs, overlaps)...")
    passages, rejected = validate_and_reconstruct(
        list(all_candidates.values()),
        sentences,
        source_text,
    )

    return {
        "status": "selected" if passages else "no_passages",
        "source_chars": len(source_text),
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "chunks": [[sentences[c[0]]["id"], sentences[c[-1]]["id"]] for c in chunks],
        "passages": passages,
        "rejected_selections": rejected_extra + rejected,
        "model_outputs": model_outputs,
        "ollama_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select answer-worthy passages from complete chunks_est fields "
            "using a local Ollama model."
        )
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument(
        "--examples",
        type=Path,
        required=True,
        help="Text file with the few-shot examples inserted into the prompt "
        "(see examples/selection_examples_fictional.txt for the format).",
    )
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Process only these chunks_est field names. Default: all fields.",
    )
    parser.add_argument("--limit-projects", type=int)
    parser.add_argument(
        "--projects",
        nargs="+",
        help="Process only these project keys (e.g. 0000000000001.json).",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=20000,
        help="Max characters of sentence text per chunk (default 20000).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=2000,
        help="Characters (whole sentences) repeated between chunks.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument(
        "--think",
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort for gpt-oss (default: Ollama's default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ollama_url = ensure_local_url(args.ollama_url)
    args.examples_text = args.examples.read_text(encoding="utf-8")
    run_started = datetime.now()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_json}")
    with args.input_json.open(encoding="utf-8") as file:
        projects = json.load(file)
    if not isinstance(projects, dict):
        raise ValueError("The input JSON root must be a dictionary of projects.")

    timestamp = run_started.strftime("%Y%m%d-%H%M%S")
    output_path = args.input_json.with_suffix(f".progress.{timestamp}.json")
    if output_path.exists():
        raise FileExistsError(f"Progress file already exists: {output_path}")
    results = new_results(args, output_path)

    models = installed_models(args.ollama_url, args.timeout)
    if args.model not in models:
        available = "\n  - ".join(models) if models else "(none)"
        raise RuntimeError(
            f"Model {args.model!r} is not installed. Installed models:\n  - {available}\n"
            f"Install it with: ollama pull {args.model}"
        )

    selected_projects = list(projects.items())
    if args.projects:
        missing = [name for name in args.projects if name not in projects]
        if missing:
            raise KeyError(f"Projects not found in input: {', '.join(missing)}")
        selected_projects = [(name, projects[name]) for name in args.projects]
    if args.limit_projects is not None:
        selected_projects = selected_projects[: args.limit_projects]

    def fields_to_process(project: Any) -> list[tuple[str, Any]]:
        if not isinstance(project, dict) or not isinstance(
            project.get("chunks_est"), dict
        ):
            return []
        return [
            (name, text)
            for name, text in project["chunks_est"].items()
            if name not in SKIPPED_FIELDS
            and (not args.fields or name in args.fields)
        ]

    def work_chars(text: Any) -> int:
        return len(text) if isinstance(text, str) and text.strip() else 0

    finished = load_finished_projects(
        args.input_json,
        args.model,
        {
            name: [field for field, _ in fields_to_process(project)]
            for name, project in selected_projects
        },
    )

    total_fields = 0
    total_chars = 0
    for name, project in selected_projects:
        if name in finished:
            continue
        for _, text in fields_to_process(project):
            total_fields += 1
            total_chars += work_chars(text)
    fields_done = 0
    chars_done = 0
    loop_started = time.monotonic()

    log(f"Model: {args.model}")
    log("Input policy: whole fields; long fields chunked with overlap")
    log(f"Skipped fields: {', '.join(SKIPPED_FIELDS)}")
    log(f"Projects: {len(selected_projects)}")
    log(f"Already finished in earlier runs (reused): {len(finished)}")
    log(f"Fields to process: {total_fields} ({total_chars:,} chars)")
    log(f"Progress file: {output_path}")

    for project_index, (project_name, project) in enumerate(
        selected_projects,
        start=1,
    ):
        log(f"\n[{project_index}/{len(selected_projects)}] {project_name}")
        if not isinstance(project, dict):
            log("  skipped: project is not an object")
            continue
        if project_name in finished:
            selection, source_file = finished[project_name]
            results["projects"][project_name] = {
                "metadados": project.get("metadados", {}),
                "passage_selection": selection,
                "reused_from": source_file,
            }
            log(f"  already done - reused from {source_file}")
            continue

        project_result = results["projects"].setdefault(
            project_name,
            {
                "metadados": project.get("metadados", {}),
                "passage_selection": {},
            },
        )
        chunks_est = project.get("chunks_est")
        if not isinstance(chunks_est, dict):
            project_result["error"] = "chunks_est is not a dictionary."
            recompute_summary(results)
            atomic_write_json(output_path, results)
            continue

        for field_name, source_text in fields_to_process(project):
            if not isinstance(source_text, str) or not source_text.strip():
                project_result["passage_selection"][field_name] = {
                    "status": "no_passages",
                    "reason": "Field is not a nonempty string.",
                    "passages": [],
                }
                log(f"  {field_name}: empty/skipped")
            else:
                try:
                    field_result = process_field(
                        field_name=field_name,
                        source_text=source_text,
                        args=args,
                    )
                except Exception as error:  # Save progress before continuing.
                    field_result = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "passages": [],
                    }
                    log(f"  {field_name}: ERROR - {error}")
                else:
                    log(
                        f"  {field_name}: "
                        f"{len(field_result['passages'])} passage(s), "
                        f"{len(field_result['rejected_selections'])} rejected"
                    )
                project_result["passage_selection"][field_name] = field_result

            recompute_summary(results)
            atomic_write_json(output_path, results)
            log("    saved to progress file")

            fields_done += 1
            chars_done += work_chars(source_text)
            elapsed = time.monotonic() - loop_started
            if chars_done:
                eta = format_seconds(elapsed / chars_done * (total_chars - chars_done))
            else:
                eta = "?"
            log(
                f"  done {fields_done}/{total_fields} fields | "
                f"elapsed {format_seconds(elapsed)} | ETA {eta}"
            )

    recompute_summary(results)
    atomic_write_json(output_path, results)
    run_finished = datetime.now()
    run = {
        "started_at": run_started.isoformat(timespec="seconds"),
        "finished_at": run_finished.isoformat(timespec="seconds"),
        "duration_seconds": round((run_finished - run_started).total_seconds()),
    }
    log("\nDone.")
    log(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    json.dump(reduce_results(results, run), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
