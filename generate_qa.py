#!/usr/bin/env python3
"""Generate up to 2 specific question/answer pairs per project with local Ollama.

Each try is one model call on one passage. The model decides whether a
relevant, self-contained and specific question can be answered by the
passage; if so, it writes the question, a short answer taken from the
passage, a verbatim evidence span and the question type (factual, como,
por_que). Automatic checks run on every generated pair (see process()).

Per project:
1. Passages starting with an infinitive ("Avaliar...") or with "Objetivo(s)"
   are skipped; they do not count as tries.
2. The other passages are tried from the longest to the shortest (passages of
   the same length in a fixed-seed random order), up to --tentativas tries,
   until --por-projeto questions are accepted.
3. A question is accepted ("conta": true) when its evidence is found in the
   passage and no warning was raised. Marked questions stay in the output
   with "conta": false. A question almost equal to one already accepted for
   the same project is discarded.
Projects used in the few-shot examples are left out.

Usage:
    # Sample of 100 projects:
    python generate_qa.py passages_filtered.json --exemplos qa_examples.json --projetos 100 > qa_sample.json
    # Same projects as a previous output:
    python generate_qa.py passages_filtered.json --exemplos qa_examples.json --projetos-de qa_sample.json > qa_sample_b.json
    # All projects:
    python generate_qa.py passages_filtered.json --exemplos qa_examples.json > qa_all.json

Input: {"project_code": ["passage", ...]} (output of filter_passages.py).
Few-shot examples: JSON with "positivos" and "negativos" lists (see
examples/qa_examples_fictional.json for the format).
Progress: one JSONL line per try in <input>.qa_v3_progress.jsonl; re-running
skips tries already done with the same settings, prompt and examples.
Output (stdout): config, summary, questions and every try per project. The
terminal shows progress and a summary in Portuguese.
The input file is only read, never changed.

Requires: pip install spacy && python -m spacy download pt_core_news_md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from passage_results import load_json, log
from select_passages import (
    MAX_OUTPUT_TOKENS,
    RETRY_WAIT_SECONDS,
    OllamaUnavailable,
    ensure_local_url,
    http_json,
    installed_models,
    parse_json_object,
)
from verify_passages import text_match


SPACY_MODEL = "pt_core_news_md"

SYSTEM_PROMPT = """Você avalia passagens de projetos de pesquisa da Embrapa e cria perguntas e respostas em português brasileiro para avaliar sistemas de busca. Siga as regras literalmente e retorne somente o JSON solicitado."""

INSTRUCTIONS = """Avalie a passagem abaixo, tirada de um projeto de pesquisa da Embrapa, e decida se é possível criar UMA pergunta relevante que ela responda.

Regras para a pergunta:
- deve ser compreensível sem consultar outras passagens (inclua na própria pergunta o contexto necessário);
- não use referências ao documento, como "segundo o texto", "nesta passagem", "no projeto", "do projeto", "na proposta" ou "descrita na proposta", e não cite códigos de projeto;
- teste: imagine alguém lendo a pergunta sem saber de que projeto ela veio; essa pessoa entende do que se trata? Se a pergunta fala de "o sistema", "a atividade", "a tecnologia", "o experimento" ou algo parecido sem dizer qual é, a resposta é não;
- deve ser específica: use detalhes concretos da passagem (espécie, cultura, local, número, instituição, método, autor citado) para que, idealmente, só este projeto a responda; se a passagem não tem nenhum detalhe que a diferencie de outros projetos, use "gerar": false;
- quando a informação depende do tempo (dados de um ano, projeções "recentes"), inclua na pergunta o ano ou a fonte citada;
- evite copiar as palavras da passagem: reformule com suas palavras, mantendo nomes próprios, números e termos técnicos;
- deve ser aberta; não faça perguntas de sim ou não (se a ideia natural for de sim ou não, reformule como pergunta aberta);
- não pode revelar a resposta nem conter parte dela;
- não pergunte fatos amplamente conhecidos, que qualquer enciclopédia traz e um leitor saberia sem consultar o projeto (por exemplo, em que regiões vivem as abelhas sem ferrão, ou qual estado tem a maior área de babaçu); dados específicos citados pela passagem (números, mecanismos, autores) podem ser perguntados;
- não atribua à passagem informações que ela não diz (por exemplo, chamar de objetivo, resultado ou meta algo que não é apresentado assim, ou comparar quando a passagem não compara); se só der para perguntar inventando isso, use "gerar": false;
- se a passagem se refere a algo que ela não especifica ("estas", "este sistema", "essa atividade", "desta matéria-prima") e a pergunta precisaria disso para ser entendida, use "gerar": false, mesmo que a passagem tenha outros detalhes concretos (nomes de empresas, municípios, números).

Regras para a resposta:
- deve ser obtida obrigatoriamente da passagem; pode copiar ou reescrever o texto, sem acrescentar informação;
- deve ser curta e trazer informação que não está na pergunta; se a resposta só repetiria a pergunta, use "gerar": false;
- deve ser um fato que a passagem afirma (o que será usado, onde, quanto, como, por quê); se a passagem só anuncia uma ação (avaliar, estudar, analisar, desenvolver...) sem nenhum detalhe concreto de como, onde, com quê, quanto ou com que resultado, use "gerar": false;
- não pode ser um rótulo interno do projeto (T1, T4, tratamento 2, atividade 3) nem uma sigla que a passagem não explica; se a passagem tem um fato melhor, pergunte sobre ele;
- "evidencia" é o trecho da passagem, copiado literalmente, que sustenta a resposta.

"tipo" é o tipo da pergunta: "factual" (qual, quem, quanto, onde, quando, o que), "como" (procedimento ou método) ou "por_que" (causa ou motivo). Escolha o tipo que a passagem permite, sem forçar.

Se não for possível criar uma pergunta assim (por exemplo, a passagem depende de outras, é um fragmento, é vaga, é genérica ou não tem informação concreta), use "gerar": false e deixe tipo, pergunta, resposta e evidencia vazios.
{justificativa_regra}
EXEMPLOS
{exemplos}

PASSAGEM A AVALIAR
Passagem: {passagem}
"""

# Version of the automatic rules; part of the progress fingerprint, so
# changing it makes a re-run redo every try.
METHOD_VERSION = 6

# Passages starting with "Objetivo(s)" are skipped before the model call.
OBJECTIVE_START = re.compile(r"^\W*objetivos?\b", re.IGNORECASE)

QUESTION_TYPES = ["factual", "como", "por_que"]

# Share of shared content words (Jaccard) above which a question counts as a
# near-duplicate of one already accepted for the same project.
DUPLICATE_THRESHOLD = 0.6

# Share of the answer's content words already in the question at or above
# which the pair is discarded as "answer repeats the question".
REPEAT_THRESHOLD = 0.8

# Words ignored when comparing questions and answers (content_stems).
STOPWORDS = {
    "será", "serão", "para", "pela", "pelo", "pelas", "pelos", "como", "qual",
    "quais", "quem", "quando", "onde", "quanto", "sobre", "entre", "este",
    "esta", "estes", "estas", "esse", "essa", "esses", "essas", "isso", "isto",
    "mais", "menos", "também", "deve", "devem", "pode", "podem", "foram",
    "sendo", "está", "estão", "cada", "todo", "toda", "todos", "todas", "seus",
    "suas", "outro", "outra", "outros", "outras", "mesmo", "ainda", "após",
    "desde", "durante", "além", "assim", "então", "porque", "pois", "forma",
    "meio", "através", "projeto", "proposta", "realizada", "realizadas",
    "realizado", "realizados", "realização",
}

# Warning "possivel_sim_nao": none of these in the question's first 4 words.
INTERROGATIVES = re.compile(
    r"\b(qual|quais|quem|quando|onde|como|por que|porque|quanto|quanta|quantos|quantas|o que|que)\b",
    re.IGNORECASE,
)
# Warning "referencia_ao_texto".
TEXT_REFERENCE = re.compile(
    r"\b(segundo o texto|no texto|do texto|nesta passagem|na passagem|neste trecho|no trecho"
    r"|neste projeto|nesta proposta|deste projeto|desta proposta)\b",
    re.IGNORECASE,
)

# Warning "cita_codigo": a 13-digit project code.
PROJECT_CODE = re.compile(r"\b\d{13}\b")

# Warning "referencia_ao_projeto": any mention of "projeto"/"proposta"
# ("no projeto", "descrita na proposta").
PROJECT_REFERENCE = re.compile(r"\b(projetos?|propostas?)\b", re.IGNORECASE)

# Warning "rotulo_interno", in the question or the answer: a label followed by
# a number ("subprojeto 7", "atividade 3.1"), an uppercase Roman numeral
# ("etapa II") or a letter/numeral in parentheses ("etapa (i)"). Roman
# numerals must be uppercase, so "ação civil" does not match.
INTERNAL_LABEL = re.compile(
    r"(?<!\w)(?:(?i:subprojetos?|atividades?|tratamentos?|etapas?|fases?|metas?"
    r"|planos? de ação|ações|ação|experimentos?|objetivos?(?: específicos?)?)|PA)"
    r"\s+(?:\d+(?:\.\d+)*\b|[IVX]+\b|(?i:\([ivx]{1,4}\)|\([a-z]\)))"
)


def output_schema(with_justification: bool) -> dict[str, Any]:
    fields = ["gerar", "tipo", "pergunta", "resposta", "evidencia"]
    if with_justification:
        fields.append("justificativa")
    properties: dict[str, Any] = {name: {"type": "string"} for name in fields}
    properties["gerar"] = {"type": "boolean"}
    properties["tipo"] = {"type": "string", "enum": QUESTION_TYPES + [""]}
    return {
        "type": "object",
        "properties": properties,
        "required": fields,
        "additionalProperties": False,
    }


def format_examples(examples: dict[str, list[dict[str, str]]], with_justification: bool) -> str:
    """Positive then negative examples, each as passage + expected JSON output.

    The example's project code is not shown in the prompt.
    """
    blocks = []
    for number, example in enumerate(examples["positivos"] + examples["negativos"], start=1):
        output: dict[str, Any] = {
            "gerar": "pergunta" in example,
            "tipo": example.get("tipo", ""),
            "pergunta": example.get("pergunta", ""),
            "resposta": example.get("resposta", ""),
            "evidencia": example.get("evidencia", ""),
        }
        if with_justification:
            output["justificativa"] = example["justificativa"]
        blocks.append(
            f"Exemplo {number}\nPassagem: {example['passagem']}\n"
            f"Saída: {json.dumps(output, ensure_ascii=False)}\n"
        )
    return "\n".join(blocks)


def build_prompt(passage: str, examples_text: str, with_justification: bool) -> str:
    rule = (
        '"justificativa": explique em uma frase curta por que a passagem permite ou não uma pergunta.\n'
        if with_justification
        else ""
    )
    return INSTRUCTIONS.format(justificativa_regra=rule, exemplos=examples_text, passagem=passage)


def call_model(args: argparse.Namespace, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One Ollama /api/chat call with a JSON-schema output; returns (parsed JSON, raw response).

    Connection failures are retried every RETRY_WAIT_SECONDS; an answer longer
    than MAX_OUTPUT_TOKENS or an empty answer raises an error.
    """
    payload = {
        "model": args.modelo,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": args.seed,
            "num_ctx": args.num_ctx,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }
    if args.think:
        payload["think"] = args.think
    attempt = 1
    while True:
        try:
            response = http_json(f"{args.ollama_url}/api/chat", payload=payload, timeout=args.timeout)
            break
        except OllamaUnavailable as error:
            log(f"    tentativa {attempt} falhou ({error}); nova tentativa em {RETRY_WAIT_SECONDS}s")
            time.sleep(RETRY_WAIT_SECONDS)
            attempt += 1
    if response.get("done_reason") == "length":
        raise RuntimeError(f"Resposta do modelo passou de {MAX_OUTPUT_TOKENS} tokens (descontrolada).")
    content = response.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("O Ollama devolveu uma resposta vazia.")
    return parse_json_object(content), response


def load_spacy():
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy não instalado. Rode: pip install spacy")
    try:
        # Only the tagger/morphology is needed.
        return spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    except OSError:
        sys.exit(f"Modelo não instalado. Rode: python -m spacy download {SPACY_MODEL}")


def starts_with_infinitive(nlp, passage: str) -> bool:
    """True when the first word of the passage is a verb in the infinitive."""
    words = [token for token in nlp(passage) if token.is_alpha]
    if not words:
        return False
    first = words[0]
    return first.pos_ in {"VERB", "AUX"} and "Inf" in first.morph.get("VerbForm")


def shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def content_stems(text: str) -> list[str]:
    """First 5 letters of content words (crude stemming: captação ~ captar)."""
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    return [word[:5] for word in words if len(word) >= 4 and word not in STOPWORDS]


def answer_repeats_question(question: str, answer: str) -> bool:
    answer_stems = content_stems(answer)
    if not answer_stems:  # e.g. "Até 40%." adds information
        return False
    question_stems = set(content_stems(question))
    share = sum(stem in question_stems for stem in answer_stems) / len(answer_stems)
    return share >= REPEAT_THRESHOLD


def discarded(item: dict[str, str], reason: str, justification: str, with_justification: bool) -> dict[str, Any]:
    """Result of a try whose pair was discarded (no question is kept)."""
    result: dict[str, Any] = {
        "projeto": item["projeto"], "passagem": item["passagem"], "gerou": False,
        "pergunta": "", "resposta": "", "evidencia": "",
        "evidencia_ok": None, "evidencia_match": None, "avisos": [],
        "descartada_por": reason,
    }
    if with_justification:
        result["justificativa"] = justification
    return result


def question_warnings(question: str) -> list[str]:
    warnings = []
    if not question.strip().endswith("?"):
        warnings.append("sem_interrogacao")
    if TEXT_REFERENCE.search(question):
        warnings.append("referencia_ao_texto")
    first_words = " ".join(question.split()[:4])
    if not INTERROGATIVES.search(first_words):
        warnings.append("possivel_sim_nao")
    if PROJECT_CODE.search(question):
        warnings.append("cita_codigo")
    if PROJECT_REFERENCE.search(question) and "referencia_ao_texto" not in warnings:
        warnings.append("referencia_ao_projeto")
    return warnings


def similarity(first: str, second: str) -> float:
    """Jaccard similarity of the content stems of two questions."""
    a, b = set(content_stems(first)), set(content_stems(second))
    return len(a & b) / len(a | b) if a | b else 0.0


def process(item: dict[str, str], args: argparse.Namespace, prompt_text: str, schema: dict[str, Any], with_justification: bool) -> dict[str, Any]:
    """One try: one passage, one model call, then the automatic checks.

    Discarded (no question kept): prompt plus output reached the context
    window ("contexto_excedido"); the answer repeats the question
    ("resposta_repete_pergunta").
    Kept, with flags: "evidencia_ok" (evidence found verbatim, or ignoring
    spaces, in the passage) and "avisos" (warnings: sem_interrogacao,
    referencia_ao_texto, possivel_sim_nao, cita_codigo, referencia_ao_projeto,
    rotulo_interno, resposta_vazia, sem_tipo).
    """
    started = time.monotonic()
    parsed, response = call_model(args, prompt_text, schema)
    prompt_tokens = response.get("prompt_eval_count") or 0
    # Ollama silently cuts prompts longer than num_ctx; the answer would then
    # come from a truncated prompt, so the try is discarded.
    if prompt_tokens + (response.get("eval_count") or 0) >= args.num_ctx:
        result = discarded(
            item, "contexto_excedido",
            f"Prompt ({prompt_tokens} tokens) mais saída passaram do contexto de {args.num_ctx} tokens.",
            with_justification,
        )
        result["tipo"] = ""
        result["segundos"] = round(time.monotonic() - started, 1)
        result["tokens_prompt"] = prompt_tokens
        result["tokens_saida"] = response.get("eval_count")
        return result
    generated = bool(parsed.get("gerar"))
    if generated and answer_repeats_question(parsed.get("pergunta", ""), parsed.get("resposta", "")):
        result = discarded(
            item, "resposta_repete_pergunta",
            "Resposta só repete a pergunta (descartada pela checagem automática).",
            with_justification,
        )
        result["tipo"] = ""
        result["descartada_original"] = {
            "tipo": parsed.get("tipo", ""),
            "pergunta": parsed.get("pergunta", ""),
            "resposta": parsed.get("resposta", ""),
            "justificativa_modelo": parsed.get("justificativa", ""),
        }
        result["segundos"] = round(time.monotonic() - started, 1)
        result["tokens_prompt"] = prompt_tokens
        result["tokens_saida"] = response.get("eval_count")
        return result
    result: dict[str, Any] = {
        "projeto": item["projeto"],
        "passagem": item["passagem"],
        "gerou": generated,
        "tipo": parsed.get("tipo", "") if generated else "",
        "pergunta": parsed.get("pergunta", "") if generated else "",
        "resposta": parsed.get("resposta", "") if generated else "",
        "evidencia": parsed.get("evidencia", "") if generated else "",
    }
    if generated:
        match = text_match(result["evidencia"], item["passagem"]) if result["evidencia"].strip() else "vazia"
        result["evidencia_ok"] = match in {"exata", "ignorando_espacos"}
        result["evidencia_match"] = match
        result["avisos"] = question_warnings(result["pergunta"])
        if INTERNAL_LABEL.search(result["pergunta"]) or INTERNAL_LABEL.search(result["resposta"]):
            result["avisos"].append("rotulo_interno")
        if not result["resposta"].strip():
            result["avisos"].append("resposta_vazia")
        if not result["tipo"]:
            result["avisos"].append("sem_tipo")
    else:
        result["evidencia_ok"] = None
        result["evidencia_match"] = None
        result["avisos"] = []
    if with_justification:
        result["justificativa"] = parsed.get("justificativa", "")
    result["segundos"] = round(time.monotonic() - started, 1)
    result["tokens_prompt"] = prompt_tokens
    result["tokens_saida"] = response.get("eval_count")
    return result


def log_try(result: dict[str, Any]) -> None:
    reason = result.get("descartada_por")
    if "segundos" in result:
        log(f"    modelo respondeu em {result['segundos']}s ({result['tokens_saida']} tokens de saída)")
    if reason == "contexto_excedido":
        log(f"    prompt de {result['tokens_prompt']} tokens passou do contexto → DESCARTADA")
    elif reason == "resposta_repete_pergunta":
        log(f"    pergunta do modelo: {result['descartada_original']['pergunta']}")
        log("    resposta repete a pergunta → DESCARTADA pela checagem automática")
    elif reason == "pergunta_repetida":
        log(f"    pergunta do modelo: {result['pergunta']}")
        log("    quase igual a uma pergunta já aceita deste projeto → DESCARTADA")
    elif result["gerou"]:
        log(f"    PERGUNTA ({result['tipo'] or 'sem tipo'}): {result['pergunta']}")
        log(f"    resposta: {shorten(result['resposta'])}")
        log(f"    evidência: {'ok' if result['evidencia_ok'] else 'NÃO ENCONTRADA na passagem'}")
        if result["avisos"]:
            log(f"    avisos: {', '.join(result['avisos'])}")
        if not result["conta"]:
            log("    marcada → mantida no JSON, mas não conta para a meta do projeto")
    else:
        log("    sem pergunta")
    if result.get("justificativa") and not reason:
        log(f"    justificativa: {shorten(result['justificativa'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("passagens_json", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--projetos", type=int, help="Sorteia N projetos. Sem a opção, usa todos.")
    group.add_argument("--projetos-de", type=Path, help="Usa os mesmos projetos de uma saída anterior deste script.")
    parser.add_argument("--por-projeto", type=int, default=2, help="Perguntas por projeto (padrão 2).")
    parser.add_argument("--tentativas", type=int, default=20, help="Máximo de passagens tentadas por projeto (padrão 20).")
    parser.add_argument("--exemplos", type=Path, required=True, help="JSON com os exemplos few-shot (positivos e negativos).")
    parser.add_argument("--modelo", default="gpt-oss:20b")
    parser.add_argument("--sem-justificativa", action="store_true", help="Não pede justificativa ao modelo.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()
    args.ollama_url = ensure_local_url(args.ollama_url)

    with_justification = not args.sem_justificativa
    data: dict[str, list[str]] = load_json(args.passagens_json)
    examples = load_json(args.exemplos)
    example_projects = {e["projeto"] for e in examples["positivos"] + examples["negativos"]}
    eligible = [project for project in data if project not in example_projects]
    if args.projetos is not None:
        if args.projetos > len(eligible):
            parser.error(f"Só há {len(eligible):,} projetos fora dos exemplos.")
        chosen = set(random.Random(args.seed).sample(eligible, args.projetos))
        projects = [project for project in eligible if project in chosen]
    elif args.projetos_de is not None:
        previous = [p["projeto"] for p in load_json(args.projetos_de)["projetos"]]
        projects = [project for project in previous if project in data and project not in example_projects]
        log(f"Projetos de {args.projetos_de.name}: {len(previous):,} | agora nos exemplos (deixados de fora): "
            f"{sum(project in example_projects for project in previous):,}")
    else:
        projects = eligible
    examples_text = format_examples(examples, with_justification)
    schema = output_schema(with_justification)

    # Tries are reused only if model, prompt and examples are the same.
    fingerprint = hashlib.sha256(
        json.dumps(
            [METHOD_VERSION, args.modelo, args.seed, args.think, args.num_ctx, SYSTEM_PROMPT, INSTRUCTIONS, examples_text, with_justification],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]

    log(f"Lendo {args.passagens_json.name}: {len(data):,} projetos")
    log(f"Projetos dos exemplos deixados de fora: {len(data) - len(eligible):,}")
    log(f"Projetos nesta rodada: {len(projects):,}" + (f" (sorteados, seed {args.seed})" if args.projetos is not None else ""))
    log("Verificando o modelo no Ollama...")
    models = installed_models(args.ollama_url, args.timeout)
    if args.modelo not in models:
        sys.exit(f"Modelo {args.modelo!r} não está instalado no Ollama.")

    log("Carregando o spaCy (para detectar passagens no infinitivo)...")
    nlp = load_spacy()

    progress_path = args.passagens_json.with_suffix(".qa_v3_progress.jsonl")
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if record.get("config") == fingerprint and "erro" not in record:
                    done[(record["projeto"], record["passagem"])] = record
    log(f"Modelo: {args.modelo} | exemplos: {args.exemplos.name} | justificativa: {'sim' if with_justification else 'não'}")
    log(f"Meta: {args.por_projeto} perguntas por projeto, até {args.tentativas} tentativas")
    log(f"Tentativas já feitas (reaproveitadas): {len(done):,}")
    log(f"Progresso salvo em: {progress_path}")

    loop_started = time.monotonic()
    calls = 0
    project_results = []
    with progress_path.open("a", encoding="utf-8") as progress:
        for number, project in enumerate(projects, start=1):
            order = list(data[project])
            random.Random(f"{args.seed}:{project}").shuffle(order)
            # Longest first (in words); the sort is stable, so passages of the
            # same length keep the shuffled order.
            order.sort(key=lambda passage: -len(passage.split()))
            log(f"\n[{number}/{len(projects)}] {project} ({len(order)} passagens)")
            accepted: list[dict[str, Any]] = []
            tries: list[dict[str, Any]] = []
            skipped = {"infinitivo": 0, "objetivo": 0}
            error = None
            for passage in order:
                if len(accepted) >= args.por_projeto or len(tries) >= args.tentativas:
                    break
                if OBJECTIVE_START.match(passage):
                    skipped["objetivo"] += 1
                    log(f"  pulada (começa com \"Objetivo\", não conta como tentativa): {shorten(passage)}")
                    continue
                if starts_with_infinitive(nlp, passage):
                    skipped["infinitivo"] += 1
                    log(f"  pulada (começa no infinitivo, não conta como tentativa): {shorten(passage)}")
                    continue
                try_number = len(tries) + 1
                log(f"  tentativa {try_number}: {shorten(passage)}")
                item = {"projeto": project, "passagem": passage}
                result = done.get((project, passage))
                if result is None:
                    prompt_text = build_prompt(passage, examples_text, with_justification)
                    try:
                        result = process(item, args, prompt_text, schema, with_justification)
                    except Exception as exc:  # Record and stop this project; redone on re-run.
                        error = f"{type(exc).__name__}: {exc}"
                        record = {**item, "erro": error, "config": fingerprint}
                        progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                        progress.flush()
                        log(f"    ERRO - {exc} (projeto será refeito na próxima execução)")
                        break
                    calls += 1
                    result["config"] = fingerprint
                    progress.write(json.dumps(result, ensure_ascii=False) + "\n")
                    progress.flush()
                    done[(project, passage)] = result
                else:
                    log("    (reaproveitada do progresso)")
                result = {key: value for key, value in result.items() if key != "config"}
                result["tentativa"] = try_number
                if result["gerou"]:
                    repeated = next(
                        (q for q in accepted if similarity(q["pergunta"], result["pergunta"]) >= DUPLICATE_THRESHOLD),
                        None,
                    )
                    if repeated:
                        result["gerou"] = False
                        result["descartada_por"] = "pergunta_repetida"
                        result["repete"] = repeated["pergunta"]
                    else:
                        # Marked questions (evidence not found or warnings) are kept but do not count.
                        result["conta"] = bool(result["evidencia_ok"]) and not result["avisos"]
                        if result["conta"]:
                            accepted.append(result)
                tries.append(result)
                log_try(result)
            project_results.append({
                "projeto": project,
                "passagens": len(order),
                "perguntas": len(accepted),
                "puladas": skipped,
                "erro": error,
                "tentativas": tries,
            })
            if calls:
                elapsed = time.monotonic() - loop_started
                eta = timedelta(seconds=round(elapsed / number * (len(projects) - number)))
                log(f"  projeto com {len(accepted)} perguntas | decorrido {timedelta(seconds=round(elapsed))} | ETA {eta}")
            else:
                log(f"  projeto com {len(accepted)} perguntas")

    questions = [
        {key: t[key] for key in ("projeto", "tentativa", "conta", "tipo", "pergunta", "resposta", "evidencia",
                                 "evidencia_ok", "evidencia_match", "avisos", "passagem")
         } | ({"justificativa": t["justificativa"]} if "justificativa" in t else {})
        for p in project_results for t in p["tentativas"] if t["gerou"]
    ]
    all_tries = [t for p in project_results for t in p["tentativas"]]
    complete = [p for p in project_results if p["erro"] is None]
    by_count = {n: sum(1 for p in complete if p["perguntas"] == n) for n in range(args.por_projeto + 1)}
    types: dict[str, int] = {}
    warnings: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for q in questions:
        types[q["tipo"] or "sem_tipo"] = types.get(q["tipo"] or "sem_tipo", 0) + 1
        for warning in q["avisos"]:
            warnings[warning] = warnings.get(warning, 0) + 1
    for t in all_tries:
        if not t["gerou"]:
            reason = t.get("descartada_por") or "modelo_nao_gerou"
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "projetos": len(project_results),
        "projetos_com_erro": len(project_results) - len(complete),
        "projetos_sem_passagens": sum(1 for p in complete if p["passagens"] == 0),
        "projetos_por_numero_de_perguntas": by_count,
        "perguntas": len(questions),
        "perguntas_que_contam": sum(1 for q in questions if q["conta"]),
        "perguntas_marcadas": sum(1 for q in questions if not q["conta"]),
        "tipos": types,
        "evidencia_ok": sum(1 for q in questions if q["evidencia_ok"]),
        "avisos": warnings,
        "tentativas": len(all_tries),
        "tentativas_sem_pergunta": reasons,
        "tokens_prompt_max": max((t.get("tokens_prompt") or 0 for t in all_tries), default=0),
        "passagens_puladas": {
            reason: sum(p["puladas"][reason] for p in project_results) for reason in ("infinitivo", "objetivo")
        },
    }
    config = {
        "modelo": args.modelo, "seed": args.seed, "think": args.think,
        "por_projeto": args.por_projeto, "tentativas": args.tentativas,
        "exemplos": args.exemplos.name, "projetos_dos_exemplos_excluidos": sorted(example_projects & set(data)),
        "sorteio_de_projetos": args.projetos,
        "projetos_de": args.projetos_de.name if args.projetos_de else None, "config_hash": fingerprint,
    }
    json.dump({"config": config, "resumo": summary, "perguntas": questions, "projetos": project_results},
              sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    log("\nRESUMO")
    log(f"  Projetos: {summary['projetos']:,} | com erro (refazer rodando de novo): {summary['projetos_com_erro']:,}")
    for n in range(args.por_projeto, -1, -1):
        log(f"    com {n} perguntas: {by_count[n]:,}")
    log(f"    sem nenhuma passagem: {summary['projetos_sem_passagens']:,}")
    log(f"  Perguntas: {len(questions):,} | contam para a meta: {summary['perguntas_que_contam']:,} "
        f"| marcadas (evidência não encontrada ou aviso): {summary['perguntas_marcadas']:,}")
    for name, count in sorted(types.items(), key=lambda kv: -kv[1]):
        log(f"    tipo {name}: {count:,}")
    if questions:
        ok = summary["evidencia_ok"]
        log(f"  Evidência encontrada na passagem: {ok:,} ({ok / len(questions):.1%})")
    for warning, count in sorted(warnings.items(), key=lambda kv: -kv[1]):
        log(f"  Aviso {warning}: {count:,}")
    log(f"  Tentativas: {len(all_tries):,}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log(f"    sem pergunta por {reason}: {count:,}")
    log(f"  Maior prompt: {summary['tokens_prompt_max']:,} tokens (contexto: {args.num_ctx:,})")
    log("  Passagens puladas (não contam como tentativa): "
        f"infinitivo {summary['passagens_puladas']['infinitivo']:,} | \"Objetivo\" {summary['passagens_puladas']['objetivo']:,}")


if __name__ == "__main__":
    main()
