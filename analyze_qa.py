"""Resume uma ou mais saídas do generate_qa.py, para comparar rodadas.

Uso:
    python analyze_qa.py rodada_a.json rodada_b.json
    python analyze_qa.py rodada_a.json --amostra 20 > amostra.txt

Para cada arquivo: taxa de perguntas que contam por tamanho da passagem,
tempo por chamada e projetos incompletos. Com --amostra N, imprime N perguntas
que contam, sorteadas (seed 42), com a passagem, para leitura manual.
"""
import argparse
import json
import random
import statistics

BUCKETS = [(0, 30), (30, 60), (60, 100), (100, 300), (300, 1000), (1000, None)]


def bucket_name(low, high):
    return f"{low}+" if high is None else f"{low}-{high}"


def summarize(path, sample_size):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tries = [t for p in data["projetos"] for t in p["tentativas"]]
    print(f"\n=== {path} ===")
    print(f"tentativas: {len(tries)}")

    print("palavras      tentativas  geraram  contam  taxa_contam")
    for low, high in BUCKETS:
        group = [t for t in tries
                 if len(t["passagem"].split()) >= low and (high is None or len(t["passagem"].split()) < high)]
        if not group:
            continue
        generated = sum(1 for t in group if t.get("gerou"))
        counted = sum(1 for t in group if t.get("conta"))
        print(f"{bucket_name(low, high):<13} {len(group):>10}  {generated:>7}  {counted:>6}  {counted / len(group):>10.0%}")

    seconds = [t["segundos"] for t in tries if "segundos" in t]
    if seconds:
        print(f"segundos por chamada: mediana {statistics.median(seconds):.1f} | "
              f"média {statistics.mean(seconds):.1f} | máx {max(seconds):.1f}")

    incomplete = [p for p in data["projetos"] if p["perguntas"] < data["config"]["por_projeto"]]
    for p in incomplete:
        print(f"incompleto: {p['projeto']} | perguntas {p['perguntas']} | "
              f"tentativas {len(p['tentativas'])} | passagens {p['passagens']} | erro {p['erro']}")

    if sample_size:
        counted = [q for q in data["perguntas"] if q["conta"]]
        sample = random.Random(42).sample(counted, min(sample_size, len(counted)))
        print(f"\n--- amostra de {len(sample)} perguntas que contam ---")
        for i, q in enumerate(sample, 1):
            print(f"\n[{i}] {q['projeto']} ({q['tipo']})")
            print(f"P: {q['pergunta']}")
            print(f"R: {q['resposta']}")
            print(f"passagem ({len(q['passagem'].split())} palavras): {q['passagem']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("saidas", nargs="+")
    parser.add_argument("--amostra", type=int, default=0)
    args = parser.parse_args()
    for path in args.saidas:
        summarize(path, args.amostra)


if __name__ == "__main__":
    main()
