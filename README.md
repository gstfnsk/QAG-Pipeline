# Geração de perguntas e respostas a partir de projetos de pesquisa

Código do TCC (UFRGS) que constrói um dataset de perguntas e respostas a partir de um corpus de projetos de pesquisa, para avaliar sistemas de busca e RAG.

**O corpus é confidencial e não está neste repositório.** Pelo mesmo motivo, os exemplos few-shot usados nos prompts foram substituídos por exemplos **fictícios** no mesmo formato (pasta `examples/`). Com os exemplos originais, os scripts geram exatamente a saída do trabalho.

Todos os modelos rodam **localmente** com o [Ollama](https://ollama.com). Os scripts recusam URLs que não sejam `localhost`, então nenhum texto do corpus sai da máquina.

## Etapas

| # | Script | Entrada → saída |
|---|---|---|
| 1 | `clean_corpus.py` | corpus original → corpus só com o código do projeto e os campos de texto, sem palavras-chave e títulos |
| 2 | `run_batches.py` + `select_passages.py` | corpus limpo → passagens selecionadas pelo LLM, em lotes de 100 projetos |
| 3 | `verify_passages.py`, `processing_stats.py` | confere que toda passagem está no texto original; estatísticas de tempo, chamadas e saída |
| 4 | `merge_passages.py` | lotes → `{código do projeto: [passagens]}` |
| 5 | `filter_passages.py` | remove passagens curtas que só nomeiam uma ação ("Avaliação de...", "Avaliar...", "Serão avaliados...") |
| 6 | `generate_qa.py` | passagens filtradas → até 2 perguntas e respostas por projeto, com evidência e checagens automáticas |
| 7 | `analyze_qa.py` | resume e compara saídas da etapa 6 |

`passage_results.py` reúne as funções de leitura usadas nas etapas 3 e 4.

## Como rodar

```bash
# 1. Limpeza
python clean_corpus.py corpus.json > corpus_clean.json

# 2. Seleção de passagens (gera corpus_clean.batch001.json, batch002, ...)
python run_batches.py corpus_clean.json --examples examples/selection_examples_fictional.txt
#    Campos que falharam podem ser refeitos com chunks menores:
python select_passages.py corpus_clean.json --examples examples/selection_examples_fictional.txt \
    --projects <projeto>.json --fields <campo> --chunk-chars 10000 > corpus_clean.retry_failed.json

# 3. Verificação e estatísticas
python verify_passages.py corpus_clean.json > verification.json
python processing_stats.py corpus_clean.json > stats.json

# 4. Junção
python merge_passages.py corpus_clean.json > passages.json

# 5. Filtro (N = 10 palavras)
python filter_passages.py passages.json --max-palavras 10 \
    --removidas passages_removed_N10.json > passages_filtered_N10.json

# 6. Perguntas e respostas
python generate_qa.py passages_filtered_N10.json --exemplos examples/qa_examples_fictional.json > qa.json

# 7. Resumo
python analyze_qa.py qa.json --amostra 20
```

Os scripts nunca alteram os arquivos de entrada. Os resultados vão para a saída padrão (`>`), e o progresso aparece no terminal. As etapas 2 e 6 salvam o progresso: rodar o mesmo comando de novo continua de onde parou.

## Configuração padrão

- Modelo: `gpt-oss:20b` (Ollama), temperatura 0, seed 42, raciocínio `low`.
- Contexto: 32.768 tokens na seleção e 16.384 na geração de perguntas.
- Filtro: spaCy `pt_core_news_md`.

A configuração de cada execução fica registrada no próprio arquivo de saída.

## Requisitos

- Python 3.9 ou mais recente.
- Ollama com o modelo instalado: `ollama pull gpt-oss:20b`.
- Etapas 5 e 6: `pip install -r requirements.txt` e `python -m spacy download pt_core_news_md`.

## Formato dos exemplos

- `examples/selection_examples_fictional.txt`: texto inserido no prompt de seleção, antes do formato de saída.
- `examples/qa_examples_fictional.json`: `positivos` (passagem, tipo, pergunta, resposta, evidência literal, justificativa) e `negativos` (passagem, justificativa). Os projetos dos exemplos ficam de fora da geração.
