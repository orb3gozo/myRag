# myRag design plan

myRag is a personal project to learn, demonstrate and experiment with retrieval-augmented generation (RAG) on local large language models (LLMs). It runs on a machine with little compute.

> This plan records the decisions agreed on 2026-11-12. It is the source of truth: when a decision changes, update it here.

---

## 1. Goals

The project grows in three phases, each built on the previous one.

1. **Learn.** Understand each part of a RAG pipeline: ingestion, chunking, embeddings, retrieval, prompting and citations.
2. **Demo.** A web interface shows the pipeline at work, with cited sources and a streamed answer.
3. **Lab.** Compare configurations with reproducible metrics. Candidates are chunk size, embedding model, LLM, k, reranking, and LlamaIndex against LangChain.

## 2. Environment constraints

| Resource | Value | Consequence |
|---|---|---|
| CPU | Intel i7-1260P, 16 threads | Everything runs on the CPU |
| GPU | Intel Iris Xe (integrated) | No CUDA acceleration |
| RAM | 15 GB total, 3 to 4 GB free in normal use | Small models, no PyTorch |
| Python | 3.14 with `uv` | All dependencies resolve with `manylinux_2_28` wheels (checked) |
| Deno | 2.7.4 | Frontend |
| LLM server | LM Studio installed (`~/.lmstudio/bin/lms`), Qwen3-8B downloaded | Ollama is optional |

The guiding rule is to pick the cheapest option in compute that still gives reasonable results.

## 3. Design decisions

### 3.1 Architecture

```
┌────────────────────┐   HTTP + SSE   ┌──────────────────────────┐  OpenAI-compatible  ┌──────────────────┐
│ frontend/ (Deno,   │ ─────────────▶ │ backend/ (Python,        │ ──────────────────▶ │ LM Studio /      │
│ Fresh)             │ ◀───────────── │ FastAPI + RagEngine)     │ ◀────────────────── │ Ollama (local)   │
└────────────────────┘                └──────────┬───────────────┘                     └──────────────────┘
                                                 │
                                       ┌─────────┴─────────┐
                                       │ LanceDB (disk)    │  embeddings: fastembed (in-process)
                                       └───────────────────┘              or an OpenAI-compatible server
```

Python is the backend. It holds the RAG logic, the command-line interface (CLI) and the API. Deno is the frontend and only calls the backend API.

### 3.2 RAG engine

The project uses **LlamaIndex** from the start, with no hand-written pipeline.

A thin `RagEngine` contract lets a **LangChain** engine arrive later without changes to the CLI, the API or the evaluation:

```python
class RagEngine(Protocol):
    def ingest(self) -> IngestReport: ...
    def retrieve(self, question: str) -> list[Source]: ...
    def query(self, question: str) -> RagAnswer: ...                # answer and sources
    def query_stream(self, question: str) -> Iterator[RagEvent]: ... # phase 2: sources, tokens, done
```

`retrieve` exists so evaluation can measure retrieval without calling the LLM. `LlamaIndexEngine` is the phase 1 implementation and `LangChainEngine` comes later. `RAG_ENGINE` selects one. The contract stays a single thin layer.

### 3.3 LLM

The LLM sits behind any server with an OpenAI-compatible API, set by `LLM_BASE_URL`:

- LM Studio (default): `http://localhost:1234/v1`, started with `lms server start`.
- Ollama: `http://localhost:11434/v1`.

The default model is **Qwen3-1.7B** at Q4 quantisation. It needs about 1.2 GB of RAM and handles Spanish and English. Qwen3 writes a reasoning block unless the prompt contains `/no_think`, so the project turns it off to cut latency. The Qwen3-8B model already on disk stays available for higher-quality demos. The client is `llama-index-llms-openai-like`.

### 3.4 Embeddings

`EMBED_PROVIDER` selects one of two providers:

| Provider | How | Trade-off |
|---|---|---|
| `fastembed` (default) | In-process, ONNX, no PyTorch | Light, needs no server |
| `openai` | `/v1/embeddings` on LM Studio or Ollama | Lighter Python process, but the server holds two models |

The model must be small and multilingual:

- Default: `intfloat/multilingual-e5-small` (384 dimensions, about 118M parameters, 512 tokens at most). fastembed 0.9.1 doesn't ship it, so the project registers it with `TextEmbedding.add_custom_model` from the official repository's fp32 ONNX export (`onnx/model.onnx`, about 470 MB), with mean pooling and normalisation. The registration was checked on the target laptop: it embeds to 384 dimensions with a peak of about 1 GB of RSS.
- e5 models need a `query: ` prefix on queries and `passage: ` on documents. The factory adds both for any `e5-` or `multilingual-e5-` model, whichever provider serves it, and leaves other models (and the `-instruct` variants, which use a different template) untouched.
- Fallback built into fastembed: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). It was trained on sequences of about 128 tokens, so it would need smaller chunks.

### 3.5 Vector store

The vector store is **LanceDB**, embedded on disk with no server, through `llama-index-vector-stores-lancedb`. `langchain-community` also supports LanceDB, so a future LangChain engine can read the same tables.

Each configuration gets its own table, named `{embedding_model}__{chunk_size}_{overlap}`, so experiments don't mix. Tables live in `backend/storage/lancedb/`, outside git.

### 3.6 Chunking

`MarkdownNodeParser` splits on headings, then `SentenceSplitter` caps the size. The defaults are **400 tokens** with **50 tokens of overlap**, both configurable. Each chunk keeps its file path, its heading path (section) and its language as metadata.

### 3.7 Retrieval

The default is **k = 4**, about 1,600 tokens of context, which a CPU handles in reasonable time. k is configurable. Phase 1 has no reranking; reranking is a phase 3 experiment.

### 3.8 Prompt and citations

The prompt sets four rules:

- Answer only from the retrieved context.
- When the context lacks the answer, reply "No lo sé con la documentación disponible." (or the English equivalent).
- Answer in the language of the question.
- Cite chunks as `[1]`, `[2]`, and so on. The output lists each source with its file path and section.

### 3.9 Corpus

The corpus is the **Spanish FastAPI documentation** (`docs/es/docs` in `fastapi/fastapi`), pinned to tag **`0.162.5`**, the latest on 2026-11-12. A script fetches it with a sparse checkout into `data/fastapi-es/`, outside git. The example code the pages include goes into `data/fastapi-docs_src/`.

Only the Spanish version is indexed. Questions can be in **Spanish or English**, relying on the multilingual embeddings for cross-lingual retrieval.

The next corpus will be an **Obsidian** vault, through LlamaIndex's `ObsidianReader`.

### 3.10 Evaluation

`eval/questions.jsonl` holds about 20 questions. Claude drafts them and the user reviews them:

- About 9 in Spanish and 8 in English, each with its **expected source document** and a **reference answer**.
- 3 that the corpus can't answer. The system must reply "I don't know" to these.

One line looks like this:

```json
{"id": "es-001", "lang": "es", "question": "...", "expected_sources": ["tutorial/path-params.md"], "reference_answer": "...", "answerable": true}
```

Phase 1 metrics are deterministic and don't call the LLM: **hit rate@k** and **mean reciprocal rank (MRR)** over retrieval. The user reviews generated answers by hand. Results go to `eval/results/<date>_<config>.json` for comparison. Using the LLM as a judge is an option for phase 3.

### 3.11 Configuration

`pydantic-settings` reads a `.env` file. The repository ships a documented `.env.example`.

| Variable | Default | Description |
|---|---|---|
| `RAG_ENGINE` | `llamaindex` | RAG engine (`llamaindex`, later `langchain`) |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible server |
| `LLM_MODEL` | `qwen3-1.7b` | Model identifier on the server |
| `LLM_API_KEY` | `lm-studio` | Placeholder; local servers don't check it |
| `LLM_TEMPERATURE` | `0.1` | |
| `LLM_NO_THINK` | `true` | Append `/no_think` for Qwen3 |
| `EMBED_PROVIDER` | `fastembed` | `fastembed` or `openai` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | |
| `EMBED_BASE_URL` | `http://localhost:1234/v1` | Only when `EMBED_PROVIDER=openai` |
| `CHUNK_SIZE` | `400` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | |
| `TOP_K` | `4` | |
| `LANCEDB_PATH` | `backend/storage/lancedb` | |
| `CORPUS_DIR` | `data/fastapi-es` | |
| `DOCS_SRC_DIR` | `data/fastapi-docs_src` | Example code that `{* ... *}` include directives point to |

YAML files for experiment sweeps wait for phase 3.

### 3.12 API (phase 2)

The FastAPI backend exposes:

- `POST /query`, which returns the full `{answer, sources}`.
- `POST /query/stream`, which streams server-sent events (SSE): a `sources` event, then `token` events, then `done`.
- `POST /ingest`, `GET /health` and `GET /config`.

CORS is open to the frontend during development.

### 3.13 Frontend (phase 2)

The frontend uses **Deno with Fresh** (Preact, server-side rendering and islands). It offers a chat with a question box, a streamed answer, the list of sources with their sections, and a panel showing the active configuration (model, k, embeddings).

## 4. Repository layout

```
myRag/
├── PLAN.md
├── README.md
├── Justfile
├── .gitignore              # data/, backend/storage/, .env, .venv
├── .env.example
├── backend/
│   ├── pyproject.toml      # uv, requires-python >=3.14
│   ├── src/myrag/
│   │   ├── config.py       # pydantic-settings
│   │   ├── corpus.py       # Markdown loading and preprocessing
│   │   ├── engine/
│   │   │   ├── base.py     # RagEngine, RagAnswer, Source, IngestReport
│   │   │   └── llamaindex_engine.py
│   │   ├── embeddings.py   # fastembed or openai factory
│   │   ├── llm.py          # OpenAI-like factory
│   │   ├── prompts.py
│   │   ├── cli.py          # typer: ingest, ask, eval
│   │   ├── api.py          # FastAPI (phase 2)
│   │   └── evaluation.py   # hit rate, MRR
│   ├── scripts/
│   │   └── fetch_corpus.sh # sparse checkout of FastAPI docs/es at a tag
│   └── tests/
├── frontend/               # Deno Fresh (phase 2)
├── data/                   # downloaded corpus (ignored by git)
└── eval/
    ├── questions.jsonl
    └── results/
```

## 5. Dependencies

Versions resolved on Python 3.14, x86_64 `manylinux_2_28`:

| Package | Version |
|---|---|
| llama-index-core | 0.14.25 |
| llama-index-llms-openai-like | 0.8.1 |
| llama-index-embeddings-fastembed | 0.7.0 |
| llama-index-embeddings-openai-like | 0.4.0 |
| llama-index-vector-stores-lancedb | 0.6.0 |
| lancedb | 0.41.0 |
| fastembed | 0.9.1 |
| onnxruntime | 1.31.1 |
| fastapi | 0.162.5 |
| pydantic-settings | 2.16.2 |
| typer | 0.29.1 |
| langchain-community (later) | 0.4.2 |

## 6. Phases and deliverables

Each phase is split into GitHub issues, and each issue lands through its own pull request.

### Phase 1: terminal MVP

1. Settings and `.env.example`.
2. Corpus loader and Markdown preprocessing.
3. Embedding factory, including the custom e5-small check.
4. LLM factory and prompts.
5. Engine contract and LlamaIndex ingestion into LanceDB.
6. Query with citations.
7. CLI commands `ingest` and `ask`.
8. Evaluation question set.
9. Retrieval metrics and `myrag eval`.
10. README.

Phase 1 is done when `myrag ask` answers in Spanish and English with cited sources, replies "I don't know" to the unanswerable questions, and `myrag eval` prints hit rate@4 and MRR.

### Phase 2: API and web

1. FastAPI with `/query`, `/query/stream` (SSE), `/ingest`, `/health` and `/config`.
2. `query_stream()` in the engine.
3. Deno Fresh frontend with a streamed chat and sources.

Phase 2 is done when a question asked in the browser shows tokens as they arrive and lists its sources.

### Phase 3: lab

- Configuration sweeps from YAML: chunk size and overlap, embedding models, k, and the 1.7B LLM against the 8B one.
- Cross-lingual experiment: does retrieval get worse with English questions?
- Reranking.
- A `LangChainEngine` that shares LanceDB, compared with LlamaIndex.
- Optional LLM-as-judge.
- Ingestion of an Obsidian vault.

## 7. Open risks

| Risk | Mitigation |
|---|---|
| fastembed doesn't ship `multilingual-e5-small` | Resolved: registered with `add_custom_model` from the official ONNX export. `paraphrase-multilingual-MiniLM-L12-v2` with smaller chunks stays as the fallback |
| Tight RAM with a browser, LM Studio and the backend running | 1.7B model, ONNX embeddings, k = 4 |
| CPU latency | `/no_think`, SSE streaming, bounded context |
| The Spanish FastAPI translation is incomplete | Evaluation questions use only pages that exist in Spanish |
| Python 3.14 is very recent | Resolution checked; versions pinned in `uv.lock` |
