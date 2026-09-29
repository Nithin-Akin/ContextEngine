<div align="center">

# ContextEngine

### Modular hybrid-retrieval RAG stack for high-precision context construction and grounded generation

**ContextEngine** integrates dense semantic retrieval, BM25 lexical search, Reciprocal Rank Fusion, and pairwise cross-encoder relevance scoring to improve evidence selection before LLM inference.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-purple)](https://www.sbert.net/)
[![BM25](https://img.shields.io/badge/Retrieval-BM25-green)](https://en.wikipedia.org/wiki/Okapi_BM25)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google)](https://ai.google.dev/)

</div>

---

## Overview

ContextEngine is an implementation-first Retrieval-Augmented Generation project that exposes the retrieval stack instead of hiding it behind a high-level framework.

It implements two complete pipelines:

### Basic RAG

```text
Query
  ↓
Embedding
  ↓
Cosine Similarity
  ↓
Top-k Chunks
  ↓
Context
  ↓
LLM
  ↓
Answer
```

### Advanced RAG

```text
                     ┌→ Dense Semantic Search ─┐
Query ───────────────┤                          ├→ RRF → Cross-Encoder → Top-k → Context → LLM → Answer
                     └→ BM25 Lexical Search ───┘
```

The core engineering idea is that generation quality is bounded by retrieval quality. ContextEngine therefore separates **candidate generation** from **candidate refinement**.

---

## Why ContextEngine?

A basic vector-search RAG pipeline can fail when:

- the answer uses different wording from the query,
- exact identifiers or keywords matter,
- semantically similar chunks are not actually answer-bearing,
- several candidates occupy similar regions in embedding space,
- retrieval noise grows with corpus size.

ContextEngine addresses those failure modes with a hybrid retrieval and reranking architecture.

---

## Core Capabilities

### Basic RAG
- document upload and UTF-8 decoding
- overlapping word-based chunking
- dense sentence embeddings using `all-MiniLM-L6-v2`
- manual cosine similarity
- semantic top-k retrieval
- grounded prompt construction
- Gemini-based generation

### Advanced RAG
- BM25 sparse lexical retrieval
- hybrid dense + sparse retrieval
- Reciprocal Rank Fusion
- cross-encoder reranking
- final top-k context construction
- transparent inspection of every retrieval stage
- retrieval relevance / faithfulness / correctness checks

---

## Architecture

```text
                              DOCUMENT
                                 │
                                 ▼
                         Read + Decode Text
                                 │
                                 ▼
                         Chunk with Overlap
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
          Dense Embeddings                BM25 Token Index
                  │                             │
USER QUERY ───────┼──────────────┬──────────────┤
                  │              │              │
                  ▼              │              ▼
           Query Embedding       │       Tokenized Query
                  │              │              │
                  ▼              │              ▼
        Cosine Similarity        │         BM25 Scores
                  │              │              │
                  └───────┬──────┴───────┬─────┘
                          ▼              ▼
                     Ranked Candidate Lists
                              │
                              ▼
                    Reciprocal Rank Fusion
                              │
                              ▼
                      Fused Candidates
                              │
                              ▼
                    Query + Chunk Pairs
                              │
                              ▼
                   Cross-Encoder Reranker
                              │
                              ▼
                        Final Top-k
                              │
                              ▼
                    Retrieved Context
                              │
                              ▼
                      Grounded Prompt
                              │
                              ▼
                             LLM
                              │
                              ▼
                           ANSWER
```

---

## Retrieval Pipeline

### 1. Chunking

The document is split into overlapping chunks:

```python
chunk_size = 40
overlap = 10
```

Overlap reduces information loss near chunk boundaries.

### 2. Dense Semantic Retrieval

Every chunk is embedded once. A runtime query is embedded into the same vector space and compared using cosine similarity.

```text
query vector ↔ chunk vectors → semantic ranking
```

### 3. BM25 Lexical Retrieval

BM25 provides an independent sparse signal based on exact term relevance.

```text
tokenized query ↔ tokenized chunks → BM25 ranking
```

### 4. Reciprocal Rank Fusion

Dense and BM25 raw scores are not directly comparable. RRF combines rank positions instead:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

This rewards candidates that rank consistently across retrieval systems.

### 5. Cross-Encoder Reranking

The strongest fused candidates are evaluated as full `(query, chunk)` pairs.

The project uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This stage is slower than bi-encoder retrieval but much more precise, so it is applied only after candidate generation.

### 6. Grounded Generation

The final top-ranked chunks are assembled into a context block and passed to Gemini with strict grounding instructions.

---

## Basic vs Advanced RAG

| Capability | Basic RAG | ContextEngine Advanced |
|---|---:|---:|
| Dense semantic retrieval | ✅ | ✅ |
| BM25 lexical retrieval | ❌ | ✅ |
| Hybrid search | ❌ | ✅ |
| Rank fusion | ❌ | ✅ RRF |
| Pairwise relevance scoring | ❌ | ✅ Cross-encoder |
| Grounded generation | ✅ | ✅ |
| Retrieval inspection | ✅ | ✅ |
| Evaluation checklist | ✅ | ✅ |

---

## Project Structure

```text
ContextEngine/
├── README.md
├── PROJECT.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── contextengine_rag_pipeline.ipynb
├── data/
│   └── sample_company_policy.txt
└── docs/
    └── architecture.md
```

---

## Tech Stack

- Python
- Google Colab / Jupyter
- Sentence Transformers
- NumPy
- BM25 / rank_bm25
- CrossEncoder
- Google Gemini API

---

## Getting Started

### Clone

```bash
git clone https://github.com/Nithin-Akin/ContextEngine.git
cd ContextEngine
```

### Install

```bash
pip install -r requirements.txt
```

### Add Gemini API key

In Google Colab:

```text
Secrets → GEMINI_API_KEY
```

The notebook loads it securely with:

```python
from google.colab import userdata
api_key = userdata.get("GEMINI_API_KEY")
```

Never hardcode API keys into the notebook.

### Run

Open:

```text
notebooks/contextengine_rag_pipeline.ipynb
```

Run cells from top to bottom and upload either your own document or the sample policy file in `data/`.

---

## Example

**Query**

```text
How long do enterprise customers have to request a refund?
```

**Grounded answer**

```text
Enterprise customers have a 60-day refund window.
```

---

## Evaluation

The notebook checks three distinct failure surfaces:

**Retrieval relevance** — Did the system retrieve the evidence required to answer?

**Faithfulness** — Is the generated answer supported by retrieved context?

**Answer correctness** — Does the final answer match the expected answer?

For larger datasets these can be automated with Recall@k and benchmarked answer/faithfulness metrics.

---

## Roadmap

- [ ] Parent-child chunking
- [ ] Metadata filtering
- [ ] Query rewriting
- [ ] HyDE
- [ ] Persistent vector database
- [ ] Multi-document ingestion
- [ ] Automated Recall@k benchmark
- [ ] Automated faithfulness evaluation
- [ ] Web interface
- [ ] Agentic / multi-hop retrieval

---

## Engineering Focus

ContextEngine is intentionally transparent: every retrieval stage is inspectable, from tokenization and embeddings through rank fusion and reranking. The repository is designed to make retrieval quality measurable rather than treating RAG as a black box.
