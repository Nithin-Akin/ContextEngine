# ContextEngine Architecture

## Basic RAG

```text
Document → Chunk → Embed
                    │
Query → Embed ──────┘
        ↓
Cosine Similarity
        ↓
Top-k
        ↓
Context
        ↓
LLM
```

Basic RAG uses a bi-encoder retrieval pattern. Chunk embeddings are computed independently and can be cached; only the query needs to be embedded at runtime.

## Advanced RAG

```text
                       Query
                    ┌────┴────┐
                    ↓         ↓
                 Dense       BM25
                 Search      Search
                    ↓         ↓
                    └────┬────┘
                         ↓
                        RRF
                         ↓
                    Candidates
                         ↓
                  Cross-Encoder
                         ↓
                    Final Top-k
                         ↓
                      Context
                         ↓
                        LLM
```

## Candidate generation vs candidate refinement

Dense retrieval and BM25 are fast enough to generate a broad candidate set.

The cross-encoder is more expensive because it evaluates the query and each candidate jointly. It is therefore used after fusion on a reduced candidate set.

```text
candidate generation → candidate fusion → candidate refinement → context construction
```

## Retrieval signals

### Dense semantic retrieval
Captures semantic similarity even when query and document wording differ.

### BM25 lexical retrieval
Captures exact terms, identifiers, and rare lexical matches.

### Reciprocal Rank Fusion
Combines ranked lists without requiring score normalization.

### Cross-encoder reranking
Jointly models query-document interactions and produces a more precise relevance ordering.

## Current scope

The project deliberately uses a small corpus and explicit Python code so every retrieval stage can be inspected.

Current non-goals:
- persistent vector storage
- production ingestion services
- multi-user infrastructure
- large-scale automated evaluation
- frontend application
