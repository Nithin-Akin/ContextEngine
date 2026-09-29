# ContextEngine

**Tagline:** Modular hybrid-retrieval RAG stack for high-precision context construction and grounded generation.

## Core capabilities

- Dense semantic retrieval
- BM25 sparse lexical retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Grounded LLM generation
- Retrieval inspection and evaluation

## Design principle

ContextEngine separates **candidate generation** from **candidate refinement**.

Fast bi-encoder and BM25 retrieval create a broad candidate set. Reciprocal Rank Fusion combines independent retrieval signals, and a cross-encoder then performs higher-precision pairwise relevance scoring before the final context is assembled.

## Current scope

The current implementation is notebook-first and intentionally transparent. Every major retrieval stage is exposed so the system can be inspected and reasoned about end-to-end.
