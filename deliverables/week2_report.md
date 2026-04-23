# Week 2 Report

## 1. Overview

### What changed from Week 1

Week 1 delivered a basic RAG pipeline (`src/pipeline/basic_rag.py`) that answered questions by querying a local ChromaDB collection seeded from JSON artifacts. It worked as a standalone retrieval function but had no interactive interface and operated independently of the Neo4j knowledge graph.

Week 2 transforms the system into a fully interactive hybrid chat interface embedded in the existing Streamlit dashboard. The core addition is `src/pipeline/hybrid_rag.py`, which combines structured retrieval from Neo4j with evidence-based retrieval from ChromaDB before passing both to an LLM for synthesis.

### New files and modules

- Hybrid retrieval module: `src/pipeline/hybrid_rag.py`
- Chat UI functions in `streamlit_app.py`: `render_chat_tab()`, `render_hybrid_sources()`

## 2. Hybrid RAG Architecture

### How it works

The `answer_hybrid_question(question)` entry point drives the full pipeline:

1. `_classify_question(question)` — routes the question into one of six types: `comparison`, `customer_needs`, `customer_segments`, `competitors`, `product_specs`, or `general`.
2. `_retrieve_neo4j_context(question)` — runs the appropriate Cypher query against the Neo4j knowledge graph to fetch structured records (products, specs, competitors, needs, or segments).
3. `_retrieve_chroma_context(question, neo4j_rows)` — first retrieves ChromaDB chunks via `evidence_ids` stored on Neo4j relationships (linked evidence), then falls back to semantic search for any remaining slots.
4. `_answer_with_llm(...)` — passes formatted Neo4j context (`[S1]`, `[S2]`, ...) and Chroma context (`[C1]`, `[C2]`, ...) to `gpt-4.1-mini` with an instruction to cite every factual claim inline.
5. Falls back to `_fallback_answer(...)` if the LLM is unavailable.

This gives the system a dual-source foundation: graph relationships for structured facts and source documents for supporting evidence.

### Comparison query improvement

For questions of the form "Compare X vs Y" or "Compare X with Y", a dedicated helper `_extract_comparison_products(question, known_products)` uses regex to split the question on the comparison keyword and independently matches each side against the full product list using token overlap. This ensures both named products are retrieved and compared, rather than relying on aggregate token scoring across the whole question.

### Neo4j resilience

If Neo4j is not configured or unreachable, `answer_hybrid_question` catches the connection error, continues with ChromaDB-only retrieval, and returns a `neo4j_error` field. The UI surfaces this as a warning banner rather than a red error, so the chat remains usable without a live graph database.

### Example function call

```python
from src.pipeline.hybrid_rag import answer_hybrid_question

result = answer_hybrid_question("Compare Rosemount 3051S with SmartLine ST700")
print(result["answer"])
print(result["neo4j_sources"])   # structured graph records
print(result["chroma_sources"])  # evidence chunks with metadata
```

## 3. Chat Interface

### Tab and session state

A new **💬 Chat** tab in `streamlit_app.py` calls `render_chat_tab()`. Session state (`st.session_state.chat_messages`) persists the full conversation history within a session. A **🗑️ Clear** button in the top-right resets the history.

Three example question buttons let users explore the system without having to type a first query.

### Source citations

Every assistant response is followed by two expandable citation panels rendered by `render_hybrid_sources()`:

- **🕸️ Neo4j Structured Data** — one entry per retrieved graph record, labelled `[Neo4j S1]`, `[Neo4j S2]`, etc. Product entries show a spec table, linked customer needs, linked customer segments, and up to three clickable source URLs drawn from the relationship properties stored in the graph.
- **📚 ChromaDB Evidence** — one entry per chunk, labelled `[Chroma C1]`, `[Chroma C2]`, etc. Each entry shows the retrieval method (linked evidence or semantic search), a clickable source URL, and a text preview of the chunk.

The LLM answer text uses the same `[Neo4j S1]` and `[Chroma C2]` notation inline, so every claim in the response can be traced back to a specific panel entry.

### Connection to the Phase 1 pipeline

The chat interface inherits all data collected by the LangGraph agent from Phase 1. The agent stores `source_urls` and `evidence_ids` on every Neo4j relationship it creates, and those identifiers are what `_retrieve_chroma_context` uses to pull the exact chunks that support each graph fact. No re-scraping or re-indexing is needed: the chat operates directly on whatever the agent has already deposited into Neo4j and ChromaDB.

## 4. Testing Notes

Manually tested via `streamlit run streamlit_app.py` → **💬 Chat** tab:

- `"What accuracy does the SmartLine ST700 need to meet?"` — correctly routes to `product_specs`, not `customer_needs`.
- `"Compare Rosemount 3051S with SmartLine ST700"` — returns two product records in the Neo4j panel (S1 and S2).
- `"What competitors does Honeywell have?"` — returns competitor records with linked products.
- Neo4j disconnected: chat shows a yellow warning and returns a ChromaDB-only answer.
- Clear button: resets conversation history.

## 5. Known Limitations

- Answer quality is bounded by what Phase 1 collected. If a product, customer need, or evidence chunk was not captured during the research pipeline run, the chat cannot surface it.
- Question classification uses keyword matching. Questions that do not contain the expected keywords (e.g., "What is the output signal range?") fall back to the `general` route, which may return less targeted context than a `product_specs` route would.
- Comparison queries resolve product names by token overlap. Highly abbreviated or aliased product names may not match correctly if the tokens do not appear in the graph.

## 6. Deliverables Included

- Hybrid retrieval module: `src/pipeline/hybrid_rag.py`
- Chat tab and citation rendering integrated into `streamlit_app.py`
- This report
