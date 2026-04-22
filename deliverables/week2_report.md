# Week 2 Report

## Overview

This week adds an interactive chat interface to the Streamlit dashboard and upgrades retrieval from a basic Chroma-only flow to a hybrid retrieval flow that uses both Neo4j and ChromaDB.

The goal was to let a user ask natural-language questions such as:

- `What competitors does Honeywell have?`
- `Compare Rosemount 3051S with SmartLine ST700`
- `What customer needs show up in oil and gas?`

The application now answers these questions by combining:

- Structured facts from Neo4j
- Supporting evidence chunks from ChromaDB

## Chat Interface

The new chat interface is available as the `💬 Chat` tab in the Streamlit app.

Key features:

- Natural-language question input using Streamlit chat components
- Persistent chat history stored in `st.session_state`
- Example prompts for common question types
- Answer display with explicit source sections

The answer area is followed by two citation panels:

1. `Neo4j Structured Data`
2. `ChromaDB Evidence`

This makes it clear which part of the answer came from graph data and which part came from source-text evidence.

## Hybrid Retrieval Design

The hybrid retrieval logic lives in `src/pipeline/hybrid_rag.py`.

The flow is:

1. Classify the user question
2. Retrieve relevant structured records from Neo4j
3. Retrieve supporting evidence chunks from ChromaDB
4. Synthesize a grounded answer
5. Return both the answer and the underlying citations

### Neo4j Retrieval

Neo4j is used for structured questions such as:

- product comparisons
- competitor lookup
- product specifications
- customer needs
- customer segments

For example, when the user asks to compare two products, the system retrieves:

- company
- product name
- technical specifications
- linked customer needs
- linked customer segments

### ChromaDB Retrieval

ChromaDB is used for supporting evidence.

The system first tries to use evidence directly linked from Neo4j records through `evidence_ids`. If that does not provide enough support, it falls back to semantic search in ChromaDB using the user question.

This gives the chat flow both:

- traceable graph facts
- human-readable source text

## Source Citation Strategy

Answers cite sources in two forms:

- `Neo4j S1`, `Neo4j S2`, etc. for structured graph records
- `Chroma C1`, `Chroma C2`, etc. for evidence chunks

The UI also displays the detailed data behind each citation:

- Neo4j product/spec tables and linked entities
- Chroma chunk IDs, source URLs, and text snippets

## Files Added or Updated

- `src/pipeline/hybrid_rag.py`
- `src/pipeline/__init__.py`
- `streamlit_app.py`

## Validation

Code was syntax-checked with:

```bash
python -m py_compile streamlit_app.py src/pipeline/hybrid_rag.py src/pipeline/__init__.py
```

## Screenshots

Add screenshots here before submission:

1. Chat answering a competitor question
2. Chat answering a product comparison question
3. Chat answering a customer-needs question
