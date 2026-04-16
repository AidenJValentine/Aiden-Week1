# Week 1 Report

## 1. Understanding the Current Pipeline

### How the agent works

The existing system is an agentic LangGraph workflow, not a RAG question-answering flow yet.

- Entry point: `main.py`
- Pipeline orchestration: `src/pipeline/graph_builder.py`
- Agent logic and tools: `src/agents/agentic_agent.py`

Flow:

1. `run_pipeline()` resets Neo4j unless incremental mode is requested.
2. `run_agent()` builds a LangGraph with two nodes: `agent` and `tools`.
3. The `agent` node calls `ChatOpenAI(...).bind_tools(TOOLS)`.
4. If the model returns tool calls, LangGraph routes to `ToolNode(TOOLS)`.
5. Tool outputs are appended to the message history and the graph loops back to `agent`.
6. The run ends when there are no more tool calls, `finish_research()` is called, or the iteration limit is reached.
7. The collected research is written to Neo4j and summary JSON files are saved for Streamlit.

This means the current codebase is optimized for autonomous research and structured data extraction, with ChromaDB used as evidence storage and Neo4j used as the knowledge graph.

### What tools are available

The LLM can choose from the following tools in `TOOLS` inside `src/agents/agentic_agent.py`:

- `search_web`
- `extract_page_content`
- `save_competitor`
- `save_product`
- `get_current_progress`
- `research_industry_needs`
- `map_needs_from_report`
- `research_customer_segments`
- `map_segments_to_products`
- `generate_house_of_quality`
- `finish_research`

These tools cover web search, content extraction, evidence storage, product/spec collection, customer segment research, industry-needs analysis, and QFD/House-of-Quality generation.

### How data is stored

The system uses two storage layers plus JSON exports:

- ChromaDB: `src/pipeline/chroma_store.py`
  - Stores raw evidence chunks from extracted web pages.
  - Uses persistent local storage under `./chroma_db`.
  - Chunks are stored with metadata such as `source_url`, `query`, `page_title`, and chunk position.
- Neo4j: `src/pipeline/graph_builder.py`
  - Stores structured entities and relationships like `Company`, `Product`, `Specification`, `CustomerNeed`, and `CustomerSegment`.
  - Relationship properties include `source_urls` and `evidence_ids` for traceability.
- JSON exports:
  - `industry_report.json`
  - `customer_segments.json`
  - `house_of_quality.json`

These JSON files are used by the Streamlit app and also became the local seed corpus for the basic RAG pipeline I added.

## 2. Basic RAG Pipeline Added

### New code

- RAG module: `src/pipeline/basic_rag.py`
- Demo runner: `demo_basic_rag.py`

### What it does

The new basic RAG pipeline follows the requested flow:

1. User question
2. Retrieve relevant chunks from ChromaDB
3. Pass chunks + question to the LLM
4. Return a grounded answer

Implementation details:

- `bootstrap_local_rag_collection()` builds a local Chroma collection named `basic_rag_artifacts`.
- It ingests the checked-in JSON artifacts into chunked documents so the pipeline can work even when the original web-scraped `chroma_db` is not present.
- `retrieve_relevant_chunks(question)` queries that collection using local hash-based embeddings stored in Chroma.
- `answer_question(question)` retrieves chunks and then:
  - uses `ChatOpenAI` when `OPENAI_API_KEY` is available
  - otherwise falls back to deterministic sentence extraction from the retrieved evidence

### Example function call

```python
from src.pipeline.basic_rag import answer_question

result = answer_question("What are the key specs of the 3051S?")
print(result["answer"])
print(result["sources"])
```

## 3. Testing Notes

I tested with the requested sample questions through `demo_basic_rag.py`:

- `What competitors does Honeywell have?`
- `What are the key specs of the 3051S?`
- `What customer needs exist in oil and gas?`

Current test status:

- A file of the results of the example questions included in the deliverables file tot serve as pictures of the output.

## 4. Deliverables Included

- Code integrated into the repo for a callable Python RAG function
- Demo script for the sample questions
- This report


