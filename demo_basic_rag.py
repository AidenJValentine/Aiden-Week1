from __future__ import annotations

import json

from src.pipeline.basic_rag import answer_question, bootstrap_local_rag_collection


QUESTIONS = [
    "What competitors does Honeywell have?",
    "What are the key specs of the 3051S?",
    "What customer needs exist in oil and gas?",
]


def main():
    bootstrap_info = bootstrap_local_rag_collection(force_rebuild=True)
    print("BOOTSTRAP")
    print(json.dumps(bootstrap_info, indent=2))
    print()

    for question in QUESTIONS:
        result = answer_question(question, top_k=5)
        print("=" * 100)
        print(f"QUESTION: {question}")
        print(f"LLM_USED: {result['llm_used']}")
        print("ANSWER:")
        print(result["answer"])
        print("SOURCES:")
        for source in result["sources"][:5]:
            print(
                f"- {source['chunk_id']} | {source['source_file']} | "
                f"{source['doc_type']} | {source['source_url'] or 'n/a'} | score={source['score']}"
            )
        print()


if __name__ == "__main__":
    main()
