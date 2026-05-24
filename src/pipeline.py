"""
pipeline.py
-----------
The full RAG pipeline: retrieve relevant chunks → generate answer.
This is the single entry point used by the Streamlit app.
"""

from retriever import MedicalRetriever
from llm import MedicalLLM


class HealthcareRAGPipeline:
    """
    End-to-end RAG pipeline for medical Q&A.

    Usage:
        pipeline = HealthcareRAGPipeline()
        answer   = pipeline.ask("What are the symptoms of diabetes?")
        print(answer)
    """

    def __init__(self, top_k: int = 5):
        print("\n=== Initialising Healthcare RAG Pipeline ===")
        self.retriever    = MedicalRetriever(top_k=top_k)
        self.llm          = MedicalLLM()
        self.chat_history = []        # stores conversation for multi-turn chat
        print("Pipeline ready ✅\n")

    def ask(self, query: str, use_history: bool = True) -> dict:
        """
        Full RAG query.

        Args:
            query:       user's question
            use_history: include previous turns for follow-up questions

        Returns:
            {
              "answer":  str,          # LLM response
              "sources": list[dict],   # retrieved chunks with scores
              "context": str,          # formatted context sent to LLM
            }
        """
        # 1. Retrieve
        hits    = self.retriever.retrieve(query)
        context = self.retriever.format_context(hits)

        # 2. Generate
        if use_history and self.chat_history:
            answer = self.llm.generate_with_history(query, context, self.chat_history)
        else:
            answer = self.llm.generate(query, context)

        # 3. Update history
        self.chat_history.append({"role": "user",      "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer":  answer,
            "sources": hits,
            "context": context,
        }

    def reset_history(self):
        """Start a fresh conversation."""
        self.chat_history = []
        print("Conversation history cleared.")


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = HealthcareRAGPipeline(top_k=5)

    questions = [
        "What are the symptoms of hypertension?",
        "How is it usually treated?",       # follow-up — uses history
        "What foods should I avoid?",        # follow-up
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer']}")
        print(f"\n📚 Sources used:")
        for s in result["sources"]:
            print(f"   [{s['source']} | {s['score']}] {s['text'][:80]}...")
        print("-" * 60)
