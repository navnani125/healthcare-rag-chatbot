"""
retriever.py
------------
Given a user query, embed it and fetch the top-k most relevant
chunks from ChromaDB using cosine similarity.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import Optional

from embeddings import (
    EMBED_MODEL,
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5


# ── Retriever class ───────────────────────────────────────────────────────────
class MedicalRetriever:
    """
    Wraps ChromaDB + sentence-transformers for semantic retrieval.
    Instantiate once and reuse across queries (model stays in memory).
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k  = top_k
        self.model  = load_embedding_model()
        self.client = get_chroma_client()
        self.col    = get_or_create_collection(self.client)

        count = self.col.count()
        if count == 0:
            raise RuntimeError(
                "Vector store is empty. Run the following first:\n"
                "  python src/ingest.py\n"
                "  python src/embeddings.py"
            )
        print(f"Retriever ready — {count} vectors loaded.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list:
        """
        Embed the query and return top-k matching chunks.

        Returns:
            list of {
                "text":     str,
                "source":   str,
                "score":    float,   # cosine distance (lower = more similar)
                "metadata": dict,
            }
        """
        k = top_k or self.top_k

        query_embedding = self.model.encode([query]).tolist()

        results = self.col.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "score":    round(1 - dist, 4),   # convert distance → similarity
                "metadata": meta,
            })

        return hits

    def format_context(self, hits: list[dict]) -> str:
        """
        Format retrieved chunks into a numbered context block
        ready to be injected into the LLM prompt.
        """
        parts = []
        for i, hit in enumerate(hits, 1):
            parts.append(
                f"[Source {i} | {hit['source'].upper()} | "
                f"relevance: {hit['score']:.2f}]\n{hit['text']}"
            )
        return "\n\n---\n\n".join(parts)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = MedicalRetriever(top_k=3)

    test_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes Alzheimer's disease?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        hits = retriever.retrieve(q)
        for h in hits:
            print(f"  [{h['source']} | {h['score']}] {h['text'][:120]}...")
