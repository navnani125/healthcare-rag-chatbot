"""
embeddings.py
-------------
Embeds text chunks using sentence-transformers (free, runs locally)
and stores them in ChromaDB (local vector database, no API key needed).
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # fast, free, 384-dim
COLLECTION    = "healthcare_rag"
BATCH_SIZE    = 64
DATA_DIR      = Path(__file__).parent.parent / "data"
CHROMA_DIR    = DATA_DIR / "chroma_db"


# ── ChromaDB client ───────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client (data saved to disk)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get existing collection or create a fresh one."""
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )


# ── Embedding ─────────────────────────────────────────────────────────────────
def load_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def embed_and_store(
    chunks: list[dict],
    model: Optional[SentenceTransformer] = None,
    reset_collection: bool = False,
) -> chromadb.Collection:
    """
    Embed all chunks and store them in ChromaDB.

    Args:
        chunks:           list of {"text": ..., "source": ..., "metadata": ...}
        model:            optional pre-loaded SentenceTransformer
        reset_collection: if True, wipe existing data and re-embed from scratch

    Returns:
        The populated ChromaDB collection
    """
    if model is None:
        model = load_embedding_model()

    client     = get_chroma_client()

    if reset_collection:
        try:
            client.delete_collection(COLLECTION)
            print(f"Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass

    collection = get_or_create_collection(client)

    # Skip if already populated and not resetting
    existing = collection.count()
    if existing > 0 and not reset_collection:
        print(f"Collection already has {existing} vectors. Skipping embedding.")
        print("  Pass reset_collection=True to re-embed from scratch.")
        return collection

    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts     = [c["text"]   for c in batch]
        sources   = [c["source"] for c in batch]
        metadatas = [c.get("metadata", {}) | {"source": src}
                     for c, src in zip(batch, sources)]
        ids       = [f"chunk_{batch_start + i}" for i in range(len(batch))]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"\n✅ Stored {collection.count()} vectors in ChromaDB")
    print(f"   Location: {CHROMA_DIR}\n")

    return collection


# ── Load from cache ───────────────────────────────────────────────────────────
def build_from_cache(reset: bool = False) -> chromadb.Collection:
    """
    Load chunks from data/chunks.json (created by ingest.py) and embed them.
    Fastest way to rebuild the vector store after ingestion.
    """
    cache_path = DATA_DIR / "chunks.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No chunk cache found at {cache_path}. "
            "Run ingest.py first: python src/ingest.py"
        )

    print(f"Loading chunks from cache: {cache_path}")
    with open(cache_path) as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded")

    model = load_embedding_model()
    return embed_and_store(chunks, model=model, reset_collection=reset)


if __name__ == "__main__":
    # Run standalone: python src/embeddings.py
    collection = build_from_cache(reset=False)
    print(f"Collection '{COLLECTION}' ready with {collection.count()} vectors.")
