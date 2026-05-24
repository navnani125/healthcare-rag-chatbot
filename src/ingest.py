"""
ingest.py
---------
Loads medical data from PubMedQA (HuggingFace) and MedQuAD (GitHub),
cleans the text, splits into overlapping chunks ready for embedding.
"""

import re
import json
import requests
from pathlib import Path
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # tokens per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
DATA_DIR      = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MEDQUAD_URL = (
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/"
    "1_CancerGov_QA/0000output.json"
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, remove non-ASCII junk."""
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"\s+", " ", text)               # extra whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # non-ASCII
    return text.strip()


def chunk_documents(docs: list[dict], source_tag: str) -> list[dict]:
    """
    Split each doc's text into overlapping chunks.
    Returns list of {"text": ..., "source": ..., "metadata": {...}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in docs:
        raw_chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(raw_chunks):
            chunks.append({
                "text": chunk,
                "source": source_tag,
                "metadata": {**doc.get("metadata", {}), "chunk_index": i},
            })
    print(f"  [{source_tag}] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_pubmedqa(max_samples: int = 1000) -> list[dict]:
    """
    Load PubMedQA from HuggingFace.
    Each record: question + context paragraphs + long answer.
    We combine them into one passage per record.
    """
    print("Loading PubMedQA from HuggingFace...")
    dataset = load_dataset(
        "pubmed_qa",
        "pqa_labeled",
        split="train",
        trust_remote_code=True,
    )

    docs = []
    for row in list(dataset)[:max_samples]:
        context   = " ".join(row["context"]["contexts"])
        question  = row["question"]
        answer    = row.get("long_answer", "")
        combined  = f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}"
        docs.append({
            "text": clean_text(combined),
            "metadata": {"pubmed_id": str(row.get("pubid", "")), "question": question},
        })

    print(f"  Loaded {len(docs)} PubMedQA records")
    return docs


def load_medquad(max_samples: int = 500) -> list[dict]:
    """
    Load a sample from MedQuAD (NIH medical Q&A).
    Falls back to a small built-in sample if the URL is unavailable.
    """
    print("Loading MedQuAD...")
    try:
        resp = requests.get(MEDQUAD_URL, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        records = raw if isinstance(raw, list) else raw.get("data", [])
    except Exception as e:
        print(f"  Could not fetch MedQuAD from GitHub ({e}). Using fallback sample.")
        records = _medquad_fallback()

    docs = []
    for row in records[:max_samples]:
        q = row.get("question", "") or row.get("Question", "")
        a = row.get("answer", "") or row.get("Answer", "")
        if q and a:
            docs.append({
                "text": clean_text(f"Question: {q}\n\nAnswer: {a}"),
                "metadata": {"focus": row.get("focus", ""), "qtype": row.get("qtype", "")},
            })

    print(f"  Loaded {len(docs)} MedQuAD records")
    return docs


def _medquad_fallback() -> list[dict]:
    """Small hardcoded fallback so the pipeline never crashes without internet."""
    return [
        {"question": "What is diabetes?",
         "answer": "Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin, or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar."},
        {"question": "What are the symptoms of hypertension?",
         "answer": "Hypertension (high blood pressure) is often called the silent killer because it typically has no symptoms. When symptoms do occur, they may include headaches, shortness of breath, or nosebleeds."},
        {"question": "What is asthma?",
         "answer": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath."},
        {"question": "What causes Alzheimer's disease?",
         "answer": "The exact causes of Alzheimer's disease are not fully understood. But at a basic level, brain proteins fail to function normally, which disrupts the work of brain cells and triggers a series of toxic events. Neurons are damaged, lose connections to each other and eventually die."},
        {"question": "How is COVID-19 treated?",
         "answer": "COVID-19 treatment depends on severity. Mild cases are managed with rest, hydration, and fever reducers. Severe cases may require hospitalization, oxygen therapy, antiviral medications like remdesivir, and in critical cases, mechanical ventilation."},
    ]


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ingestion(
    pubmedqa_samples: int = 1000,
    medquad_samples: int = 500,
) -> list[dict]:
    """
    Full ingestion pipeline. Returns all chunks ready for embedding.
    Also saves to data/chunks.json for caching.
    """
    print("\n=== Starting Data Ingestion ===\n")

    all_chunks = []

    # 1. PubMedQA
    pubmed_docs   = load_pubmedqa(max_samples=pubmedqa_samples)
    pubmed_chunks = chunk_documents(pubmed_docs, source_tag="pubmedqa")
    all_chunks.extend(pubmed_chunks)

    # 2. MedQuAD
    medquad_docs   = load_medquad(max_samples=medquad_samples)
    medquad_chunks = chunk_documents(medquad_docs, source_tag="medquad")
    all_chunks.extend(medquad_chunks)

    # 3. Save cache
    cache_path = DATA_DIR / "chunks.json"
    with open(cache_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n✅ Ingestion complete.")
    print(f"   Total chunks : {len(all_chunks)}")
    print(f"   Saved to     : {cache_path}\n")

    return all_chunks


if __name__ == "__main__":
    chunks = run_ingestion(pubmedqa_samples=1000, medquad_samples=500)
    print(f"Sample chunk:\n{chunks[0]['text'][:300]}")
