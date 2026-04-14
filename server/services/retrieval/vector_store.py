import os
import re
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from utils.logger import get_logger

logger = get_logger(__name__)

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "chroma")
_client = None
_embedding_fn = DefaultEmbeddingFunction()

# Patterns that signal a reference/bibliography chunk
_REF_PATTERNS = re.compile(
    r"(\[\d+\].*){3,}"           # three or more [N] citation entries
    r"|^\s*references\s*$"       # standalone "References" heading
    r"|bibliography",
    re.IGNORECASE | re.MULTILINE,
)


def _is_reference_chunk(text: str) -> bool:
    return bool(_REF_PATTERNS.search(text))


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(_DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=_DB_DIR)
    return _client


def _get_collection(name: str, cosine: bool = False):
    metadata = {"hnsw:space": "cosine"} if cosine else {}
    client = _get_client()
    try:
        col = client.get_or_create_collection(
            name,
            embedding_function=_embedding_fn,
            metadata=metadata if metadata else None,
        )
    except Exception:
        # Collection exists with different metadata; delete and recreate with new metric
        logger.warning("Recreating collection %r to apply new distance metric", name)
        client.delete_collection(name)
        col = client.create_collection(
            name,
            embedding_function=_embedding_fn,
            metadata=metadata,
        )
    distance = col.metadata.get("hnsw:space", "l2") if col.metadata else "l2"
    logger.info("Collection %r ready (distance=%s)", name, distance)
    return col


def index_chunks(paper_id: str, source: str, chunks: list[str]) -> dict:
    content_ids, content_docs, content_meta = [], [], []
    ref_ids, ref_docs, ref_meta = [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{paper_id}__chunk_{i}"
        meta = {"paper_id": paper_id, "source": source, "chunk_index": i}
        if _is_reference_chunk(chunk):
            ref_ids.append(chunk_id)
            ref_docs.append(chunk)
            ref_meta.append(meta)
        else:
            content_ids.append(chunk_id)
            content_docs.append(chunk)
            content_meta.append(meta)

    logger.info(
        "Chunk classification for paper_id=%r: content=%d reference=%d",
        paper_id, len(content_docs), len(ref_docs),
    )

    if content_docs:
        col = _get_collection(f"{source}_papers", cosine=True)
        logger.info("Indexing %d content chunks into %r", len(content_docs), col.name)
        col.upsert(documents=content_docs, ids=content_ids, metadatas=content_meta)

    if ref_docs:
        ref_col = _get_collection(f"{source}_papers_refs")
        logger.info("Storing %d reference chunks into %r", len(ref_docs), ref_col.name)
        ref_col.upsert(documents=ref_docs, ids=ref_ids, metadatas=ref_meta)

    return {"content_count": len(content_docs), "reference_count": len(ref_docs)}


def query_chunks(query: str, paper_id: str, source: str, k: int) -> list[dict]:
    logger.info("Querying collection: query=%r paper_id=%r source=%r k=%d", query, paper_id, source, k)
    collection = _get_collection(f"{source}_papers", cosine=True)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"paper_id": paper_id},
    )
    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        chunks.append({"chunk_index": meta.get("chunk_index"), "text": doc, "distance": dist})

    indices = [c["chunk_index"] for c in chunks]
    logger.info("Retrieved %d chunks: indices=%s", len(chunks), indices)
    return chunks
