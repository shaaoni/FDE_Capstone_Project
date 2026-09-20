"""
Vector search over the knowledge base (data/documentation.json).

Two entry points:
  build_index()   -- run once (or whenever the KB changes) to (re)build
                      the Chroma store on disk.
  retrieve(query)  -- used at request time by the rest of the pipeline.

CHUNK_SIZE / CHUNK_OVERLAP are called out in the Setup Guide as a real
design decision, not a default -- the guide explicitly asks you to try
at least two configurations, measure the difference in retrieval
quality, and record which you picked and why. Don't just ship the
starting values below without testing them; that's exactly the kind of
choice the assessment is looking for you to make deliberately.
"""
import json
import shutil
import gc
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH, EMBEDDING_MODEL, RETRIEVAL_TOP_K

CHUNK_SIZE = 800      # starting point from the Setup Guide -- test this
CHUNK_OVERLAP = 120   # starting point from the Setup Guide -- test this

# Tracks which persist_directory the last build_index() call wrote to, so
# retrieve() queries the store that's actually current rather than always
# assuming CHROMA_PATH.
_active_path: str = CHROMA_PATH
_store_cache: Chroma | None = None


def build_index(
    doc_path: str = "data/documentation.json",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    persist_directory: str = CHROMA_PATH,
) -> None:
    """
    Builds (or rebuilds) the vector store at persist_directory.

    Chroma.from_texts() APPENDS to whatever already exists in
    persist_directory rather than replacing it, so this clears the
    directory first -- without that, repeated builds silently stack
    duplicate copies of every passage on top of each other.

    NOTE on comparing multiple configurations in one Python process
    (e.g. a chunk-size comparison script): if you call build_index()
    twice pointed at the SAME directory in the same process, deleting
    the directory the second time can fail on Windows with
    PermissionError even after the first store is no longer referenced
    anywhere in your own code. Newer chromadb versions cache the
    underlying sqlite client internally, independent of whatever
    variable holds the Chroma object, so garbage-collecting your own
    reference doesn't release the file handle.

    The fix isn't a longer retry loop -- it's to never delete a
    directory a live connection might still be holding open. Pass a
    DIFFERENT persist_directory for each configuration you're comparing
    in the same run, e.g.:

        build_index(chunk_size=500, persist_directory="storage/chroma_500")
        build_index(chunk_size=800, persist_directory="storage/chroma_800")

    Each gets its own folder, so nothing is ever deleted out from under
    an open connection.
    """
    global _store_cache, _active_path

    _store_cache = None
    _active_path = persist_directory
    gc.collect()

    existing = Path(persist_directory)
    if existing.exists():
        try:
            shutil.rmtree(existing)
        except PermissionError as e:
            raise RuntimeError(
                f"Could not clear {persist_directory} -- a Chroma client in "
                "this process still has it open (this is a known Windows + "
                "chromadb interaction, not a bug in your data). If you're "
                "comparing multiple configs in one run, give each one its "
                "own persist_directory instead of reusing the same path."
            ) from e

    with open(doc_path) as f:
        documents = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    texts, metadatas = [], []
    for doc in documents:
        for chunk in splitter.split_text(doc["content"]):
            texts.append(chunk)
            metadatas.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc.get("category", ""),
            })

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_texts(
        texts=texts, metadatas=metadatas, embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"stored {len(texts)} passages from {len(documents)} documents "
          f"-> {persist_directory}")


def _get_store() -> Chroma:
    global _store_cache
    if _store_cache is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _store_cache = Chroma(persist_directory=_active_path, embedding_function=embeddings)
    return _store_cache


def retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Returns the top-k passages as [{text, doc_id, title, category, score}]."""
    store = _get_store()
    results = store.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "text": doc.page_content,
            "doc_id": doc.metadata["doc_id"],
            "title": doc.metadata["title"],
            "category": doc.metadata.get("category", ""),
            "score": score,
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    build_index()