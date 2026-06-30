import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIRECTORY = "db/chroma_db"

# Synthetic test queries — answers live in the docs folder
QUERIES = [
    "What does Google focus on?",
    "Who founded Microsoft and when?",
    "What is Nvidia known for in AI?",
]

# ── Load vector store ─────────────────────────────────────────────────────────
def load_vector_store(persist_directory: str) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory}'.\n"
            "Run ingestion_pipeline.py first to create it."
        )

    print(f"Loading vector store from '{persist_directory}'...")

    # Must use the same embedding model that was used during ingestion
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        # NOTE: collection_metadata not passed — it's already set on disk
    )

    print(f"Vector store loaded. Total vectors: {vectorstore._collection.count()}\n")
    return vectorstore


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(vectorstore: Chroma, query: str, k: int = 3) -> list:
    """
    Retrieve the top-k most relevant chunks for a query.

    Two retriever modes are available (switch by uncommenting):
      1. Plain similarity  — returns top-k chunks regardless of score
      2. Score threshold   — only returns chunks above a minimum similarity score
    """

    # Mode 1: plain similarity search (default)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Mode 2: similarity with score threshold (uncomment to use)
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": k,
    #         "score_threshold": 0.3,  # only return chunks with similarity >= 0.3
    #     },
    # )

    return retriever.invoke(query)


# ── Display ───────────────────────────────────────────────────────────────────
def display_results(query: str, docs: list) -> None:
    """Pretty-print retrieval results."""
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    if not docs:
        print("No relevant documents found.\n")
        return

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n[{i}] Source: {source}")
        print(f"    Length: {len(doc.page_content)} characters")
        print(f"    Content:\n{doc.page_content}")
        print("-" * 60)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== RAG Retrieval Pipeline ===\n")

    vectorstore = load_vector_store(PERSIST_DIRECTORY)

    for query in QUERIES:
        docs = retrieve(vectorstore, query, k=3)
        display_results(query, docs)


if __name__ == "__main__":
    main()
