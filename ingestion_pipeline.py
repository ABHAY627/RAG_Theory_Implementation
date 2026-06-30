import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. "
            "Please create it and add your .txt files."
        )

    # TextLoader needs utf-8 encoding on Windows to avoid codec errors
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. Please add your documents."
        )

    print(f"Loaded {len(documents)} document(s).\n")
    for i, doc in enumerate(documents[:2]):  # Preview first 2
        print(f"Document {i + 1}:")
        print(f"  Source  : {doc.metadata['source']}")
        print(f"  Length  : {len(doc.page_content)} characters")
        print(f"  Preview : {doc.page_content[:120].strip()}...")
        print(f"  Metadata: {doc.metadata}\n")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into overlapping chunks for better retrieval context."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunk(s).\n")

    for i, chunk in enumerate(chunks[:5]):  # Preview first 5
        print(f"--- Chunk {i + 1} ---")
        print(f"Source : {chunk.metadata['source']}")
        print(f"Length : {len(chunk.page_content)} characters")
        print(f"Content:\n{chunk.page_content}")
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunk(s).")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create a ChromaDB vector store from document chunks and persist it."""
    print("\nCreating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    os.makedirs(persist_directory, exist_ok=True)

    print("Building vector store — this may take a moment...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = vectorstore._collection.count()
    print(f"Vector store saved to '{persist_directory}' ({count} vectors).")
    return vectorstore


def load_vector_store(persist_directory="db/chroma_db"):
    """Load an existing ChromaDB vector store from disk."""
    print(f"Loading existing vector store from '{persist_directory}'...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} vector(s).")
    return vectorstore


def main():
    """Main ingestion pipeline entry point."""
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # If the vector store already exists, skip re-processing
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("✅ Vector store already exists — skipping ingestion.")
        return load_vector_store(persist_directory)

    print("No existing vector store found. Starting fresh ingestion...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persist_directory)

    print("\n✅ Ingestion complete! Documents are ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()
