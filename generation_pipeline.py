import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
PERSIST_DIRECTORY = "db/chroma_db"
GEMINI_MODEL      = "gemini-1.5-flash"
TOP_K             = 3

# ── Setup ──────────────────────────────────────────────────────────────────────
def setup():
    """Load vector store and initialise Gemini model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print(
            "[ERROR] GOOGLE_API_KEY is not set.\n"
            "Add your Gemini API key to .env:\n"
            "  GOOGLE_API_KEY=your_actual_key_here"
        )
        sys.exit(1)

    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        print(
            f"[ERROR] No vector store found at '{PERSIST_DIRECTORY}'.\n"
            "Run ingestion_pipeline.py first."
        )
        sys.exit(1)

    print(f"Loading vector store from '{PERSIST_DIRECTORY}'...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
    )
    print(f"Vector store ready — {db._collection.count()} vectors loaded.\n")

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )

    return db, model


# ── History-aware question rewriter ───────────────────────────────────────────
def rewrite_question(model, chat_history: list, user_question: str) -> str:
    """
    If there is prior chat history, ask Gemini to rewrite the user's question
    as a self-contained, searchable query so retrieval doesn't depend on pronouns
    or references that only make sense in context.
    """
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(
            content=(
                "Given the chat history below, rewrite the new question so it is "
                "fully standalone and can be understood without the history. "
                "Return only the rewritten question, nothing else."
            )
        ),
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]

    result = model.invoke(messages)
    return result.content.strip()


# ── Retrieve ───────────────────────────────────────────────────────────────────
def retrieve_docs(db: Chroma, search_question: str, k: int = TOP_K) -> list:
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(search_question)


# ── Generate ───────────────────────────────────────────────────────────────────
def generate_answer(model, chat_history: list, user_question: str, docs: list) -> str:
    if not docs:
        return "I couldn't find any relevant documents. Try rephrasing your question."

    context = "\n".join([f"- {doc.page_content}" for doc in docs])

    combined_input = (
        f"Based on the following documents, please answer this question: {user_question}\n\n"
        f"Documents:\n{context}\n\n"
        "Please provide a clear, helpful answer using only the information from these documents. "
        "If you can't find the answer in the documents, say "
        "'I don't have enough information to answer that question based on the provided documents.'"
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions based on "
                "provided documents and conversation history."
            )
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    return result.content


# ── Main ask function ──────────────────────────────────────────────────────────
def ask_question(db: Chroma, model, chat_history: list, user_question: str) -> str:
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: rewrite question to be standalone (history-aware retrieval)
    search_question = rewrite_question(model, chat_history, user_question)
    if search_question != user_question:
        print(f"Rewritten for search: {search_question}")

    # Step 2: retrieve relevant chunks
    docs = retrieve_docs(db, search_question)
    print(f"Found {len(docs)} relevant chunk(s):")
    for i, doc in enumerate(docs, 1):
        source  = doc.metadata.get("source", "unknown")
        preview = " ".join(doc.page_content.split()[:15])
        print(f"  [{i}] {source} → {preview}...")

    # Step 3: generate answer with full chat history for conversational context
    answer = generate_answer(model, chat_history, user_question, docs)

    # Step 4: update history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer: {answer}")
    return answer


# ── Chat loop ──────────────────────────────────────────────────────────────────
def start_chat(db: Chroma, model) -> None:
    chat_history = []

    print("=" * 60)
    print("  RAG Chat — powered by Gemini (history-aware)")
    print("  Type your question and press Enter.")
    print("  Type 'history' to see the conversation so far.")
    print("  Type 'clear' to reset the conversation history.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if question.lower() == "history":
            if not chat_history:
                print("No conversation history yet.")
            else:
                print("\n--- Conversation History ---")
                for msg in chat_history:
                    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                    print(f"{role}: {msg.content}\n")
            continue

        if question.lower() == "clear":
            chat_history.clear()
            print("Conversation history cleared.")
            continue

        ask_question(db, model, chat_history, question)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db, model = setup()
    start_chat(db, model)
