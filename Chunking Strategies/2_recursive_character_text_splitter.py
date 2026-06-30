"""
Chunking Strategy 2: RecursiveCharacterTextSplitter
-----------------------------------------------------
Tries a list of separators in order: ["\n\n", "\n", ". ", " ", ""].
If a chunk produced by the first separator is still larger than chunk_size,
it falls through to the next separator — all the way down to single characters
if needed. This guarantees no chunk exceeds chunk_size.

This is the recommended default splitter for most RAG pipelines.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges
Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


def run_splitter(chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(tesla_text)


def print_chunks(chunks: list[str], label: str) -> None:
    print(f"\n--- {label} ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({len(chunk)} chars):")
        print(f"    {chunk!r}")


def main():
    print("=" * 60)
    print("RECURSIVE CHARACTER TEXT SPLITTER — cascading separators")
    print("=" * 60)

    # ── Example 1: no overlap ─────────────────────────────────────────────────
    # The long paragraph gets broken at sentence / word boundaries as needed.
    chunks_no_overlap = run_splitter(chunk_size=100, chunk_overlap=0)
    print_chunks(chunks_no_overlap, "chunk_size=100, chunk_overlap=0")

    # ── Example 2: with overlap ───────────────────────────────────────────────
    # Overlap carries context from the end of one chunk into the start of the
    # next — improves retrieval when an answer spans a chunk boundary.
    chunks_with_overlap = run_splitter(chunk_size=100, chunk_overlap=20)
    print_chunks(chunks_with_overlap, "chunk_size=100, chunk_overlap=20")

    # ── Side-by-side stats ────────────────────────────────────────────────────
    print("\n--- Stats ---")
    print(f"  No overlap   : {len(chunks_no_overlap)} chunks")
    print(f"  With overlap : {len(chunks_with_overlap)} chunks  (more chunks due to overlap)")

    all_within_limit = all(len(c) <= 100 for c in chunks_no_overlap)
    print(f"\n  All chunks ≤ 100 chars (no overlap)? {all_within_limit}")

    print("\n")
    print("Key takeaway:")
    print("  RecursiveCharacterTextSplitter cascades through separators until")
    print("  every chunk fits within chunk_size — solving the oversized-chunk")
    print("  problem that CharacterTextSplitter has.")
    print("  chunk_overlap > 0 preserves cross-boundary context for retrieval.")


if __name__ == "__main__":
    main()
