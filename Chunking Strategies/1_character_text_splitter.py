"""
Chunking Strategy 1: CharacterTextSplitter
-------------------------------------------
Splits text using a SINGLE separator (default: "\n\n").
Problem: if a chunk is still larger than chunk_size after splitting on that
separator, LangChain cannot split it further — it just keeps it oversized
and prints a warning. This makes it unreliable for real-world text.
"""

from langchain_text_splitters import CharacterTextSplitter

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges
Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


def demo_separator(separator: str, separator_label: str) -> None:
    print(f"\n--- separator={separator_label!r} ---")
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=100,
        chunk_overlap=0,
    )
    chunks = splitter.split_text(tesla_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:80].strip()!r}{'...' if len(chunk) > 80 else ''}")


def main():
    print("=" * 60)
    print("CHARACTER TEXT SPLITTER — single separator")
    print("=" * 60)

    # ── Separator: double newline (default) ───────────────────────────────────
    # Splits on paragraph breaks. The long paragraph at the end has no "\n\n"
    # so it cannot be split and will exceed chunk_size.
    demo_separator("\n\n", "\\n\\n (default)")

    # ── Separator: single newline ─────────────────────────────────────────────
    # Finer splits, but the single long line still can't be broken further.
    demo_separator("\n", "\\n")

    # ── Separator: period + space ─────────────────────────────────────────────
    # Sentence-level splits. Works better for prose but misses header lines.
    demo_separator(". ", ". (sentence)")

    # ── Separator: space ─────────────────────────────────────────────────────
    # Word-level splits. Very granular — chunks lose sentence context.
    demo_separator(" ", "space (word level)")

    print("\n")
    print("Key takeaway:")
    print("  CharacterTextSplitter uses ONE separator at a time.")
    print("  If a chunk still exceeds chunk_size after the split, it stays oversized.")
    print("  Use RecursiveCharacterTextSplitter (see file 2) to fix this.")


if __name__ == "__main__":
    main()
