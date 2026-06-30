"""
Chunking Strategy 3: SemanticChunker
--------------------------------------
Unlike character-based splitters that cut text at fixed sizes or separators,
SemanticChunker groups sentences by MEANING using embeddings.

It computes cosine similarity between consecutive sentences and inserts a
chunk boundary wherever the semantic similarity drops below a threshold.
Result: each chunk is topically coherent, not just structurally split.

Embedding model: HuggingFace all-MiniLM-L6-v2 (free, no API key needed).
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Sample text — three distinct topics ───────────────────────────────────────
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


def print_chunks(chunks: list[str], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f'  "{chunk}"')
    print(f"\nTotal chunks: {len(chunks)}")


def main():
    print("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Model ready.\n")

    # ── Method 1: percentile threshold ────────────────────────────────────────
    # Looks at all similarity drop-off scores across the text, then draws a
    # boundary wherever the drop is in the top (100 - threshold) percentile.
    # Lower value → more sensitive → more, smaller chunks.
    # Higher value → less sensitive → fewer, larger chunks.
    chunker_percentile = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,  # split at the top 30% sharpest drops
    )
    chunks_percentile = chunker_percentile.split_text(tesla_text)
    print_chunks(chunks_percentile, "Method 1 — percentile threshold (70)")

    # ── Method 2: standard deviation threshold ────────────────────────────────
    # Draws a boundary wherever the similarity drop exceeds
    # (mean − threshold × std_dev). Lower value → more splits.
    chunker_std = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.0,
    )
    chunks_std = chunker_std.split_text(tesla_text)
    print_chunks(chunks_std, "Method 2 — standard_deviation threshold (1.0)")

    # ── Method 3: interquartile range ─────────────────────────────────────────
    # Splits where the drop exceeds Q3 + threshold × IQR.
    # More robust to outliers than standard deviation.
    chunker_iqr = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="interquartile",
        breakpoint_threshold_amount=1.5,
    )
    chunks_iqr = chunker_iqr.split_text(tesla_text)
    print_chunks(chunks_iqr, "Method 3 — interquartile threshold (1.5)")

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  percentile (70)          : {len(chunks_percentile)} chunk(s)")
    print(f"  standard_deviation (1.0) : {len(chunks_std)} chunk(s)")
    print(f"  interquartile (1.5)      : {len(chunks_iqr)} chunk(s)")

    print("""
Key takeaways:
  - SemanticChunker groups sentences by meaning, not by character count.
  - The tesla_text has 3 clear topics (financials, Model Y, production).
    A well-tuned threshold should produce exactly 3 chunks.
  - percentile is the most intuitive threshold to tune.
  - standard_deviation and interquartile are better for longer, noisier text.
  - Unlike CharacterTextSplitter / RecursiveCharacterTextSplitter, chunk size
    is NOT fixed — chunks grow or shrink based on topic coherence.
  - Uses HuggingFace all-MiniLM-L6-v2 embeddings (free, no API key needed).
""")


if __name__ == "__main__":
    main()
