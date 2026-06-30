"""
Chunking Strategy 4: Agentic Chunking
---------------------------------------
Instead of splitting text by character count or embedding similarity,
we ask an LLM to decide WHERE to split based on semantic understanding.

The LLM reads the full text and inserts "<<<SPLIT>>>" markers at natural
topic boundaries — effectively acting as a domain-aware chunking agent.

LLM used: Gemini 1.5 Flash (via langchain-google-genai, same as generation pipeline).
No embeddings needed — this is purely LLM-driven.
"""

import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

SPLIT_MARKER = "<<<SPLIT>>>"

# ── Sample text ───────────────────────────────────────────────────────────────
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


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(text: str, max_chars: int = 200) -> str:
    return f"""You are a text chunking expert. Split the text below into logical chunks.

Rules:
- Each chunk should be around {max_chars} characters or less
- Split ONLY at natural topic boundaries
- Keep related sentences together in the same chunk
- Insert the exact marker {SPLIT_MARKER} between chunks — nothing else
- Do NOT add any commentary, numbering, or explanations
- Return the full text with {SPLIT_MARKER} markers inserted

Text:
{text}

Return the text with {SPLIT_MARKER} markers only:"""


# ── Chunker ───────────────────────────────────────────────────────────────────
def agentic_chunk(llm, text: str, max_chars: int = 200) -> list[str]:
    """Ask Gemini to insert split markers, then split on them."""
    print("Asking Gemini to chunk the text...")
    prompt  = build_prompt(text, max_chars)
    response = llm.invoke(prompt)
    marked_text = response.content

    raw_chunks = marked_text.split(SPLIT_MARKER)
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks


# ── Display ───────────────────────────────────────────────────────────────────
def print_chunks(chunks: list[str], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f'  "{chunk}"')
    print(f"\nTotal chunks: {len(chunks)}")


def print_stats(chunks: list[str]) -> None:
    sizes = [len(c) for c in chunks]
    print(f"\n--- Stats ---")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg size     : {sum(sizes) // len(sizes)} chars")
    print(f"  Largest      : {max(sizes)} chars")
    print(f"  Smallest     : {min(sizes)} chars")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print(
            "[ERROR] GOOGLE_API_KEY is not set.\n"
            "Add your Gemini API key to .env:\n"
            "  GOOGLE_API_KEY=your_actual_key_here"
        )
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,   # deterministic — we want consistent splits
    )

    # ── Run 1: ~200 char target ───────────────────────────────────────────────
    chunks_200 = agentic_chunk(llm, tesla_text, max_chars=200)
    print_chunks(chunks_200, "Agentic Chunking — target ≤ 200 chars")
    print_stats(chunks_200)

    # ── Run 2: ~100 char target (finer splits) ────────────────────────────────
    chunks_100 = agentic_chunk(llm, tesla_text, max_chars=100)
    print_chunks(chunks_100, "Agentic Chunking — target ≤ 100 chars")
    print_stats(chunks_100)

    print(f"""
{'=' * 60}
Key takeaways:
  - The LLM reads the full text and splits at MEANINGFUL boundaries,
    not arbitrary character counts.
  - Unlike SemanticChunker (embedding similarity), this uses language
    understanding — it knows a heading like "Model Y Performance"
    starts a new topic.
  - Downside: slower and costs API calls per document.
  - Best used when document structure matters (reports, articles, docs)
    and chunking quality is more important than speed.
{'=' * 60}
""")


if __name__ == "__main__":
    main()
