# Module 08 - Embeddings & Semantic Search
# 8.5 Chunking Text for Embedding
# Real documents are too long to embed whole. Chunk them into pieces that fit
# within the embedding model's token limit (typically 512-8192 tokens
# depending on the model).
#
# This file is pure Python - no API key or network access needed to run it.

from dataclasses import dataclass
import re


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """
    Split text into chunks that respect sentence boundaries. ``max_chars`` is
    a soft limit when one sentence alone is longer than it. Overlap keeps whole
    sentences so words are never cut at chunk edges.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars - 1.")

    # Keep source spans so Chunk.char_start/char_end point to the exact text.
    sentences = [
        (match.start(), match.end())
        for match in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", text, re.DOTALL)
    ]

    chunks: list[Chunk] = []
    current: list[tuple[int, int]] = []
    chunk_idx = 0

    for sentence_span in sentences:
        candidate_start = current[0][0] if current else sentence_span[0]
        candidate_end = sentence_span[1]

        if current and candidate_end - candidate_start > max_chars:
            start, end = current[0][0], current[-1][1]
            chunks.append(Chunk(doc_id, chunk_idx, text[start:end], start, end))
            chunk_idx += 1

            # Retain whole trailing sentences that intersect the requested
            # overlap window, then add the new sentence.
            overlap_boundary = end - overlap_chars
            current = [span for span in current if span[1] > overlap_boundary]
            current.append(sentence_span)

            # Drop the oldest overlap sentences if they make the new chunk too
            # large. The new sentence itself is never discarded.
            while len(current) > 1 and current[-1][1] - current[0][0] > max_chars:
                current.pop(0)
        else:
            current.append(sentence_span)

    # Save final chunk
    if current:
        start, end = current[0][0], current[-1][1]
        chunks.append(Chunk(doc_id, chunk_idx, text[start:end], start, end))

    return chunks


DOCUMENT = """
Large language models (LLMs) are neural networks trained on vast amounts of text data.
They learn to predict the next token in a sequence, which gives them broad language understanding.
Models like GPT-4 and Claude are examples of LLMs used in production today.

Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
This allows the model to answer questions about recent events or private data it was never trained on.

The retrieval step in RAG typically uses embedding-based semantic search.
A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
These documents are then injected into the LLM's context window alongside the query.
"""


if __name__ == "__main__":
    chunks = chunk_by_sentences(DOCUMENT.strip(), doc_id="intro_to_llms", max_chars=300, overlap_chars=50)
    for c in chunks:
        print(f"Chunk {c.chunk_index} ({c.char_start}-{c.char_end}): {c.text[:80]}...")
        print()
