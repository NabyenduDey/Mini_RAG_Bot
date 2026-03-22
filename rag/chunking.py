"""Split markdown/text files into overlapping character windows."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, max_chars - 1)) if max_chars > 1 else 0
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap if overlap else end
    return chunks


def chunk_markdown_file(content: str, max_chars: int, overlap: int) -> list[str]:
    """Prefer paragraph boundaries, then fall back to sliding windows."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        if len(p) > max_chars:
            if buf:
                out.extend(chunk_text("\n\n".join(buf), max_chars, overlap))
                buf = []
                buf_len = 0
            out.extend(chunk_text(p, max_chars, overlap))
            continue
        added = buf_len + len(p) + (2 if buf else 0)
        if added <= max_chars:
            buf.append(p)
            buf_len = added
        else:
            if buf:
                out.extend(chunk_text("\n\n".join(buf), max_chars, overlap))
            buf = [p]
            buf_len = len(p)
    if buf:
        out.extend(chunk_text("\n\n".join(buf), max_chars, overlap))
    return out
