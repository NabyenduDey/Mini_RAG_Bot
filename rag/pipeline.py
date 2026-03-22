"""
End-to-end answer path in three simple steps:

1. Collect question text — what the user typed plus text from the image (local HF model or Tesseract).
2. Look up similar passages in the local document index (vector search).
3. Call the local chat model (Ollama) with those passages and answer.

See HOW_IT_WORKS.md in the project folder for the same story without code jargon.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import httpx

from . import config
from .embeddings import encode_texts
from .image_text import extract_text_from_image
from .prompts import SYSTEM_RAG, build_user_prompt
from .user_messages import NEED_QUESTION, PHOTO_CAPTION_REQUIRED
from .vector_store import _connect, init_schema, search

logger = logging.getLogger(__name__)

# One SQLite connection per thread pool worker — avoids open/close + init_schema on every query.
_reader_local = threading.local()
_http_lock = threading.Lock()
_http_client: httpx.AsyncClient | None = None


def _reader_conn():
    conn = getattr(_reader_local, "conn", None)
    if conn is None:
        conn = _connect(config.SQLITE_PATH)
        init_schema(conn)
        _reader_local.conn = conn
    return conn


def _build_search_query(*, typed_message: str, text_from_image: str) -> str:
    """
    Text used for embedding + retrieval.

    Photos must include a caption (enforced in the bot). We combine:
    on-screen text from the vision model (if any) + the user’s one-line caption for search.
    """
    typed = (typed_message or "").strip()
    from_img = (text_from_image or "").strip()
    if from_img and typed:
        return f"{from_img}\n\nAdditional question from the user (photo caption): {typed}"
    if from_img:
        return from_img
    return typed


def _narrow_retrieval(
    rows: list[tuple[int, str, str, float]],
) -> list[tuple[int, str, str, float]]:
    """
    Drop neighbors much farther than the best match so unrelated docs (e.g. policy vs recipe)
    do not reach the LLM. Also cut at a “elbow” (big jump in distance) so two clusters
    (recipe vs HR) are not merged when the user asked about one topic only.
    """
    if not rows:
        return rows
    margin = config.RETRIEVAL_DISTANCE_MARGIN
    if margin <= 0:
        return rows[: config.TOP_K]

    best_d = rows[0][3]
    band = [r for r in rows if r[3] <= best_d + margin]
    if not band:
        band = [rows[0]]

    elbow = config.RETRIEVAL_ELBOW_GAP
    out: list[tuple[int, str, str, float]] = [band[0]]
    for i in range(1, len(band)):
        prev_d, d = band[i - 1][3], band[i][3]
        if d - prev_d > elbow:
            break
        out.append(band[i])
        if len(out) >= config.TOP_K:
            break

    return out[: config.TOP_K]


def _find_similar_chunks(search_text: str) -> list[tuple[int, str, str, float]]:
    """Vector search, then distance-margin filter (worker thread)."""
    conn = _reader_conn()
    q = search_text.strip() or "help"
    q_emb = encode_texts(config.EMBEDDING_MODEL, [q])[0]
    candidates = search(conn, q_emb, config.TOP_K_CANDIDATES)
    return _narrow_retrieval(candidates)


async def _ollama_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    with _http_lock:
        if _http_client is None:
            timeout = httpx.Timeout(config.OLLAMA_HTTP_TIMEOUT)
            _http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
    return _http_client


async def close_ollama_http_client() -> None:
    """Close the shared Ollama HTTP client (call from Application.post_shutdown)."""
    global _http_client
    with _http_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


async def _call_ollama_chat(system: str, user: str) -> str:
    """Step 3 — ask the local LLM to produce the final reply."""
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload: dict = {
        "model": config.OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    opts: dict = {"temperature": config.OLLAMA_TEMPERATURE}
    if config.OLLAMA_NUM_PREDICT:
        opts["num_predict"] = int(config.OLLAMA_NUM_PREDICT)
    if config.OLLAMA_NUM_CTX:
        opts["num_ctx"] = int(config.OLLAMA_NUM_CTX)
    payload["options"] = opts

    client = await _ollama_client()
    r = await client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


async def answer_user_message(
    *,
    typed_text: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """
    RAG entry: `typed_text` is /ask text or the photo caption; `image_bytes` is set for photos.

    Every photo must ship with a one-line caption (checked in `on_photo` and again here).
    """
    typed = (typed_text or "").strip()

    if image_bytes and not typed:
        return PHOTO_CAPTION_REQUIRED

    from_image = ""
    if image_bytes:
        from_image = (await asyncio.to_thread(extract_text_from_image, image_bytes)).strip()

    if not typed and not from_image:
        return NEED_QUESTION

    search_query = _build_search_query(
        typed_message=typed, text_from_image=from_image
    ).strip()
    if not search_query:
        return (
            "Could not build a search query. If you sent a photo, try a clearer screenshot "
            "or a more specific caption."
        )

    try:
        rows = await asyncio.to_thread(_find_similar_chunks, search_query)
    except Exception as e:
        logger.exception("Retrieval failed")
        return f"Could not search the knowledge base: {e}"

    context = "\n\n---\n\n".join(f"[{doc}]\n{chunk}" for _, doc, chunk, _ in rows)

    user_prompt = build_user_prompt(
        typed_message=typed,
        text_from_image=from_image,
        context=context,
        image_attached=bool(image_bytes),
    )

    try:
        return await _call_ollama_chat(SYSTEM_RAG, user_prompt)
    except httpx.ConnectError as e:
        logger.warning(
            "Ollama unreachable at %s (%s)", config.OLLAMA_BASE_URL, e
        )
        return (
            "Cannot connect to Ollama (nothing is listening at that address).\n\n"
            f"Configured URL: {config.OLLAMA_BASE_URL}\n"
            f"Model: {config.OLLAMA_CHAT_MODEL}\n\n"
            "Important: 127.0.0.1 is always THIS machine — the one running the Python bot.\n"
            "If you installed Ollama on your Mac but run the bot on a remote server (SSH, DSW, "
            "Docker elsewhere), the server cannot see your Mac’s Ollama. Either run the bot on "
            "the same Mac as Ollama, or install Ollama on the server and point OLLAMA_BASE_URL there.\n\n"
            "Same machine as Ollama (e.g. Mac):\n"
            "  1) Open the Ollama app (menu bar) so the API is up — see https://ollama.com/download/mac\n"
            f"  2) Terminal: ollama pull {config.OLLAMA_CHAT_MODEL}\n"
            f"  3) On the machine that runs THIS bot, test: curl -sS {config.OLLAMA_BASE_URL}/api/version\n\n"
            "Remote bot + Ollama on another PC: set OLLAMA_BASE_URL in .env to a reachable host "
            "(SSH tunnel, LAN IP, or run Ollama on the server)."
        )
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out at %s", config.OLLAMA_BASE_URL)
        return (
            "Ollama did not respond in time. Try again, use a smaller/faster model, "
            "or increase load time on the Ollama host."
        )
    except httpx.HTTPStatusError as e:
        return (
            f"The chat model returned an error ({e.response.status_code}). "
            "Is Ollama running, and did you run `ollama pull` for your model? "
            f"Details: {e.response.text[:400]}"
        )
    except Exception as e:
        logger.exception("LLM chat failed")
        return f"Could not get an answer from the chat model: {e}"
