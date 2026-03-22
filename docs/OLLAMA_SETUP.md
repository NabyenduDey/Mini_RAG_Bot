# Ollama and this bot

The bot calls **Ollama’s HTTP API** at `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`).

## `127.0.0.1` means “this computer”

Whichever machine runs `python3 -m rag.app` is the one that must have something listening on that URL.

| Where the bot runs | Where Ollama must run |
|--------------------|------------------------|
| Your Mac terminal | Mac ([ollama.com/download/mac](https://ollama.com/download/mac)) |
| Remote Linux (DSW, SSH, server) | **Same server**, or change `.env` to a URL that server can reach |

Installing Ollama on your Mac does **not** make it available to a bot process on a remote server.

## Mac: make sure the API is up

1. Open **Ollama** from Applications (menu bar icon = daemon usually running).
2. In Terminal:
   ```bash
   ollama pull llama3.2
   curl -s http://127.0.0.1:11434/api/tags
   ```
   If `curl` fails with connection refused, quit and reopen Ollama.

3. Run the bot **on the same Mac**:
   ```bash
   cd "/path/to/mini_rag_telegram_bot"
   source .venv/bin/activate
   python3 -m rag.app
   ```

## Remote server + Ollama only on Mac

- **Easiest:** Run the bot on your Mac while Ollama stays on Mac.
- **Or:** Install Ollama on the server and start it there; keep `OLLAMA_BASE_URL=http://127.0.0.1:11434` on that server.

## Port 11434 already in use (e.g. Cursor)

If `lsof -nP -iTCP:11434 -sTCP:LISTEN` shows **Cursor** (or anything other than **ollama**), that process owns the port. Hitting `http://127.0.0.1:11434` then talks to the wrong app → **empty reply** / **EOF**.

**Fix:** Run Ollama on another port and point everything at it:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama serve
```

In another terminal (same `export`):

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama run llama3.2
```

Add `export OLLAMA_HOST=127.0.0.1:11435` to `~/.zshrc` if you want it every session.

In the bot `.env`: `OLLAMA_BASE_URL=http://127.0.0.1:11435`

In **Cursor**, if you use local Ollama, set the base URL to port **11435** (or free 11434 by changing Cursor’s setting, if available).
