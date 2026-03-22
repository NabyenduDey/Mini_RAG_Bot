# How this bot works (plain language)

Think of four boxes in a row:

1. **You** — You send a **written question** (`/ask …`) and/or a **picture** (for example a screenshot). If you add a **caption** to the picture, that counts as extra words from you.

2. **Reading the picture** — If there is a picture, a **local program on your computer** (a Hugging Face model such as BLIP or LLaVA, or optional Tesseract) tries to **turn what it sees into words**. For screenshots, we ask the model to focus on visible text. Results are not perfect.

3. **Finding the right notes** — Your words + any text from the image are used as a search. The program compares that to **small text files** in the `knowledge/` folder (already stored in a local database). It picks the **few pieces** that look most related.

4. **Writing the answer** — **Ollama** (another local program) reads those pieces and writes a **short answer** back to you on Telegram.

**Choosing the image model:** Set `IMAGE_TEXT_BACKEND` in `.env` — `blip_vqa` (default, lighter), `blip_caption`, `llava` (heavier, needs GPU for comfort), `clip_interrogator` (extra pip package), or `tesseract` (classic OCR + system install).

**What you need:** `pip install` the Python packages in `requirements.txt`; the first run may **download model weights** from Hugging Face (needs internet once, unless you point to a local cache).

**Voice / audio is not supported.**
