"""User-visible bot copy — keeps handlers and pipeline text in one place."""

# --- Generic ----------------------------------------------------------------

NEED_QUESTION = (
    "I need something to work with: use /ask with your question, "
    "or send a photo that includes a one-line caption."
)

# --- Photos / screenshots -------------------------------------------------

PHOTO_CAPTION_REQUIRED = (
    "A one-line caption is required with every photo or screenshot.\n\n"
    "Add your caption in Telegram when you send the image (the text field above the image). "
    'Example: "Answer using our recipes" or "What does policy say about this?"\n\n'
    "Then send the photo again with that caption."
)

# --- /start ----------------------------------------------------------------

START_INTRO = (
    "Hi. I answer questions using your local knowledge files.\n\n"
    "• Text: /ask followed by your question.\n"
    "• Screenshot or photo: you must add a one-line caption when sending the image. "
    "A vision model reads text from the picture; your caption says what you want "
    '(e.g. "How do I make this?").\n'
    "• /help — more detail"
)

# --- /help -----------------------------------------------------------------

HELP_TEXT = (
    "/ask <your question>\n"
    "  Example: /ask How do I request time off?\n\n"
    "Photo or screenshot\n"
    "  Mandatory: add a single-line caption before sending (Telegram’s caption box on the image).\n"
    "  The bot reads on-screen text with a local model (BLIP-2 or LLaVA per your .env), "
    "merges it with your caption, searches the knowledge base, then replies.\n\n"
    "/start — short intro"
)
