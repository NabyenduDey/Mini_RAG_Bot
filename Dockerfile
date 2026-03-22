FROM python:3.11-slim-bookworm

# No system Tesseract required unless you set IMAGE_TEXT_BACKEND=tesseract
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "rag.app"]
