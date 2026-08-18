FROM python:3.12-slim

WORKDIR /app



# System dependencies needed to build some Python packages.
# Cleaned up in the same layer to keep image size down.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Install CPU-only torch FIRST, and separately from requirements.txt.
# Doing this in its own layer means Docker's build cache reuses it
# whenever only application code changes — the biggest, slowest
# dependency is not re-downloaded on every rebuild.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu




# Install the rest of the dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



COPY app/ ./app/
COPY data/raw/ ./data/raw/


# Pre-download the local HF models at BUILD time, not at first request.
# This is what makes the container's startup fast and independent of
# HuggingFace Hub availability at runtime.

RUN python -c "from app.emotion.emotion_detector import get_pipeline; get_pipeline()"
RUN python -c "from app.intent.intent_detector import get_pipeline; get_pipeline()"


# Build the RAG embedding index at BUILD time, from the real product
# data copied above. The image ships ready to serve — no manual step
# needed after deploy.
RUN python -m app.rag.build_embeddings


# Directory for analytics logs written at runtime.
RUN mkdir -p data/analytics

EXPOSE 8000

# Sync FastAPI app (Path A design) run with uvicorn.
# --workers 1 for now: scaling to multiple workers is a deployment-time
# decision once real traffic justifies it, not a Dockerfile concern.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]









