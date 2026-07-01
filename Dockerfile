FROM python:3.11-slim

WORKDIR /app

# Install system deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-build catalog index at image build time so cold starts are fast
# GEMINI_API_KEY is NOT needed for catalog building
RUN python catalog/fetch_catalog.py && python catalog/build_index.py

# Run
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
