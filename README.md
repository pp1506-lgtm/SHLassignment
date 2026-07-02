# SHL Assessment Advisor

A conversational FastAPI agent that helps hiring managers select SHL Individual Test assessments through dialogue.

## Quick Start

```bash
cd shl-agent
pip install -r requirements.txt

# 1. Build catalog and indexes (one-time)
python catalog/fetch_catalog.py
python catalog/build_index.py

# 2. Set Gemini API key
export GEMINI_API_KEY=your_key_here  # Windows: $env:GEMINI_API_KEY="..."

# 3. Run the API
uvicorn app.main:app --reload

# 4. Test health
curl http://localhost:8000/health

# 5. Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer"}]}'
```

## Run Evaluation

```bash
# Adversarial probes (no LLM needed)
python eval/adversarial_probes.py

# Recall@10 against ground truth traces (no LLM needed, retrieval only)
python eval/eval_harness.py

# Debug a specific trace
python eval/eval_harness.py C9
```

## Project Structure

```
shl-agent/
├── catalog/
│   ├── fetch_catalog.py      # Downloads + normalizes SHL catalog JSON
│   ├── build_index.py        # Builds FAISS + BM25 indexes
│   ├── shl_catalog.json      # Processed catalog (generated)
│   └── indexes/              # FAISS + BM25 index files (generated)
├── retrieval/
│   └── retriever.py          # Hybrid retriever with soft/hard filters
├── agent/
│   ├── state_machine.py      # Deterministic context extraction + state decisions
│   ├── prompt_templates.py   # All prompt text (centralised)
│   └── gemini_client.py      # Gemini Flash wrapper with JSON-mode
├── app/
│   ├── main.py               # FastAPI app with /health + /chat
│   └── schemas.py            # Pydantic request/response models
├── eval/
│   ├── traces.py             # Ground truth C1-C10 labeled traces
│   ├── eval_harness.py       # Recall@10 evaluation
│   └── adversarial_probes.py # 12 adversarial behavior probes
├── requirements.txt
├── Dockerfile
└── render.yaml
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Here are assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini Flash API key (free tier) |
