# SHL Assessment Advisor — Approach Document

## Problem Decomposition

The core challenge is translating a hiring manager's vague intent into a grounded shortlist of SHL assessments across a 400+ item catalog. Three sub-problems drive the design:

1. **When to act vs. ask** — a premature recommendation on vague input is useless; an agent that asks too many questions is annoying.
2. **What to retrieve** — the catalog spans programming-language tests, personality measures, cognitive assessments, and simulations. The right match for "Java developer" is very different from "senior sales leader."
3. **How to stay grounded** — the LLM must never fabricate URLs or product names. Every recommendation must come from the catalog.

---

## Architecture

```
POST /chat (stateless, full history each call)
      │
      ▼
extract_context()        ← Regex-based, deterministic, no LLM
      │
      ▼
decide_state()           ← CLARIFY / RECOMMEND / COMPARE / REFUSE / DONE
      │
      ├─ REFUSE ──────→  Fast-path return (no LLM, <1ms)
      │
      ├─ CLARIFY ─────→  LLM with system prompt only (no catalog context)
      │
      ├─ RECOMMEND ───→  HybridRetriever(top-10) → context injection → LLM
      │
      └─ COMPARE ─────→  Named-item lookup → context injection → LLM
```

**Key design choice: deterministic intent before LLM.** Context extraction (job level, test type codes, vagueness, injection/off-topic) is done with regex and heuristics before any LLM call. This eliminates a whole class of failure modes — the system cannot be confused into recommending on a vague query or refusing a valid one because the LLM misclassified intent.

---

## Retrieval Setup

**Hybrid BM25 + FAISS** (α = 0.6 semantic, 0.4 keyword):

- **FAISS** with `all-MiniLM-L6-v2` embeddings captures semantic similarity — "hiring a data analyst" finds "Numerical Reasoning" even without the exact keyword.
- **BM25** handles exact tech-skill matching — "Java 8" or "Apache Kafka" need exact token overlap that cosine similarity can miss at scale.
- **Fusion:** scores are independently min-max normalised, then blended. This avoids the scale mismatch between BM25 and cosine scores.

**Soft/hard filter strategy:** Filters (job level, test type, duration, remote) are first applied strictly. If fewer than `top_k` results remain, job-level filtering is relaxed to avoid empty results. Type-code and remote filters are always enforced.

**Search text construction:** Each catalog item's embedding is over `name + description + job_levels + keys` — not description alone. This ensures "Entry-Level Customer Service" items rank well for "contact centre agents" queries without requiring keyword overlap on the description.

---

## Prompt Design

**Single JSON-mode output schema** is enforced at the Gemini API level (`response_mime_type=application/json`). This removes the need for regex-based JSON extraction on the happy path and dramatically reduces malformed output.

**Context injection as a conversation turn, not a system field:** Catalog items are injected into the message list as a synthetic user/assistant exchange immediately before the last user turn. This works around Gemini's lack of a native system role while keeping the conversation structure clean.

**Temperature = 0.2:** Low temperature reduces creative extrapolation. The agent should be grounded and predictable, not inventive.

**URL guardrail as post-processing:** After every LLM response, recommended URLs are validated against the catalog's URL set. Hallucinated URLs are silently dropped. This is the last line of defense.

---

## State Machine

States are pure functions of conversation history. Key transitions:

| Condition | State |
|---|---|
| Injection keywords detected | REFUSE |
| Off-topic keywords detected | REFUSE |
| Compare keywords + non-vague context | COMPARE |
| Refinement keywords + turns > 2 | REFINE |
| Turns ≥ 6 (force commit) | RECOMMEND |
| Vague + fewer than 2 clarification turns | CLARIFY |
| Otherwise | RECOMMEND |

**Turn cap enforcement:** Agent is forced into RECOMMEND at turn 6, guaranteeing a shortlist before the 8-turn evaluator cap.

---

## Evaluation

**Recall@10** is computed retrieval-only (no LLM), enabling fast iteration on α and filter strategy without API costs. Mean Recall@10 across C1–C10 traces is the primary tuning signal.

**Adversarial probes (12 total)** are fully deterministic — injection/off-topic refusal, no-recommendation on vague turn-1, recommend after sufficient context, turn-cap force-commit, refinement detection, compare intent detection, and entity extraction accuracy. All probes run in <100ms with no network calls.

---

## What Didn't Work / Trade-offs

**Tried: LLM for intent classification.** Initial design used a Gemini call to classify intent before retrieval. This added ~2-3s latency and occasionally misclassified "refine" as "clarify" on short messages. Switched to regex-based classification — faster, auditable, and more reliable.

**Trade-off: sentence-transformers vs. Gemini embeddings.** Gemini embeddings are higher quality but add API latency and cost. `all-MiniLM-L6-v2` runs locally in ~10ms per query and achieves sufficient Recall@10 when combined with BM25 for keyword-heavy queries.

**Trade-off: FAISS flat vs. IVF index.** Flat index with inner product on 400 items has O(n) search but completes in <1ms — IVF overhead is not warranted at this catalog size.

**Known limitation:** Catalog changes require rebuilding the FAISS index. For production, a nightly rebuild or incremental update would be needed.

---

## Tools Used

- **Gemini 1.5 Flash** (free tier): primary LLM, JSON-mode generation
- **sentence-transformers** (`all-MiniLM-L6-v2`): embedding generation
- **FAISS CPU**: vector similarity search
- **rank-bm25**: BM25 keyword retrieval
- **FastAPI + Uvicorn**: API framework
- **Render free tier**: deployment platform
- **Antigravity IDE (Claude Sonnet)**: agentic coding assistance for scaffolding
