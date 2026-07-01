"""
main.py
FastAPI application exposing GET /health and POST /chat.
All state is derived from the request body — service is fully stateless.
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import ChatRequest, ChatResponse, Recommendation
from agent.state_machine import (
    ConversationContext,
    decide_state,
    extract_context,
    format_catalog_context,
    STATE_CLARIFY,
    STATE_COMPARE,
    STATE_RECOMMEND,
    STATE_REFINE,
    STATE_REFUSE,
    STATE_DONE,
    MAX_TURNS,
)
from agent.prompt_templates import (
    SYSTEM_PROMPT,
    COMPARE_CONTEXT_PROMPT,
    RECOMMENDATION_CONTEXT_PROMPT,
    REFUSAL_REPLY,
    INJECTION_REFUSAL_REPLY,
)
from agent.gemini_client import chat_completion
from retrieval.retriever import get_retriever

logger = logging.getLogger("shl_agent")
logging.basicConfig(level=logging.INFO)

# ── Lifespan: warm up retriever at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retriever indexes...")
    try:
        retriever = get_retriever()
        logger.info(f"Retriever ready — {retriever.catalog_size()} catalog items.")
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}. Will retry on first request.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Advisor",
    description="Conversational agent for SHL Individual Test assessment selection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INTERNAL_TIMEOUT = 25  # seconds — buffer under the 30s API timeout


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "SHL Assessment Advisor",
        "version": "1.0.0",
        "status": "running",
        "description": "Conversational agent for SHL Individual Test assessment selection.",
        "endpoints": {
            "GET /health": "Health check",
            "POST /chat": "Send conversation messages, receive assessment recommendations",
        },
        "docs": "/docs",
        "example_request": {
            "url": "POST /chat",
            "body": {
                "messages": [
                    {"role": "user", "content": "I need to hire a senior Java developer"}
                ]
            }
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    messages = [m.model_dump() for m in request.messages]

    # ── Hard turn cap ──────────────────────────────────────────────────────────
    if len(messages) > MAX_TURNS:
        return ChatResponse(
            reply="We've reached the maximum conversation length. Based on our discussion, here's my final recommendation.",
            recommendations=[],
            end_of_conversation=True,
        )

    # ── Extract context (deterministic, no LLM) ────────────────────────────────
    ctx: ConversationContext = extract_context(messages)
    state = decide_state(ctx)

    logger.info(
        f"turns={len(messages)} state={state} "
        f"job_level={ctx.job_level} vague={ctx.is_vague} "
        f"types={ctx.test_type_codes}"
    )

    # ── Fast-path refusals (no LLM needed) ────────────────────────────────────
    if ctx.is_injection:
        return ChatResponse(
            reply=INJECTION_REFUSAL_REPLY,
            recommendations=[],
            end_of_conversation=False,
        )
    if ctx.is_off_topic:
        return ChatResponse(
            reply=REFUSAL_REPLY,
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Build system prompt ────────────────────────────────────────────────────
    retriever = get_retriever()
    system = SYSTEM_PROMPT.format(catalog_size=retriever.catalog_size())

    # ── Retrieve relevant catalog items ───────────────────────────────────────
    retrieved_items: list[dict] = []
    if state in (STATE_RECOMMEND, STATE_REFINE, STATE_COMPARE):
        query = ctx.role_description or " ".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        retrieved_items = retriever.retrieve(
            query,
            job_level=ctx.job_level,
            type_codes=ctx.test_type_codes if ctx.test_type_codes else None,
            top_k=10,
        )

        if state == STATE_COMPARE and ctx.compare_names:
            # Add named items to context even if retrieval missed them
            for name in ctx.compare_names:
                item = retriever.get_by_name(name)
                if item and item not in retrieved_items:
                    retrieved_items.insert(0, item)

    # ── Augment messages with catalog context ──────────────────────────────────
    augmented_messages = list(messages)
    if retrieved_items:
        catalog_ctx = format_catalog_context(retrieved_items[:10])
        if state == STATE_COMPARE:
            context_injection = COMPARE_CONTEXT_PROMPT.format(items_context=catalog_ctx)
        else:
            context_injection = RECOMMENDATION_CONTEXT_PROMPT.format(items_context=catalog_ctx)
        # Inject as a system-style user message before the last user message
        augmented_messages = messages[:-1] + [
            {"role": "user", "content": context_injection},
            {"role": "assistant", "content": "I have reviewed the catalog. Now responding to the user."},
            messages[-1],
        ]

    # ── LLM call ──────────────────────────────────────────────────────────────
    remaining = max(5, INTERNAL_TIMEOUT - int(time.time() - start))
    llm_response = chat_completion(system, augmented_messages, timeout=remaining)

    # ── Post-process: enforce catalog-only URLs ────────────────────────────────
    safe_recs = _filter_catalog_urls(llm_response.get("recommendations", []), retriever)

    # ── Determine end_of_conversation ─────────────────────────────────────────
    eoc = llm_response.get("end_of_conversation", False)
    if len(messages) >= MAX_TURNS - 1 and safe_recs:
        eoc = True  # Force close at turn cap

    return ChatResponse(
        reply=llm_response.get("reply", ""),
        recommendations=[
            Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
            for r in safe_recs
        ],
        end_of_conversation=eoc,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _filter_catalog_urls(
    recs: list[dict],
    retriever,
) -> list[dict]:
    """
    Hard guardrail: drop any recommendation whose URL is not in the catalog.
    Prevents hallucinated URLs from reaching the evaluator.
    """
    catalog_urls = {item["url"] for item in retriever._catalog}
    safe = []
    for r in recs:
        url = r.get("url", "")
        if url in catalog_urls:
            safe.append(r)
        else:
            logger.warning(f"Dropped hallucinated URL: {url}")
    return safe
