"""
eval_harness.py
Recall@10 evaluation against C1-C10 ground truth traces.
Run: python eval/eval_harness.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.traces import TRACES
from retrieval.retriever import get_retriever
from agent.state_machine import extract_context, format_catalog_context


def recall_at_k(predicted_urls: list[str], ground_truth_urls: list[str], k: int = 10) -> float:
    """
    Recall@K = |relevant ∩ top-K| / |relevant|
    """
    if not ground_truth_urls:
        return 1.0
    top_k = set(predicted_urls[:k])
    relevant = set(ground_truth_urls)
    return len(top_k & relevant) / len(relevant)


def evaluate_retrieval_only():
    """
    Fast evaluation using the retriever directly (no LLM, no API).
    Shows baseline Recall@10 before any LLM reranking.
    """
    retriever = get_retriever()
    results = []

    print("\n" + "=" * 70)
    print("SHL Agent — Retrieval Recall@10 Evaluation (C1–C10)")
    print("=" * 70)

    for trace in TRACES:
        # Build query from seed messages
        query = " ".join(
            m["content"] for m in trace["seed_messages"] if m["role"] == "user"
        )

        # Extract context for filters
        ctx = extract_context(trace["seed_messages"])

        # Retrieve
        retrieved = retriever.retrieve(
            query,
            job_level=ctx.job_level,
            type_codes=ctx.test_type_codes if ctx.test_type_codes else None,
            top_k=10,
        )
        predicted_urls = [item["url"] for item in retrieved]
        gt_urls = [r["url"] for r in trace["final_recommendations"]]

        r10 = recall_at_k(predicted_urls, gt_urls, k=10)
        results.append(r10)

        status = "✓" if r10 >= 0.8 else ("~" if r10 >= 0.5 else "✗")
        print(f"\n[{status}] {trace['id']}: {trace['description'][:55]}")
        print(f"   Recall@10 = {r10:.3f}  ({int(r10 * len(gt_urls))}/{len(gt_urls)} found)")
        print(f"   Ground truth: {[r['name'][:30] for r in trace['final_recommendations']]}")

        missed = [url for url in gt_urls if url not in set(predicted_urls)]
        if missed:
            # Find names for missed URLs
            url_to_name = {r["url"]: r["name"] for r in trace["final_recommendations"]}
            print(f"   Missed: {[url_to_name.get(u, u) for u in missed]}")

    mean_r10 = sum(results) / len(results)
    print("\n" + "=" * 70)
    print(f"Mean Recall@10 across {len(TRACES)} traces: {mean_r10:.4f}")
    print("=" * 70)
    return mean_r10


def print_catalog_match_debug(trace_id: str):
    """Debug helper: show what the retriever returns for a specific trace."""
    retriever = get_retriever()
    trace = next(t for t in TRACES if t["id"] == trace_id)
    query = " ".join(m["content"] for m in trace["seed_messages"] if m["role"] == "user")
    retrieved = retriever.retrieve(query, top_k=10)
    print(f"\nTrace {trace_id}: top-10 retrieved items:")
    for i, item in enumerate(retrieved, 1):
        print(f"  {i}. {item['name']} ({item['test_type']})")
        print(f"     {item['url']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_catalog_match_debug(sys.argv[1])
    else:
        evaluate_retrieval_only()
