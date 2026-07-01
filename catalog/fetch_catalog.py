"""
fetch_catalog.py
Downloads the SHL product catalog JSON and normalizes it for retrieval.
"""
import json
import re
import urllib.request
from pathlib import Path

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = Path(__file__).parent / "shl_catalog.json"

# Mapping from keys[] string → single-letter test_type code used in API response
KEY_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def normalize_duration(raw: str) -> str:
    """Extract a clean duration string from raw text."""
    if not raw:
        return ""
    m = re.search(r"(\d+)\s*minutes?", raw, re.I)
    if m:
        return f"{m.group(1)} minutes"
    if "untimed" in raw.lower():
        return "Untimed"
    return raw.strip()


def fetch_and_normalize() -> list[dict]:
    """Fetch catalog JSON and return normalized list of products."""
    print(f"Fetching catalog from {CATALOG_URL} ...")
    with urllib.request.urlopen(CATALOG_URL, timeout=60) as resp:
        content = resp.read()
    # Use strict=False to handle embedded control characters in scraped descriptions
    try:
        raw: list[dict] = json.loads(content, strict=False)
    except json.JSONDecodeError:
        # Fallback: decode bytes and replace control chars
        text = content.decode("utf-8", errors="replace")
        import re as _re
        text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
        raw: list[dict] = json.loads(text)

    print(f"  Got {len(raw)} raw entries.")
    normalized = []
    for item in raw:
        if item.get("status") != "ok":
            continue
        keys = item.get("keys", [])
        # Derive primary test_type from first key
        test_type = KEY_TO_TYPE.get(keys[0], "K") if keys else "K"
        # All type codes for this product
        type_codes = list({KEY_TO_TYPE.get(k, "K") for k in keys})

        entry = {
            "entity_id": item["entity_id"],
            "name": item["name"],
            "url": item["link"],
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": normalize_duration(item.get("duration_raw", "")),
            "remote": item.get("remote", "no") == "yes",
            "adaptive": item.get("adaptive", "no") == "yes",
            "test_type": test_type,        # primary type code
            "type_codes": type_codes,      # all type codes
            "keys": keys,                  # original category strings
            # Combined text field for embedding / BM25
            "search_text": _build_search_text(item, keys),
        }
        normalized.append(entry)

    print(f"  Normalized {len(normalized)} valid entries.")
    return normalized


def _build_search_text(item: dict, keys: list[str]) -> str:
    """Concatenate all searchable fields into a single string."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("job_levels", [])),
        " ".join(keys),
    ]
    return " ".join(p for p in parts if p)


def main():
    catalog = fetch_and_normalize()
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"Saved to {OUTPUT_PATH}")
    # Print sample
    sample = catalog[0]
    print(f"\nSample entry: {sample['name']}")
    print(f"  test_type={sample['test_type']}, job_levels={sample['job_levels'][:2]}")
    print(f"  url={sample['url']}")


if __name__ == "__main__":
    main()
