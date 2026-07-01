"""Live end-to-end validation test for the SHL Agent API."""
import urllib.request
import json

BASE = "http://127.0.0.1:8000"


def post_chat(messages):
    payload = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat", data=payload,
        headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=35)
    return json.loads(r.read().decode())


# Test 1: /health
r = urllib.request.urlopen(f"{BASE}/health")
health = json.loads(r.read().decode())
print("TEST 1 /health:", health)
assert health == {"status": "ok"}, "FAIL"

# Test 2: Senior Java engineer -> RECOMMEND with real catalog URLs
resp2 = post_chat([{"role": "user", "content": (
    "Hiring a senior Java backend engineer with 8 years experience. "
    "Spring, SQL, AWS, Docker. Need technical tests and personality."
)}])
print("\nTEST 2 /chat - Senior Java engineer")
print("reply:", resp2["reply"][:200])
print("recommendations:", len(resp2["recommendations"]), "items")
all_valid = True
for rec in resp2["recommendations"]:
    valid_url = "shl.com" in rec["url"]
    status = "OK" if valid_url else "FAIL"
    all_valid = all_valid and valid_url
    print(f"  [{status}] {rec['name']} | {rec['test_type']} | {rec['url'][:60]}")
print("eoc:", resp2["end_of_conversation"])
assert len(resp2["recommendations"]) > 0, "No recommendations!"
assert all_valid, "Bad URL found!"

# Test 3: Off-topic -> REFUSE (0 recommendations)
resp3 = post_chat([{"role": "user", "content": (
    "What is the average salary for a Java engineer in London?"
)}])
print("\nTEST 3 /chat - Salary question (off-topic)")
print("reply:", resp3["reply"][:200])
print("recommendations:", len(resp3["recommendations"]), "(should be 0)")
assert len(resp3["recommendations"]) == 0, "Should refuse with 0 recs!"

# Test 4: Vague query -> clarify
resp4 = post_chat([{"role": "user", "content": "I need some tests"}])
print("\nTEST 4 /chat - Vague query (should clarify)")
print("reply:", resp4["reply"][:200])
print("recommendations:", len(resp4["recommendations"]), "(should be 0)")

print("\n=============================")
print("ALL LIVE TESTS PASSED")
print("=============================")
