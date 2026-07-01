"""debug_retrieval.py - quick debug for missed traces"""
import sys
sys.path.insert(0, '.')
from retrieval.retriever import HybridRetriever

r = HybridRetriever()

cases = [
    ("C9 - full-stack JD",
     "Senior Full-Stack Engineer 5+ years Core Java Spring REST Angular SQL relational databases AWS Docker microservice CI/CD mentor"),
    ("C7 - HIPAA healthcare admin",
     "bilingual healthcare admin South Texas patient records HIPAA compliance Spanish medical terminology Microsoft Word dependability"),
    ("C4 - graduate financial analyst",
     "graduate financial analysts final-year students numerical reasoning finance knowledge accounting statistics situational judgement"),
    ("C2 - Rust networking engineer",
     "senior Rust engineer high-performance networking infrastructure Linux systems cognitive ability OPQ"),
    ("C5 - sales org reskilling",
     "sales organization re-skill restructuring annual talent audit OPQ motivation global skills"),
]

for label, query in cases:
    results = r.retrieve(query, top_k=10)
    print(f"\n{label}:")
    for i, item in enumerate(results, 1):
        print(f"  {i:2d}. [{item['test_type']}] {item['name']}")
