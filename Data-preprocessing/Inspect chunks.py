"""
Quick chunk inspector — shows full text of retrieved chunks for a query.
Run: python inspect_chunks.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from RAG_Engine import ChromaRetriever, QueryPreprocessor

queries = [
    ("What are the general exclusions in a standard IRDAI health insurance policy?", "insurance_policy"),
    ("What is the CGHS rate for a specialist consultation at a Non-NABH hospital?", "medical_billing"),
]

retriever    = ChromaRetriever()
preprocessor = QueryPreprocessor()

for question, category in queries:
    print(f"\n{'='*65}")
    print(f"  Q: {question}")
    print(f"{'='*65}")
    processed_q, _ = preprocessor.process(question)
    chunks = retriever.retrieve(processed_q, n_results=4, category_filter=category)
    for i, c in enumerate(chunks, 1):
        print(f"\n  [{i}] {c.filename} | p.{c.page_number} | score:{c.score:.3f}")
        print(f"  Section: {c.section_title}")
        print(f"  {'─'*55}")
        print(f"  {c.text[:400]}")
        print()