import sys
import json
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path().parent))
from  RAG_Engine import RagEngine

SMOKE_TEST_QUERIES = [
    # Insurance
    (
        "What is the penalty for late claim submission under IRDAI regulations?",
        "insurance_policy",
    ),
    (
        "What pre-existing diseases are excluded in the first 48 months of a health policy?",
        "insurance_policy",
    ),
    # Medical Billing
    (
        "What are the CGHS rates for total knee replacement surgery?",
        "medical_billing",
    ),
    (
        "What is the GST rate applicable on hospital services as per CGHS billing?",
        "medical_billing",
    ),
    # Financial Audit
    (
        "What are the key audit matters related to revenue recognition in healthcare?",
        "financial_audit",
    ),
    # Multi-category
    (
        "What documents are required for reimbursement of CGHS claims?",
        None,   # no category filter — tests cross-category retrieval
    ),
]

ABLATION_QUERIES = [
    "What are the exclusions in a standard IRDAI health insurance policy?",
    "What is the penalty for charging above CGHS rates?",
    "What TDS compliance issue was noted in the audit report?",
]


def run_test(engine: RagEngine):
    passed = 0
    failed = 0

    for i, (query,cat) in enumerate(SMOKE_TEST_QUERIES, 1):
        try:
            result = engine.query(
                query,
                mode = 'llama3',
                category_filter=cat,
                conversation_id='smoke_test')

            has_answer = len(result.answer) > 50
            has_citations = len(result.citations) > 0
            reasonable_latency = result.latency_ms <60000

            status = "PASS" if (has_answer and has_citations) else "WARN"
            marker = "✓" if status == "PASS" else "⚠"

            if has_answer:
                # Show first 200 chars of answer
                preview = result.answer[:200].replace("\n", " ")
                print(f"     Answer: {preview}...")

            if status == "PASS":
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  ✗ FAIL | Error: {e}")
            failed += 1

        return passed, failed


def run_ablation(engine: RagEngine):
    all_results = []
    for query in ABLATION_QUERIES:
        comparison = engine.ablation_compare(query)
        all_results.append({"query": query, "results": comparison})

        for model, data in comparison.items():
            if "error" in data:
                print(f"  [{model:<15}] ERROR: {data['error']}")
                continue

            latency = data["latency_ms"]
            answer = data["answer"][:180].replace("\n", " ")
            print(f"\n  [{model:<15}] {latency}ms")
            print(f"  Answer : {answer}...")
    output_path = Path("data/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Ablation results saved: {output_path}")
    print(f"\n  {'─' * 55}")
    print(f"  {'Model':<20} {'Avg Latency':>12} {'Errors':>8}")
    print(f"  {'─' * 55}")
    for model in ["llama3", "roberta", "distilroberta"]:
        latencies = [
            r["results"][model]["latency_ms"]
            for r in all_results
            if "error" not in r["results"].get(model, {})
        ]
        errors = sum(
            1 for r in all_results
            if "error" in r["results"].get(model, {})
        )
        avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0
        print(f"  {model:<20} {avg_lat:>10}ms {errors:>8}")
    print(f"  {'─' * 55}\n")


def run_interactive(engine: RagEngine):
    session_id = f"session_{int(time.time())}"
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("  Goodbye!")
            break

        if user_input.lower() == "clear":
            engine.clear_memory(session_id)
            print("  Memory cleared.\n")
            continue
        mode = "llama3"
        query = user_input
        if user_input.lower().startswith("roberta:"):
            mode = "roberta"
            query = user_input[8:].strip()
        elif user_input.lower().startswith("distil:"):
            mode = "distilroberta"
            query = user_input[7:].strip()

        result = engine.query(query, mode=mode, conversation_id=session_id)
        print(result.pretty())


def main():
    parser = argparse.ArgumentParser(
        description="MediFinance RAG Engine — Smoke Test & CLI"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive Q&A session"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study across all 3 models"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single query across all 3 modes"
    )
    args = parser.parse_args()
    engine = RagEngine()

    if args.query:
        print(f"\n  Running: '{args.query}'")
        for mode in ["llama3", "roberta", "distilroberta"]:
            try:
                r = engine.query(args.query, mode=mode)
                print(r.pretty())
            except Exception as e:
                print(f"  [{mode}] Error: {e}")

    elif args.ablation:
        run_ablation(engine)

    elif args.interactive:
        run_interactive(engine)

    else:
        run_test(engine)


if __name__ == "__main__":
    main()



