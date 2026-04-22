import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from RAG_Engine import ChromaRetriever, QueryPreprocessor

# All 25 evaluation questions
EVAL_QUESTIONS = [
    ("INS_001", "insurance_policy",  "What is the waiting period for pre-existing diseases under a standard IRDAI health insurance policy?"),
    ("INS_002", "insurance_policy",  "What penalty applies if a claim is not intimated within the stipulated timeframe under IRDAI regulations?"),
    ("INS_003", "insurance_policy",  "What is the minimum hospitalisation period required for a claim to be admissible?"),
    ("INS_004", "insurance_policy",  "What is covered under post-hospitalisation expenses in a health insurance policy?"),
    ("INS_005", "insurance_policy",  "What are the general exclusions in a standard IRDAI health insurance policy?"),
    ("INS_006", "insurance_policy",  "What is the maximum amount covered for emergency ambulance transportation under an IRDAI health insurance policy?"),
    ("INS_007", "insurance_policy",  "What is the role of the Insurance Ombudsman in resolving policyholder grievances under IRDAI regulations?"),
    ("INS_008", "insurance_policy",  "What is the coverage for AYUSH treatment under a health insurance policy?"),
    ("AUD_001", "financial_audit",   "What are the Key Audit Matters identified in the financial audit report and what risks do they highlight?"),
    ("AUD_002", "financial_audit",   "What TDS compliance issue was noted in the CAG audit report?"),
    ("AUD_003", "financial_audit",   "Under which accounting standard is revenue recognition audited for Indian listed companies?"),
    ("AUD_004", "financial_audit",   "What is the auditor's responsibility under Section 143(10) of the Companies Act 2013?"),
    ("AUD_005", "financial_audit",   "What auditing standards must Indian auditors follow when conducting a statutory audit under the Companies Act 2013?"),
    ("AUD_006", "financial_audit",   "What GST reconciliation issue was flagged in the management letter points of the audit?"),
    ("AUD_007", "financial_audit",   "Which SEBI regulation governs related party transactions for listed healthcare companies?"),
    ("MED_001", "medical_billing",   "What is the CGHS rate for a specialist consultation at a Non-NABH empanelled hospital?"),
    ("MED_002", "medical_billing",   "What is the penalty for a CGHS empanelled hospital charging above the prescribed package rates?"),
    ("MED_003", "medical_billing",   "What GST rate applies to hospital services under the GST Council circular?"),
    ("MED_004", "medical_billing",   "Within how many days must a patient submit the discharge summary and bills to the TPA for cashless claim processing?"),
    ("MED_005", "medical_billing",   "What coding system is used for diagnosis coding in Indian hospitals under CGHS?"),
    ("MED_006", "medical_billing",   "What is the CGHS rate for an ICU stay per day at a NABH accredited hospital?"),
    ("BNK_001", "banking_finance",   "What does the RBI master circular say about premature termination of term deposits in case of death of depositor?"),
    ("BNK_002", "banking_finance",   "What is the purpose of RBI Master Directions on Priority Sector Lending?"),
    ("BNK_003", "banking_finance",   "What are the KYC and AML requirements for banks under RBI guidelines?"),
    ("BNK_004", "banking_finance",   "What is the Fair Practices Code requirement for NBFCs under RBI master circular?"),
]

COVERAGE_THRESHOLD   = 0.45   # below this = poor coverage
ACCEPTABLE_THRESHOLD = 0.55   # above this = good coverage

def main():
    print("\n" + "="*65)
    print("  Corpus Coverage Diagnostic — 25 Evaluation Questions")
    print("="*65)

    retriever   = ChromaRetriever()
    preprocessor = QueryPreprocessor()

    poor      = []   # score < 0.45 — remove/replace
    moderate  = []   # 0.45-0.55 — keep but note
    good      = []   # > 0.55 — keep

    print(f"\n  {'ID':<10} {'Score':>7} {'Status':<12} Question")
    print(f"  {'─'*10} {'─'*7} {'─'*12} {'─'*40}")

    for q_id, category, question in EVAL_QUESTIONS:
        processed_q, _ = preprocessor.process(question)
        chunks = retriever.retrieve(
            processed_q,
            n_results=3,
            category_filter=category
        )

        top_score = chunks[0].score if chunks else 0.0
        top_section = chunks[0].section_title[:30] if chunks else "N/A"

        if top_score < COVERAGE_THRESHOLD:
            status = "❌ REMOVE"
            poor.append((q_id, top_score, question))
        elif top_score < ACCEPTABLE_THRESHOLD:
            status = "⚠️  MODERATE"
            moderate.append((q_id, top_score, question))
        else:
            status = "✅ GOOD"
            good.append((q_id, top_score, question))

        short_q = question[:45] + "..." if len(question) > 45 else question
        print(f"  {q_id:<10} {top_score:>7.3f} {status:<12} {short_q}")

    # Summary
    print(f"\n  {'='*65}")
    print(f"  SUMMARY")
    print(f"  {'─'*65}")
    print(f"  ✅ Good coverage  (score > 0.55) : {len(good):>3} questions — keep")
    print(f"  ⚠️  Moderate       (0.45 - 0.55)  : {len(moderate):>3} questions — keep, note")
    print(f"  ❌ Poor coverage  (score < 0.45) : {len(poor):>3} questions — replace")

    if poor:
        print(f"\n  Questions to REPLACE (no meaningful corpus coverage):")
        for q_id, score, q in poor:
            print(f"    {q_id} [{score:.3f}] — {q[:60]}")

    if moderate:
        print(f"\n  Questions to REVIEW (marginal coverage):")
        for q_id, score, q in moderate:
            print(f"    {q_id} [{score:.3f}] — {q[:60]}")

    print(f"\n  {'='*65}")
    print(f"  ACTION: Replace the {len(poor)} poor-coverage questions with")
    print(f"  questions that are directly answerable from your corpus.")
    print(f"  This will meaningfully improve Recall and Precision scores.")
    print(f"  {'='*65}\n")


if __name__ == "__main__":
    main()