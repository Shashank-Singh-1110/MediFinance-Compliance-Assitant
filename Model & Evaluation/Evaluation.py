import os
import re
import json
import time
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from RAG_Engine import RagEngine, QueryResult

RAG_OK =True

EVAL_DIR = Path("data/evaluation")
RESULTS_FILE = EVAL_DIR / "eval_results.json"
REPORT_FILE = EVAL_DIR / "eval_report.csv"
SUMMARY_FILE = EVAL_DIR / "eval_summary.json"
LOG_DIR = Path("logs")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
JUDGE_TIMEOUT = 90
MODELS_TO_EVAL = ["llama3", "roberta", "distilroberta"]

LOG_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Evaluator")

EVAL_DATASET = [
    {
        "id": "INS_001",
        "category": "insurance_policy",
        "question": "What is the waiting period for pre-existing diseases under a standard IRDAI health insurance policy?",
        "ground_truth": "The waiting period for pre-existing diseases is 48 months (4 years) of continuous coverage from the policy inception date.",
        "difficulty": "medium",
    },
    {
        "id": "INS_002",
        "category": "insurance_policy",
        "question": "What penalty applies if a claim is not intimated within the stipulated timeframe under IRDAI regulations?",
        "ground_truth": "Claims not intimated within the stipulated timeframe may be rejected or subject to a 10% penalty on the admissible claim amount as per IRDAI circular IRDAI/HLT/REG/CIR/194/08/2020.",
        "difficulty": "medium",
    },
    {
        "id": "INS_003",
        "category": "insurance_policy",
        "question": "What is the minimum hospitalisation period required for a claim to be admissible under a standard health insurance policy?",
        "ground_truth": "A minimum of 24 hours of hospitalisation is required for a claim to be admissible, except for Day Care procedures listed under Annexure II.",
        "difficulty": "easy",
    },
    {
        "id": "INS_004",
        "category": "insurance_policy",
        "question": "What is covered under post-hospitalisation expenses in a health insurance policy?",
        "ground_truth": "Post-hospitalisation medical expenses incurred up to 90 days after discharge are covered up to 10% of Sum Insured or Rs. 50,000 whichever is lower.",
        "difficulty": "medium",
    },
    {
        "id": "INS_005",
        "category": "insurance_policy",
        "question": "What are the general exclusions in a standard IRDAI health insurance policy?",
        "ground_truth": "General exclusions include pre-existing diseases during the first 48 months, diseases contracted within the first 30 days, self-inflicted injuries, cosmetic treatments, obesity treatments, war-related injuries, dental treatment unless from accident, and experimental treatments not approved by DCGI.",
        "difficulty": "hard",
    },
    {
        "id": "INS_006",
        "category": "insurance_policy",
        "question": "What is the maximum amount covered for emergency ambulance transportation under an IRDAI health insurance policy?",
        "ground_truth": "Emergency ambulance charges are covered up to Rs. 5,000 per hospitalisation.",
        "difficulty": "easy",
    },
    {
        "id": "INS_007",
        "category": "insurance_policy",
        "question": "What is the role of the Insurance Ombudsman in resolving policyholder grievances under IRDAI regulations?",
        "ground_truth": "If unresolved within 30 days, the complaint may be escalated to the Insurance Ombudsman under Rule 13 of the Insurance Ombudsman Rules 2017, or to IRDAI's Bima Bharosa portal at igms.irda.gov.in.",
        "difficulty": "medium",
    },
    {
        "id": "INS_008",
        "category": "insurance_policy",
        "question": "What is the coverage for AYUSH treatment under a health insurance policy?",
        "ground_truth": "Treatment under Ayurveda, Yoga, Unani, Siddha, and Homeopathy in a registered AYUSH Hospital is covered up to Rs. 25,000 per policy year.",
        "difficulty": "medium",
    },

    # ── FINANCIAL AUDIT (7 questions) ────────────────────
    {
        "id": "AUD_001",
        "category": "financial_audit",
        "question": "What are the Key Audit Matters identified in the financial audit report and what risks do they highlight?",
        "ground_truth": "Key Audit Matters are those matters that in the auditor's professional judgement were of most significance in the audit of the financial statements. They are included to provide transparency about the most significant risks and areas requiring special attention during the audit.",
        "difficulty": "medium",
    },
    {
        "id": "AUD_002",
        "category": "financial_audit",
        "question": "What TDS compliance issue was noted in the CAG audit report?",
        "ground_truth": "The audit noticed that the Assessing Officer (TDS) did not invoke provisions of Section 276B/276BB/278A against deductors where tax was deducted but not deposited to the government treasury.",
        "difficulty": "medium",
    },
    {
        "id": "AUD_003",
        "category": "financial_audit",
        "question": "Under which accounting standard is revenue recognition audited for Indian listed companies?",
        "ground_truth": "Revenue recognition is audited under Indian Accounting Standard Ind AS 115, which governs revenue from contracts with customers.",
        "difficulty": "medium",
    },
    {
        "id": "AUD_004",
        "category": "financial_audit",
        "question": "What is the auditor's responsibility under Section 143(10) of the Companies Act 2013?",
        "ground_truth": "Under Section 143(10) of the Companies Act 2013, auditors must conduct their audit in accordance with the Standards on Auditing specified by the Institute of Chartered Accountants of India (ICAI).",
        "difficulty": "hard",
    },
    {
        "id": "AUD_005",
        "category": "financial_audit",
        "question": "What auditing standards must Indian auditors follow when conducting a statutory audit under the Companies Act 2013?",
        "ground_truth": "Auditors must conduct their audit in accordance with the Standards on Auditing specified under Section 143(10) of the Companies Act 2013, as issued by the Institute of Chartered Accountants of India (ICAI).",
        "difficulty": "easy",
    },
    {
        "id": "AUD_006",
        "category": "financial_audit",
        "question": "What GST reconciliation issue was flagged in the management letter points of the audit?",
        "ground_truth": "Discrepancies were identified between GSTR-2A auto-populated data and the purchase register, with management directed to reconcile and file amended returns under Section 39 of the CGST Act 2017.",
        "difficulty": "hard",
    },
    {
        "id": "AUD_007",
        "category": "financial_audit",
        "question": "Which SEBI regulation governs related party transactions for listed healthcare companies?",
        "ground_truth": "SEBI (Listing Obligations and Disclosure Requirements) Regulations 2015, also known as SEBI LODR Regulations, govern related party transactions for listed companies including healthcare entities.",
        "difficulty": "medium",
    },

    # ── MEDICAL BILLING (6 questions) ────────────────────
    {
        "id": "MED_001",
        "category": "medical_billing",
        "question": "What is the CGHS rate for a specialist consultation at a Non-NABH empanelled hospital?",
        "ground_truth": "The CGHS rate for a specialist consultation at a Non-NABH hospital is Rs. 350.",
        "difficulty": "easy",
    },
    {
        "id": "MED_002",
        "category": "medical_billing",
        "question": "What is the penalty for a CGHS empanelled hospital charging above the prescribed package rates?",
        "ground_truth": "Hospitals found charging beyond CGHS rates shall be liable for immediate suspension of empanelment and recovery of the excess amount charged with 24% interest per annum.",
        "difficulty": "medium",
    },
    {
        "id": "MED_003",
        "category": "medical_billing",
        "question": "What GST rate applies to hospital services under the GST Council circular?",
        "ground_truth": "Pure healthcare services are exempt from GST under Notification No. 12/2017. GST at 5% is applicable on non-healthcare services as per GST Council Circular No. 32/06/2018-GST.",
        "difficulty": "hard",
    },
    {
        "id": "MED_004",
        "category": "medical_billing",
        "question": "Within how many days must a patient submit the discharge summary and bills to the TPA for cashless claim processing?",
        "ground_truth": "All cashless claim documents must be submitted within 15 days of discharge. Delay beyond 30 days may result in claim rejection.",
        "difficulty": "medium",
    },
    {
        "id": "MED_005",
        "category": "medical_billing",
        "question": "What coding system is used for diagnosis coding in Indian hospitals under CGHS?",
        "ground_truth": "ICD-10 (International Classification of Diseases, 10th Revision) as adapted by the Ministry of Health and Family Welfare (MoHFW) Government of India is used for diagnosis coding.",
        "difficulty": "easy",
    },
    {
        "id": "MED_006",
        "category": "medical_billing",
        "question": "What is the CGHS rate for an ICU stay per day at a NABH accredited hospital?",
        "ground_truth": "The CGHS rate for ICU Monitoring per day at a NABH accredited hospital is Rs. 4,000.",
        "difficulty": "medium",
    },

    # ── BANKING FINANCE (4 questions) ────────────────────
    {
        "id": "BNK_001",
        "category": "banking_finance",
        "question": "What does the RBI master circular say about premature termination of term deposits in case of death of depositor?",
        "ground_truth": "In case of death of the depositor, premature termination of term deposits is allowed without attracting any penal charge. Banks must incorporate a clause in the account opening form specifying the conditions for such premature withdrawal.",
        "difficulty": "medium",
    },
    {
        "id": "BNK_002",
        "category": "banking_finance",
        "question": "What is the purpose of RBI Master Directions on Priority Sector Lending?",
        "ground_truth": "RBI Master Directions on Priority Sector Lending define which sectors banks must lend to mandatorily, including agriculture, MSMEs, education, housing, and social infrastructure, to ensure credit flow to underserved sectors of the economy.",
        "difficulty": "medium",
    },
    {
        "id": "BNK_003",
        "category": "banking_finance",
        "question": "What are the KYC and AML requirements for banks under RBI guidelines?",
        "ground_truth": "Banks must conduct Customer Due Diligence (CDD), maintain customer identification records, monitor transactions for suspicious activity, and report suspicious transactions to the Financial Intelligence Unit India (FIU-IND) under the Prevention of Money Laundering Act.",
        "difficulty": "hard",
    },
    {
        "id": "BNK_004",
        "category": "banking_finance",
        "question": "What is the Fair Practices Code requirement for NBFCs under RBI master circular?",
        "ground_truth": "NBFCs must adopt a Fair Practices Code covering loan application processing, loan appraisal, disbursement conditions, interest rate policy, and grievance redressal mechanism, ensuring transparency and fair treatment of borrowers.",
        "difficulty": "medium",
    },
]


@dataclass
class MetricScores:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    rouge1:             float = 0.0
    rougeL:             float = 0.0
    bleu:               float = 0.0
    def average(self) -> float:
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_precision + self.context_recall) / 4, 4
        )


@dataclass
class EvalRecord:
    question_id: str
    category: str
    question: str
    ground_truth: str
    difficulty: str
    model: str
    answer: str
    latency_ms: int
    chunks_used: int
    top_retrieval_score: float
    scores: MetricScores
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class RAGASEvaluator:
    def __init__(self):
        self._verify_ollama()

    def _verify_ollama(self):
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                logger.info(f"  Judge model: {OLLAMA_MODEL} ready ✓")
            else:
                logger.warning(f"  Judge model '{OLLAMA_MODEL}' not found")
        except Exception:
            logger.warning("  Ollama not reachable — scores will be 0")

    def _judge(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # deterministic judge
                        "num_predict": 200,
                    },
                },
                timeout=JUDGE_TIMEOUT,
            )
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"  Judge error: {e}")
            return ""

    def _extract_score(self, text: str) -> float:
        patterns = [
            r'\b(0\.\d+|1\.0)\b',  # decimal 0.x or 1.0
            r'\b([0-9])\s*/\s*10\b',  # x/10
            r'\b([0-9])\s*/\s*5\b',  # x/5
            r'score[:\s]+([0-9.]+)',  # score: x
            r'\b([01])\b',  # binary 0 or 1
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                # Normalise to 0-1
                if '/10' in text and val <= 10:
                    return round(val / 10, 3)
                if '/5' in text and val <= 5:
                    return round(val / 5, 3)
                if val <= 1.0:
                    return round(val, 3)
        return 0.5
    def faithfulness(self, answer: str, context: str) -> float:
        if not answer or not context:
            return 0.0

        prompt = f"""You are evaluating whether an answer is faithful to its source context.

    CONTEXT:
    {context[:2000]}

    ANSWER:
    {answer[:800]}

    TASK: 
    1. List each factual claim in the answer (max 5 claims)
    2. For each claim, state if it is SUPPORTED or NOT SUPPORTED by the context
    3. Calculate: supported_claims / total_claims
    4. Output only a decimal score between 0.0 and 1.0

    Example output format:
    Claim 1: "X" - SUPPORTED
    Claim 2: "Y" - NOT SUPPORTED
    Score: 0.5

    Your evaluation:"""

        response = self._judge(prompt)
        return self._extract_score(response)

    def answer_relevancy(self, question: str, answer: str) -> float:
        if not answer or not question:
            return 0.0

        prompt = f"""You are evaluating whether an answer is relevant to a question.

    QUESTION: {question}

    ANSWER: {answer[:600]}

    TASK: Rate how well the answer addresses the question on a scale of 0.0 to 1.0 where:
      1.0 = Answer directly and completely addresses the question
      0.7 = Answer mostly addresses the question with minor gaps
      0.5 = Answer partially addresses the question
      0.3 = Answer is tangentially related but doesn't answer the question
      0.0 = Answer is completely irrelevant or says "I don't know"

    Consider:
    - Does the answer contain information relevant to the question?
    - Does it avoid going off-topic?
    - If it says information is not in documents, score 0.2 (honest but unhelpful)

    Output only a decimal score between 0.0 and 1.0:"""

        response = self._judge(prompt)
        return self._extract_score(response)

    def context_precision(self, question: str, context: str) -> float:
        if not context or not question:
            return 0.0

        prompt = f"""You are evaluating whether retrieved document chunks are relevant to a question.

    QUESTION: {question}

    RETRIEVED CONTEXT:
    {context[:2000]}

    TASK: For each [SOURCE N] chunk in the context above:
    1. State if it is RELEVANT or NOT RELEVANT to answering the question
    2. Calculate: relevant_chunks / total_chunks
    3. Output a decimal score between 0.0 and 1.0

    Output only a decimal score between 0.0 and 1.0:"""

        response = self._judge(prompt)
        return self._extract_score(response)
    def context_recall(
            self, ground_truth: str, context: str
    ) -> float:
        if not ground_truth or not context:
            return 0.0

        prompt = f"""You are evaluating whether a retrieved context contains the information needed to answer a question correctly.

    GROUND TRUTH ANSWER: {ground_truth}

    RETRIEVED CONTEXT:
    {context[:2000]}

    TASK:
    1. Break the ground truth into key facts/claims
    2. For each fact, check if the context contains that information
    3. Calculate: facts_found_in_context / total_facts
    4. Output a decimal score between 0.0 and 1.0

    Output only a decimal score between 0.0 and 1.0:"""

        response = self._judge(prompt)
        return self._extract_score(response)

    def lexical_scores(self, answer: str, ground_truth: str) -> dict:
        """
        Compute ROUGE-1, ROUGE-L and BLEU against ground truth.
        Pure string comparison — no LLM calls needed.
        """
        if not answer or not ground_truth:
            return {"rouge1": 0.0, "rougeL": 0.0, "bleu": 0.0}

        # ROUGE
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(ground_truth, answer)

        # BLEU — tokenise at word level
        from nltk.tokenize import word_tokenize
        ref = [word_tokenize(ground_truth.lower())]
        hyp = word_tokenize(answer.lower())
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu(ref, hyp, smoothing_function=smooth)

        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
            "bleu": round(bleu, 4),
        }

    def evaluate(
            self,
            question: str,
            answer: str,
            context: str,
            ground_truth: str,
    ) -> MetricScores:
        lexical = self.lexical_scores(answer, ground_truth)
        return MetricScores(
            faithfulness=self.faithfulness(answer, context),
            answer_relevancy=self.answer_relevancy(question, answer),
            context_precision=self.context_precision(question, context),
            context_recall=self.context_recall(ground_truth, context),
            rouge1= lexical["rouge1"],
            rougeL= lexical["rougeL"],
            bleu= lexical["bleu"],
        )


class EvaluationRunner:

    def __init__(self, engine: "RagEngine"):
        self.engine = engine
        self.evaluator = RAGASEvaluator()
        self.records: list[EvalRecord] = []

    def _get_context_from_result(self, result: QueryResult) -> str:
        if not result.citations:
            return ""
        parts = []
        for i, cit in enumerate(result.citations, 1):
            parts.append(
                f"[SOURCE {i}] {cit.filename} | "
                f"Page {cit.page_number} | "
                f"Section: {cit.section_title}"
            )
        return "\n".join(parts)

    def run_question(
            self,
            item: dict,
            model: str,
    ) -> EvalRecord:
        q_id = item["id"]
        question = item["question"]
        gt = item["ground_truth"]
        category = item["category"]
        difficulty = item["difficulty"]

        logger.info(f"  [{q_id}] [{model}] {question[:60]}...")

        # Run RAG query
        result = self.engine.query(
            question,
            mode=model,
            conversation_id=f"eval_{model}_{q_id}",
            category_filter=category,
        )

        context = self._get_context_from_result(result)
        scores = self.evaluator.evaluate(
            question=question,
            answer=result.answer,
            context=context,
            ground_truth=gt,
        )

        top_score = (
            result.citations[0].relevance_score
            if result.citations else 0.0
        )

        record = EvalRecord(
            question_id=q_id,
            category=category,
            question=question,
            ground_truth=gt,
            difficulty=difficulty,
            model=model,
            answer=result.answer[:500],
            latency_ms=result.latency_ms,
            chunks_used=result.chunks_used,
            top_retrieval_score=top_score,
            scores=scores,
        )

        logger.info(
            f"    F:{scores.faithfulness:.2f} "
            f"R:{scores.answer_relevancy:.2f} "
            f"P:{scores.context_precision:.2f} "
            f"Rc:{scores.context_recall:.2f} "
            f"avg:{scores.average():.2f} "
            f"| {result.latency_ms}ms"
        )

        return record

    def run(
            self,
            models: list[str] = None,
            max_questions: int = None,
    ) -> list[EvalRecord]:
        if models is None:
            models = MODELS_TO_EVAL

        dataset = EVAL_DATASET
        if max_questions:
            dataset = dataset[:max_questions]

        total = len(dataset) * len(models)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Module 4 — Evaluation Framework")
        logger.info(f"  Questions : {len(dataset)}")
        logger.info(f"  Models    : {models}")
        logger.info(f"  Total runs: {total}")
        logger.info(f"{'=' * 60}")

        records = []
        done = 0

        for model in models:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"  Evaluating model: {model.upper()}")
            logger.info(f"{'─' * 60}")

            for item in dataset:
                try:
                    record = self.run_question(item, model)
                    records.append(record)
                    done += 1
                    logger.info(
                        f"  Progress: {done}/{total} "
                        f"({100 * done // total}%)"
                    )
                except Exception as e:
                    logger.error(
                        f"  FAILED [{item['id']}] [{model}]: {e}"
                    )
        self.records = records
        return records


class ReportGenerator:
    def generate(self, records: list[EvalRecord]) -> dict:
        if not records:
            logger.error("No records to report on.")
            return {}
        raw = []
        for r in records:
            d = asdict(r)
            d["ragas_average"] = r.scores.average()
            raw.append(d)

        with open(RESULTS_FILE, "w") as f:
            json.dump(raw, f, indent=2)
        logger.info(f"  Raw results: {RESULTS_FILE}")
        rows = []
        for r in records:
            rows.append({
                "question_id": r.question_id,
                "category": r.category,
                "difficulty": r.difficulty,
                "model": r.model,
                "faithfulness": r.scores.faithfulness,
                "answer_relevancy": r.scores.answer_relevancy,
                "context_precision": r.scores.context_precision,
                "context_recall": r.scores.context_recall,
                "ragas_average": r.scores.average(),
                "rouge1": r.scores.rouge1,
                "rougeL": r.scores.rougeL,
                "bleu": r.scores.bleu,
                "latency_ms": r.latency_ms,
                "chunks_used": r.chunks_used,
                "top_retrieval_score": r.top_retrieval_score,
            })

        df = pd.DataFrame(rows)
        df.to_csv(REPORT_FILE, index=False)
        logger.info(f"  CSV report: {REPORT_FILE}")

        summary = {"generated_at": datetime.now().isoformat(), "models": {}}

        for model in df["model"].unique():
            mdf = df[df["model"] == model]
            summary["models"][model] = {
                "faithfulness": round(mdf["faithfulness"].mean(), 4),
                "answer_relevancy": round(mdf["answer_relevancy"].mean(), 4),
                "context_precision": round(mdf["context_precision"].mean(), 4),
                "context_recall": round(mdf["context_recall"].mean(), 4),
                "ragas_average": round(mdf["ragas_average"].mean(), 4),
                "rouge1": round(mdf["rouge1"].mean(), 4),
                "rougeL": round(mdf["rougeL"].mean(), 4),
                "bleu": round(mdf["bleu"].mean(), 4),
                "avg_latency_ms": int(mdf["latency_ms"].mean()),
                "questions_evaluated": len(mdf),
                "by_category": {},
                "by_difficulty": {},
            }
            # Per-category breakdown
            for cat in mdf["category"].unique():
                cdf = mdf[mdf["category"] == cat]
                summary["models"][model]["by_category"][cat] = {
                    "ragas_average": round(cdf["ragas_average"].mean(), 4),
                    "count": len(cdf),
                }
            # Per-difficulty breakdown
            for diff in mdf["difficulty"].unique():
                ddf = mdf[mdf["difficulty"] == diff]
                summary["models"][model]["by_difficulty"][diff] = {
                    "ragas_average": round(ddf["ragas_average"].mean(), 4),
                    "count": len(ddf),
                }

        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary: {SUMMARY_FILE}")
        self._print_report(summary, df)
        return summary

    def _print_report(self, summary: dict, df: pd.DataFrame):
        print("\n" + "=" * 70)
        print("  MODULE 4 — EVALUATION RESULTS")
        print("=" * 70)

        print(f"\n  {'Model':<16} {'Faith':>7} {'Relev':>7} "
              f"{'Prec':>7} {'Recall':>7} {'Avg':>7} {'Latency':>10}")
        print(f"  {'─' * 16} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 10}")

        for model, stats in summary["models"].items():
            print(
                f"  {model:<16} "
                f"{stats['faithfulness']:>7.3f} "
                f"{stats['answer_relevancy']:>7.3f} "
                f"{stats['context_precision']:>7.3f} "
                f"{stats['context_recall']:>7.3f} "
                f"{stats['ragas_average']:>7.3f} "
                f"{stats['rouge1']:>7.3f} "
                f"{stats['rougeL']:>7.3f} "
                f"{stats['bleu']:>7.3f} "
                f"{stats['avg_latency_ms']:>8}ms"
            )

        print(f"\n  {'─' * 70}")
        print(f"  RAGAS Average by Category\n")
        categories = ["insurance_policy", "financial_audit",
                      "medical_billing", "banking_finance"]
        header = f"  {'Category':<22}"
        for model in summary["models"]:
            header += f" {model:>14}"
        print(header)
        print(f"  {'─' * 70}")

        for cat in categories:
            row = f"  {cat:<22}"
            for model, stats in summary["models"].items():
                score = stats["by_category"].get(cat, {}).get(
                    "ragas_average", "N/A"
                )
                row += f" {score:>14.3f}" if isinstance(score, float) else f" {'N/A':>14}"
            print(row)

        print(f"\n  {'─' * 70}")
        print(f"  RAGAS Average by Difficulty\n")
        header2 = f"  {'Difficulty':<22}"
        for model in summary["models"]:
            header2 += f" {model:>14}"
        print(header2)
        print(f"  {'─' * 70}")

        for diff in ["easy", "medium", "hard"]:
            row = f"  {diff:<22}"
            for model, stats in summary["models"].items():
                score = stats["by_difficulty"].get(diff, {}).get(
                    "ragas_average", "N/A"
                )
                row += f" {score:>14.3f}" if isinstance(score, float) else f" {'N/A':>14}"
            print(row)

def main():
    parser = argparse.ArgumentParser(
        description="MediFinance — Module 4 Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate single model: llama3 | roberta | distilroberta"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run first 5 questions only (smoke test)"
    )
    args = parser.parse_args()

    if not RAG_OK:
        print("[ERROR] Could not import RAG_Engine. "
              "Ensure RAG_Engine.py is in the same directory.")
        return

    models = [args.model] if args.model else MODELS_TO_EVAL
    max_q = 5 if args.quick else None
    engine = RagEngine()
    runner = EvaluationRunner(engine)
    records = runner.run(models=models, max_questions=max_q)


if __name__ == "__main__":
    main()