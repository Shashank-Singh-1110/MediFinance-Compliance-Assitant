# MediFinance Compliance Assistant

> A Retrieval-Augmented Generation system for Indian healthcare and financial regulatory compliance. Ask questions in plain English. Get cited, accurate answers grounded in actual regulatory documents.

---

## What It Does

Indian regulatory compliance spans seven bodies — IRDAI, CGHS, CAG, SEBI, RBI, NHA, ICAI — each publishing hundreds of circulars, rate schedules, and audit reports annually. Compliance professionals waste hours manually searching these documents. A single wrong answer can trigger a regulatory violation.

**MediFinance solves this.** It ingests 40 regulatory documents across four categories, semantically indexes 794 chunks, and answers compliance questions with full source citations — traceable to the exact document and page number.

```
User: "What is the penalty for a CGHS hospital charging above prescribed rates?"

MediFinance: "Hospitals found charging beyond CGHS rates shall be liable for
immediate suspension of empanelment and recovery of the excess amount charged
with 24% interest per annum. (MEDICA_007_Medical_Bill_04, p.1 · COMPLIANCE
& BILLING NOTES · Score: 0.693)"
```

**Zero API keys. Fully local. Fully explainable.**

---

## Key Design Decision

RAG over fine-tuning — because in compliance, every answer must be traceable to its source. Fine-tuning produces fluent answers but you cannot verify which document they came from. RAG forces every claim to be grounded in a retrieved chunk, making the system auditable by design.

---

## Regulatory Coverage

| Category | Documents | Regulators |
|----------|-----------|------------|
| Insurance Policy | 12 | IRDAI |
| Financial Audit | 10 | CAG, SEBI, ICAI |
| Medical Billing | 10 | CGHS, NHA, PM-JAY |
| Banking Finance | 8 | RBI, NBFC |

---

## Architecture

```
User Query
    ↓
QueryPreprocessor     — normalise · expand acronyms · detect category
    ↓
ChromaRetriever       — embed query · cosine similarity · top 6 chunks
    ↓
ContextBuilder        — deduplicate · mode-aware reranking · section boosting
    ↓
Generator             — LLaMA 3 8B / RoBERTa / DistilBERT
    ↓
CitationFormation     — replace [SOURCE N] with document references
    ↓
ConversationMemory    — sliding window of 6 turns per session
    ↓
QueryResult           — JSON → Flask API → Frontend
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3 8B via Ollama |
| Embedding Model | sentence-transformers/all-mpnet-base-v2 (768-dim) |
| Vector Database | ChromaDB (persistent, local) |
| Extractive QA | deepset/roberta-base-squad2 |
| Distilled QA | distilbert/distilbert-base-cased-distilled-squad |
| PDF Extraction | PyMuPDF + pypdf |
| Synthetic PDFs | ReportLab |
| API Layer | Flask + Flask-CORS |
| Frontend | HTML + CSS + JavaScript |
| Evaluation | Custom RAGAS-equivalent + ROUGE + BLEU |

---

## Project Structure

```
MediFinance Compliance Assistant/
│
├── RAG_Engine.py               # Core RAG pipeline — 7 classes
├── Ingestion Pipeline.py       # PDF extraction, chunking, embedding, indexing
├── Evaluation.py               # RAGAS + ROUGE + BLEU evaluation framework
├── app.py                      # Flask REST API server
│
├── index.html                  # Chat interface frontend
├── Dashboard.html              # Evaluation dashboard frontend
│
├── Synthetic Data Generator.py # ReportLab PDF generation for 3 categories
├── Web Scrapper.py             # v3 multi-strategy regulatory document scraper
├── CGHS & SEBI Scrapper.py    # Gap-fill scraper for blocked sources
├── Data Validator.py           # MD5 deduplication, validation, categorisation
│
├── Test File.py                # CLI for --query, --ablation, --interactive modes
├── Inspect chunks.py           # ChromaDB chunk inspector for debugging
├── Corpus.py                   # Corpus coverage diagnostic tool
│
├── data/
│   ├── processed/              # 40 validated, categorised PDF documents
│   ├── vectorstore/chromadb/   # Persistent ChromaDB collection (794 chunks)
│   └── evaluation/             # eval_results.json, eval_report.csv, eval_summary.json
│
└── logs/                       # Pipeline and evaluation logs
```

---

## Quickstart

### Prerequisites

```bash
# Install Ollama and pull LLaMA 3
brew install ollama          # macOS
ollama pull llama3

# Install Python dependencies
pip install flask flask-cors chromadb sentence-transformers transformers \
            torch pymupdf pypdf reportlab requests beautifulsoup4 \
            rouge-score nltk pandas
```

### Step 1 — Ingest Documents

If you are starting fresh and need to build the ChromaDB index:

```bash
python "Ingestion Pipeline.py"
```

This reads from `data/processed/data_inventory.json`, extracts and chunks all PDFs, embeds them with all-mpnet-base-v2, and stores 794 chunks in ChromaDB.

### Step 2 — Start the Backend

Open two terminals:

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start Flask API
python app.py
```

Flask will start on `http://localhost:8000`.

### Step 3 — Open the Frontend

```bash
# Option A — serve with Python (recommended, avoids CORS issues)
cd frontend/
python3 -m http.server 3000
# Open http://localhost:3000

# Option B — open directly in browser
open index.html
```

---

## Usage

### Chat Interface

Navigate to `index.html`. Select a model from the sidebar, optionally filter by document category, and ask any compliance question.

**Example queries:**
- *"What is the waiting period for pre-existing diseases under IRDAI health insurance?"*
- *"What TDS compliance issue was noted in the CAG audit report?"*
- *"What are the KYC and AML requirements for banks under RBI guidelines?"*
- *"What is the CGHS rate for a specialist consultation at a Non-NABH hospital?"*

### Command Line

```bash
# Single query
python "Test File.py" --query "What are the exclusions in a standard IRDAI policy?"

# Ablation comparison across all 3 models
python "Test File.py" --ablation

# Interactive mode
python "Test File.py" --interactive
```

### Evaluation

```bash
# Full evaluation — 25 questions × 3 models (≈90 minutes)
python Evaluation.py

# Quick smoke test — 5 questions only
python Evaluation.py --quick

# Single model
python Evaluation.py --model llama3
```

---

## Evaluation Results

25 questions × 3 models = 75 runs. Custom RAGAS-equivalent metrics using LLaMA 3 as local judge.

| Metric | LLaMA 3 | RoBERTa | DistilBERT |
|--------|---------|---------|------------|
| Faithfulness | **0.858** | 0.680 | 0.616 |
| Answer Relevancy | **0.684** | 0.204 | 0.204 |
| Context Precision | 0.407 | 0.407 | 0.407 |
| Context Recall | 0.460 | 0.460 | 0.460 |
| **RAGAS Average** | **0.622** | 0.428 | 0.412 |
| Avg Latency | 12,208ms | 1,202ms | 901ms |

**Key findings:**
- LLaMA 3 achieves the highest RAGAS score (0.622) with near-zero hallucination
- Context Precision and Recall are identical across all models — retrieval is model-independent
- LLaMA 3 is 13× slower than DistilBERT but 51% better on RAGAS — quality wins for compliance
- DistilBERT's failure on list questions is architectural (CLS token extraction) — documented finding

---

## API Reference

### POST /query

```json
Request:
{
  "query": "What is the penalty for charging above CGHS rates?",
  "mode": "llama3",
  "category_filter": "medical_billing",
  "conversation_id": "session_001"
}

Response:
{
  "query": "What is the penalty for charging above CGHS rates?",
  "answer": "Hospitals found charging beyond CGHS rates...",
  "citations": ["[MEDICA_007] Medical Bill 04 | Page 1 | COMPLIANCE & BILLING NOTES | Score: 0.693"],
  "mode": "llama3",
  "chunks_used": 3,
  "latency_ms": 13331,
  "timestamp": "2026-04-12T00:54:39"
}
```

### GET /health

Returns API status, engine state, and corpus statistics.

### GET /stats

Returns chunk count, embedding model, document count, and regulatory body list.

**Available modes:** `llama3` · `roberta` · `distilroberta`

**Available category filters:** `insurance_policy` · `financial_audit` · `medical_billing` · `banking_finance`

---

## Future Work

- **Cross-encoder re-ranking** for improved Context Precision (~+15%)
- **200-token chunks** for numerical rate lookup questions
- **Cloud deployment** on Vercel with Docker containerisation
- **Expanded corpus** — direct SEBI circulars and MoHFW document integration
- **Mistral as evaluation judge** to eliminate self-evaluation bias

---

## Authors

**Shashank Singh** · Roll No. D015
**Bhavya Sharma** · Roll No. D007

Semester 2 · Generative AI Project
