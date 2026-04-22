import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

CHROMA_DIR = Path("data/vectorstore/chromadb")
COLLECTION_NAME = "medifinance_compliance"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120

TOP_K_RETRIEVAL = 6
TOP_K_CONTEXT = 4
MAX_CONTEXT_CHARS = 3500
CONVERSATION_WINDOW = 6

ROBERTA_MODEL = "deepset/roberta-base-squad2"
DISTILROBERTA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    from transformers import pipeline as hf_pipeline
    HF_OK = True
except ImportError:
    HF_OK = False

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(f"logs/rag_engine_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RAGEngine")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    filename: str
    category: str
    page_number: int
    section_title: str
    source_name: str


@dataclass
class Citation:
    doc_id: str
    filename: str
    page_number: int
    section_title: str
    source_name: str
    relevance_score: float

    def __str__(self):
        name = self.filename.replace("_", " ").replace(".pdf", "")
        return (
            f"[{self.doc_id}] {name} | "
            f"Page {self.page_number} | "
            f"Section: {self.section_title} | "
            f"Score: {self.relevance_score:.3f}"
        )


@dataclass
class QueryResult:
    query: str
    answer: str
    citations: list[Citation]
    mode: str
    chunks_used: int
    latency_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    conversation_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [str(c) for c in self.citations]
        return d

    def pretty(self) -> str:
        lines = [
            f"\n{'─' * 60}",
            f"  Query   : {self.query}",
            f"  Mode    : {self.mode.upper()}",
            f"  Latency : {self.latency_ms}ms",
            f"{'─' * 60}",
            f"\n  Answer:\n",
        ]
        for para in self.answer.split("\n"):
            if para.strip():
                lines.append(f"  {para}")
            else:
                lines.append("")
        if self.citations:
            lines.append(f"\n{'─' * 60}")
            lines.append(f"  Sources ({len(self.citations)} chunks used):\n")
            for i, cit in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {cit}")
        lines.append(f"{'─' * 60}\n")
        return "\n".join(lines)


class QueryPreprocessor:
    CATEGORY_SIGNALS = {
        "insurance_policy": [
            "insurance", "policy", "premium", "claim", "cover", "irdai",
            "hospitalisation", "exclusion", "pre-existing", "sum insured",
            "cashless", "tpa", "mediclaim", "health plan", "maternity",
        ],
        "financial_audit": [
            "audit", "sebi", "compliance", "financial", "revenue", "gst",
            "tds", "balance sheet", "profit", "loss", "icai", "ind as",
            "related party", "key audit", "chartered accountant", "cag",
        ],
        "medical_billing": [
            "billing", "cghs", "pm-jay", "ayushman", "procedure", "icd",
            "rate", "package", "hospital charge", "surgery", "nabh",
            "nha", "reimbursement", "empanelment", "day care",
        ],
        "banking_finance": [
            "rbi", "bank", "nbfc", "lending", "credit", "priority sector",
            "master circular", "monetary", "repo", "liquidity",
        ],
    }

    EXPANSIONS = {
        r"\bIRDAI\b": "Insurance Regulatory and Development Authority of India IRDAI",
        r"\bCGHS\b": "Central Government Health Scheme CGHS",
        r"\bPM-JAY\b": "Pradhan Mantri Jan Arogya Yojana PM-JAY Ayushman Bharat",
        r"\bNHA\b": "National Health Authority NHA",
        r"\bCAG\b": "Comptroller and Auditor General CAG",
        r"\bSEBI\b": "Securities and Exchange Board of India SEBI",
        r"\bRBI\b": "Reserve Bank of India RBI",
        r"\bNABH\b": "National Accreditation Board for Hospitals NABH",
        r"\bICD\b": "International Classification of Diseases ICD",
        r"\bTPA\b": "Third Party Administrator TPA",
        r"\bDPDP\b": "Digital Personal Data Protection DPDP",
    }

    def process(self, query: str) -> tuple[str, Optional[str]]:
        q = query.strip()
        q = re.sub(r"\s+", " ", q)
        q = q.rstrip("?.,!") + "?"
        q_lower = q.lower()
        best_cat = None
        best_score = 0
        for cat, signals in self.CATEGORY_SIGNALS.items():
            score = sum(1 for s in signals if s in q_lower)
            if score > best_score:
                best_score = score
                best_cat = cat
        detected_category = best_cat if best_score >= 2 else None
        q_expanded = q
        for pattern, expansion in self.EXPANSIONS.items():
            q_expanded = re.sub(pattern, expansion, q_expanded)
        return q_expanded, detected_category


class ChromaRetriever:
    def __init__(self):
        if not CHROMA_OK:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")
        if not ST_OK:
            raise RuntimeError("sentence-transformers not installed.")
        logger.info(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"  Connecting to ChromaDB: {CHROMA_DIR}")
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        logger.info(f"  Collection ready — {self.collection.count()} chunks indexed")

    def retrieve(
            self,
            query: str,
            n_results: int = TOP_K_RETRIEVAL,
            category_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        where = {"category": category_filter} if category_filter else None
        try:
            results = self.collection.query(
                query_embeddings=query_vec,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            results = self.collection.query(
                query_embeddings=query_vec,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        chunks = []
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metadatas, distances):
            score = round(1.0 - dist, 4)
            chunks.append(RetrievedChunk(
                chunk_id=meta.get("chunk_hash", ""),
                text=doc,
                score=score,
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                category=meta.get("category", ""),
                page_number=int(meta.get("page_number", 0)),
                section_title=meta.get("section_title", ""),
                source_name=meta.get("source_name", ""),
            ))
        if category_filter and len(chunks) < 2:
            return self.retrieve(query, n_results, category_filter=None)
        return chunks


class ContextBuilder:
    BOOST_TERMS = [
        "exclusion", "inclusion", "coverage", "benefit", "claim",
        "penalty", "rate", "charge", "procedure", "billing",
        "audit", "compliance", "tds", "gst", "kyc", "aml",
        "circular", "direction", "lending", "deposit",
        "general exclusion", "section b", "key audit",
    ]
    PENALISE_TERMS = [
        "regulatory compliance", "grievance", "signature",
        "terms and conditions", "disclaimer", "footer", "document content",
    ]

    def _score_chunk(self, c: RetrievedChunk) -> float:
        score = c.score
        title_lower = c.section_title.lower()
        if any(t in title_lower for t in self.BOOST_TERMS):
            score += 0.05
        if any(t in title_lower for t in self.PENALISE_TERMS):
            score -= 0.05
        return score

    def build(
            self,
            chunks: list[RetrievedChunk],
            top_k: int = TOP_K_CONTEXT,
            mode: str = "llama3",
    ) -> tuple[str, list[RetrievedChunk]]:
        if not chunks:
            return "", []

        seen_hashes = set()
        unique = []
        for c in chunks:
            h = hashlib.md5(c.text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(c)

        if mode in ("roberta", "distilroberta"):
            selected = sorted(unique, key=self._score_chunk, reverse=True)[:top_k]
        else:
            doc_counts: dict[str, int] = {}
            reranked = []
            for c in unique:
                repeat_penalty = doc_counts.get(c.doc_id, 0) * 0.08
                adjusted_score = c.score - repeat_penalty
                doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1
                reranked.append((adjusted_score, c))
            reranked.sort(key=lambda x: x[0], reverse=True)
            selected = [c for _, c in reranked[:top_k]]

        parts = []
        total_chars = 0
        final_selected = []

        for i, chunk in enumerate(selected, 1):
            header = (
                f"[SOURCE {i}] {chunk.filename} | "
                f"Page {chunk.page_number} | "
                f"Section: {chunk.section_title}\n"
            )
            block = header + chunk.text + "\n"
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                available = MAX_CONTEXT_CHARS - total_chars - len(header) - 20
                if available > 200:
                    trimmed = chunk.text[:available] + "...[truncated]"
                    block = header + trimmed + "\n"
                    parts.append(block)
                    final_selected.append(chunk)
                break
            parts.append(block)
            final_selected.append(chunk)
            total_chars += len(block)

        context = "\n".join(parts)
        return context, final_selected


class LLaMA3Generator:
    SYSTEM_PROMPT = """You are MediFinance Assistant, a precise and helpful expert on Indian healthcare and financial compliance regulations.

Your knowledge base covers: IRDAI health insurance regulations, CGHS rate lists, PM-JAY/Ayushman Bharat guidelines, CAG audit reports, SEBI circulars, RBI master directions, and ICAI auditing standards.

RULES:
1. Answer using the provided [SOURCE] excerpts as your primary basis.
2. Cite every factual claim using the [SOURCE N] label it came from.
3. If the sources contain PARTIAL information — use what is available, provide a clear answer from that, and briefly note what aspect is not covered.
4. Only say "insufficient information" when the sources contain absolutely NOTHING relevant to the question.
5. Never refuse to engage — even partial answers are valuable in compliance contexts.
6. Use Indian regulatory terminology — IRDAI, CGHS, PM-JAY, IndAS, RBI, SEBI — not US equivalents.
7. Format monetary values in Indian Rupees (Rs. / ₹), dates in DD/MM/YYYY.
8. For compliance questions, structure your answer with: the rule/clause, the amount or timeline if applicable, and the consequence of non-compliance if mentioned in sources.
9. Keep answers concise and factual — 3 to 5 sentences for simple questions, up to 8 for complex ones."""

    def __init__(self):
        self._check_ollama()

    def _check_ollama(self):
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                logger.info(f"  Ollama: {OLLAMA_MODEL} ready ✓")
            else:
                logger.warning(f"  Ollama running but '{OLLAMA_MODEL}' not found. Run: ollama pull llama3")
        except requests.exceptions.ConnectionError:
            logger.warning("  Ollama not reachable at localhost:11434. Start with: ollama serve")

    def generate(self, query: str, context: str, conversation_history: list[dict] = None) -> str:
        history_str = ""
        if conversation_history:
            for turn in conversation_history[-4:]:
                history_str += f"User: {turn['query']}\nAssistant: {turn['answer'][:300]}...\n\n"

        prompt = f"""{self.SYSTEM_PROMPT}

{'--- CONVERSATION HISTORY ---' if history_str else ''}
{history_str}
--- RETRIEVED DOCUMENT EXCERPTS ---

{context}

--- QUESTION ---
{query}

--- YOUR ANSWER (cite [SOURCE N] for every claim) ---"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 800,
                "stop": ["--- QUESTION", "User:"],
            },
        }
        try:
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "Generation timed out. Please retry in a moment."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Ensure it is running: ollama serve"
        except Exception as e:
            logger.error(f"  LLaMA3 error: {e}")
            return f"Generation error: {e}"


class RoBERTaExtractor:
    def __init__(self, model_name: str = ROBERTA_MODEL):
        if not HF_OK:
            raise RuntimeError("transformers not installed. Run: pip install transformers torch")
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            logger.info(f"  Loading extractive QA model: {self.model_name}")
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self._torch = torch
            self._pipeline = True
            logger.info("  Extractive QA model loaded ✓")

    def _run_qa(self, question: str, context: str) -> dict:
        import torch
        inputs = self._tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        start_score = torch.softmax(outputs.start_logits, dim=1)[0][start].item()
        end_score = torch.softmax(outputs.end_logits, dim=1)[0][end - 1].item()
        score = (start_score + end_score) / 2
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        answer = self._tokenizer.convert_tokens_to_string(tokens).strip()
        answer = re.sub(r'^[\s\[<\]>]+', '', answer)
        answer = re.sub(r'\b(CLS|SEP|PAD|UNK|s)\b', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        if answer in ('', '[CLS]', '[SEP]', '[PAD]', '[UNK]', 'CLS', 'SEP'):
            answer = ''
        return {"answer": answer, "score": score, "start": int(start)}

    def extract(self, query: str, chunks: list[RetrievedChunk]) -> str:
        self._load()
        context_text = "\n\n".join(
            f"[{c.filename} | p.{c.page_number}]\n{c.text}"
            for c in chunks[:TOP_K_CONTEXT]
        )
        if len(context_text) > 1800:
            context_text = context_text[:1800]
        try:
            result = self._run_qa(query, context_text)
            answer = result.get("answer", "")
            score = result.get("score", 0.0)
            start = result.get("start", 0)
            if not answer or len(answer) < 3 or score < 0.05:
                return "Extractive model could not find a confident answer in the retrieved passages (score below threshold)."
            source_ref = "unknown source"
            cumulative = 0
            for c in chunks[:TOP_K_CONTEXT]:
                chunk_ctx = f"[{c.filename} | p.{c.page_number}]\n{c.text}\n\n"
                if cumulative + len(chunk_ctx) >= start:
                    source_ref = f"{c.filename} | Page {c.page_number} | {c.section_title}"
                    break
                cumulative += len(chunk_ctx)
            return f"{answer}\n\n[Extracted from: {source_ref} | confidence: {score:.3f}]"
        except Exception as e:
            logger.error(f"  Extractor error: {e}")
            return f"Extraction error: {e}"


class DistilRoBERTaExtractor(RoBERTaExtractor):
    def __init__(self):
        super().__init__(model_name=DISTILROBERTA_MODEL)


class CitationFormation:
    def format(self, answer: str, chunks: list[RetrievedChunk], mode: str = "llama3") -> tuple[str, list[Citation]]:
        citations = [
            Citation(
                doc_id=c.doc_id,
                filename=c.filename.replace(".pdf", ""),
                page_number=c.page_number,
                section_title=c.section_title,
                source_name=c.source_name,
                relevance_score=c.score,
            )
            for c in chunks
        ]
        if mode == "llama3":
            formatted = answer
            for i, cit in enumerate(citations, 1):
                short_ref = f"({cit.filename[:30]}, p.{cit.page_number})"
                formatted = formatted.replace(f"[SOURCE {i}]", short_ref)
            return formatted, citations
        return answer, citations


class ConversationMemory:
    def __init__(self, window_size: int = CONVERSATION_WINDOW):
        self.window_size = window_size
        self.sessions: dict[str, list[dict]] = {}

    def add_turn(self, conversation_id: str, query: str, answer: str, citations: list[Citation]):
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = []
        self.sessions[conversation_id].append({
            "query": query,
            "answer": answer,
            "citations": [str(c) for c in citations],
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.sessions[conversation_id]) > self.window_size:
            self.sessions[conversation_id] = self.sessions[conversation_id][-self.window_size:]

    def get_history(self, conversation_id: str) -> list[dict]:
        return self.sessions.get(conversation_id, [])

    def clear(self, conversation_id: str):
        self.sessions.pop(conversation_id, None)

    def list_sessions(self) -> list[str]:
        return list(self.sessions.keys())


class RagEngine:
    def __init__(self):
        logger.info("\n" + "=" * 55)
        logger.info("  MediFinance RAG Engine — Initialising")
        logger.info("=" * 55)
        self.preprocessor = QueryPreprocessor()
        self.retriever = ChromaRetriever()
        self.context_builder = ContextBuilder()
        self.llama3 = LLaMA3Generator()
        self.roberta = None
        self.distilroberta = None
        self.citation_fmt = CitationFormation()
        self.memory = ConversationMemory()
        logger.info("  RAG Engine ready ✓\n")

    def query(
            self,
            query_text: str,
            mode: str = "llama3",
            conversation_id: str = "default",
            category_filter: Optional[str] = None,
    ) -> QueryResult:
        t_start = time.time()
        logger.info(f"\n  Query [{mode}]: {query_text[:80]}")
        processed_query, detected_cat = self.preprocessor.process(query_text)
        effective_cat = category_filter or detected_cat
        logger.info(f"  Category filter: {effective_cat or 'none (full search)'}")
        chunks = self.retriever.retrieve(
            processed_query,
            n_results=TOP_K_RETRIEVAL,
            category_filter=effective_cat,
        )
        logger.info(
            f"  Retrieved {len(chunks)} chunks | top score: {chunks[0].score:.3f}"
            if chunks else "  No chunks found"
        )
        if not chunks:
            return QueryResult(
                query=query_text,
                answer="No relevant documents found in the knowledge base for this query.",
                citations=[],
                mode=mode,
                chunks_used=0,
                latency_ms=int((time.time() - t_start) * 1000),
                conversation_id=conversation_id,
            )

        context, selected_chunks = self.context_builder.build(chunks, mode=mode)
        history = self.memory.get_history(conversation_id)

        if mode == "llama3":
            raw_answer = self.llama3.generate(query_text, context, history)
        elif mode == "roberta":
            if self.roberta is None:
                self.roberta = RoBERTaExtractor()
            raw_answer = self.roberta.extract(query_text, selected_chunks)
        elif mode == "distilroberta":
            if self.distilroberta is None:
                self.distilroberta = DistilRoBERTaExtractor()
            raw_answer = self.distilroberta.extract(query_text, selected_chunks)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose: llama3 | roberta | distilroberta")

        formatted_answer, citations = self.citation_fmt.format(raw_answer, selected_chunks, mode)
        self.memory.add_turn(conversation_id, query_text, formatted_answer, citations)
        latency = int((time.time() - t_start) * 1000)
        logger.info(f"  Done — {latency}ms | {len(citations)} citations")

        return QueryResult(
            query=query_text,
            answer=formatted_answer,
            citations=citations,
            mode=mode,
            chunks_used=len(selected_chunks),
            latency_ms=latency,
            conversation_id=conversation_id,
        )

    def ablation_compare(self, query_text: str) -> dict:
        logger.info(f"\n  Ablation compare: {query_text[:60]}")
        results = {}
        for mode in ["llama3", "roberta", "distilroberta"]:
            try:
                r = self.query(query_text, mode=mode, conversation_id=f"ablation_{mode}")
                results[mode] = {
                    "answer": r.answer,
                    "latency_ms": r.latency_ms,
                    "chunks_used": r.chunks_used,
                    "citations": [str(c) for c in r.citations],
                }
            except Exception as e:
                results[mode] = {"error": str(e), "latency_ms": 0}
        return results

    def clear_memory(self, conversation_id: str = "default"):
        self.memory.clear(conversation_id)
        logger.info(f"  Memory cleared: {conversation_id}")