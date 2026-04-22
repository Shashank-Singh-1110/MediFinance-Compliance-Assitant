import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import fitz
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PYMUPDF_AVAILABLE = True
PYPDF_AVAILABLE   = True
CHROMA_AVAILABLE  = True
ST_AVAILABLE      = True

PROCESSED_DIR   = Path("data/processed")
INVENTORY_FILE  = PROCESSED_DIR / "data_inventory.json"
CHROMA_DIR      = Path("data/vectorstore/chromadb")
REPORT_FILE     = Path("data/vectorstore/ingestion_report.json")
LOG_DIR         = Path("logs")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"   # 768-dim
COLLECTION_NAME = "medifinance_compliance"


CHUNK_MAX_TOKENS    = 600
CHUNK_MIN_TOKENS    = 80
CHUNK_OVERLAP_CHARS = 150
BATCH_EMBED_SIZE    = 32

SECTION_HEADER_PATTERNS = [
    r'^(SECTION\s+[A-Z0-9]+[\s\-—]+.{3,80})$',           # SECTION A — COVERAGE
    r'^(CHAPTER\s+[IVXLC0-9]+[\s\-—]+.{3,80})$',          # CHAPTER IV — CLAIMS
    r'^(\d+\.\s+[A-Z][A-Z\s&,/]{5,60})$',                 # 1. BACKGROUND
    r'^(CLAUSE\s+\d+[\s\-—:]+.{3,60})$',                  # CLAUSE 14 — PENALTIES
    r'^(RULE\s+\d+[\s\-—:]+.{3,60})$',                    # RULE 5 — ELIGIBILITY
    r'^(REGULATION\s+\d+[\s\-—:]+.{3,60})$',              # REGULATION 12 — REPORTING
    r'^(SCHEDULE\s+[A-Z0-9]+[\s\-—:]+.{3,60})$',          # SCHEDULE II — EXCLUSIONS
    r'^(ANNEXURE\s+[A-Z0-9]+[\s\-—:]+.{3,60})$',          # ANNEXURE A — RATE LIST
    r'^(PART\s+[A-Z0-9]+[\s\-—:]+.{3,60})$',              # PART B — AUDIT FINDINGS
    r'^([A-Z][A-Z\s]{8,50}:?\s*)$',                       # ALL CAPS HEADERS (e.g. KEY AUDIT MATTERS)
    r'^(\d+\.\d+\s+[A-Z][a-zA-Z\s]{5,60})$',              # 3.1 Revenue Recognition
    r'^((?:Key|Key Audit|General|Special)\s+\w[\w\s]{5,50})$',  # Key Audit Matters
]

COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_HEADER_PATTERNS]

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Ingestion")

@dataclass
class PageContent:
    page_number: int
    text: str
    char_count: int

@dataclass
class TextChunk:
    chunk_id:      str
    doc_id:        str
    text:          str
    token_estimate: int
    page_number:   int
    section_title: str
    chunk_index:   int
    category:      str
    source_type:   str
    source_name:   str
    filename:      str
    chunk_hash:    str = ""

    def __post_init__(self):
        self.chunk_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]

@dataclass
class IngestionResult:
    doc_id:         str
    filename:       str
    category:       str
    total_pages:    int
    total_chars:    int
    total_chunks:   int
    valid_chunks:   int
    discarded_chunks: int
    chunk_strategy: str
    status:         str
    error:          str = ""

class PDFExtractor:
    def extract(self, pdf_path: Path) -> list[PageContent]:
        if PYMUPDF_AVAILABLE:
            return self._extract_pymupdf(pdf_path)
        elif PYPDF_AVAILABLE:
            logger.warning("  PyMuPDF not available, using pypdf (lower quality)")
            return self._extract_pypdf(pdf_path)
        else:
            raise RuntimeError("Neither PyMuPDF nor pypdf is installed.")

    def _extract_pymupdf(self, pdf_path: Path) -> list[PageContent]:
        pages = []
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            # Extract with layout preservation
            blocks = page.get_text("blocks", sort=True)
            text_parts = []
            for block in blocks:
                if block[6] == 0:  # text block (not image)
                    block_text = block[4].strip()
                    if block_text:
                        text_parts.append(block_text)
            text = "\n".join(text_parts)
            text = self._clean_text(text)
            if text:
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    char_count=len(text)
                ))
        doc.close()
        return pages

    def _extract_pypdf(self, pdf_path: Path) -> list[PageContent]:
        pages = []
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if text:
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    char_count=len(text)
                ))
        return pages

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        lines = text.split('\n')
        lines = [l for l in lines if len(l.strip()) > 2 or l.strip() == '']
        text = '\n'.join(lines)
        text = text.replace('\u2013', '-').replace('\u2014', '—')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        return text.strip()

class SemanticChunker:
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def chunk_document(self, pages: list[PageContent],
                       doc_meta: dict) -> list[TextChunk]:
        if not pages:
            return []

        full_text = ""
        page_map = {}
        offset = 0
        for page in pages:
            page_map[offset] = page.page_number
            full_text += page.text + "\n\n"
            offset += len(page.text) + 2

        sections = self._detect_sections(full_text)

        if len(sections) >= 2:
            chunks = self._semantic_split(sections, full_text, page_map, doc_meta)
            strategy = "semantic"
            if len(chunks) == 0:
                chunks = self._recursive_split(full_text, page_map, doc_meta)
                strategy = "recursive_fallback"
        else:
            chunks = self._recursive_split(full_text, page_map, doc_meta)
            strategy = "recursive"

        for chunk in chunks:
            chunk.chunk_strategy = strategy  # type: ignore

        logger.info(
            f"    Chunking: {len(pages)} pages → {len(chunks)} chunks "
            f"[strategy: {strategy}]"
        )
        return chunks

    def _detect_sections(self, text: str) -> list[tuple[int, str]]:
        found = {}
        for pattern in COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                header = match.group(1).strip()
                pos = match.start()
                if pos not in found:
                    found[pos] = header
        return sorted(found.items(), key=lambda x: x[0])

    def _semantic_split(self, sections: list[tuple[int, str]],
                        full_text: str,
                        page_map: dict,
                        doc_meta: dict) -> list[TextChunk]:
        chunks = []
        chunk_idx = 0

        for i, (start_pos, header) in enumerate(sections):
            if i + 1 < len(sections):
                end_pos = sections[i + 1][0]
            else:
                end_pos = len(full_text)

            section_text = full_text[start_pos:end_pos].strip()

            if not section_text:
                continue

            page_num = self._get_page_for_offset(start_pos, page_map)

            if self.estimate_tokens(section_text) <= CHUNK_MAX_TOKENS:
                chunk = self._make_chunk(
                    text=section_text,
                    section_title=header,
                    page_num=page_num,
                    chunk_idx=chunk_idx,
                    doc_meta=doc_meta
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                sub_chunks = self._split_large_section(
                    section_text, header, page_num, chunk_idx, doc_meta
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        return chunks

    def _split_large_section(self, text: str, section_title: str,
                              page_num: int, start_idx: int,
                              doc_meta: dict) -> list[TextChunk]:
        chunks = []
        paragraphs = re.split(r'\n\n+', text)

        current_text = f"{section_title}\n\n"
        chunk_idx = start_idx

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = current_text + para + "\n\n"
            if self.estimate_tokens(candidate) > CHUNK_MAX_TOKENS and current_text.strip():
                chunk = self._make_chunk(
                    text=current_text.strip(),
                    section_title=section_title,
                    page_num=page_num,
                    chunk_idx=chunk_idx,
                    doc_meta=doc_meta
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_idx += 1
                overlap = self._get_overlap(current_text)
                current_text = f"{section_title} (cont.)\n\n{overlap}{para}\n\n"
            else:
                current_text = candidate
        if current_text.strip():
            chunk = self._make_chunk(
                text=current_text.strip(),
                section_title=section_title,
                page_num=page_num,
                chunk_idx=chunk_idx,
                doc_meta=doc_meta
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, page_map: dict,
                         doc_meta: dict) -> list[TextChunk]:
        chunks = []
        chunk_size_chars = CHUNK_MAX_TOKENS * 4
        step = chunk_size_chars - CHUNK_OVERLAP_CHARS
        chunk_idx = 0

        for start in range(0, len(text), step):
            end = start + chunk_size_chars
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            if end < len(text):
                last_period = chunk_text.rfind('. ')
                if last_period > chunk_size_chars * 0.6:
                    chunk_text = chunk_text[:last_period + 1]

            page_num = self._get_page_for_offset(start, page_map)
            chunk = self._make_chunk(
                text=chunk_text,
                section_title="Document Content",
                page_num=page_num,
                chunk_idx=chunk_idx,
                doc_meta=doc_meta
            )
            if chunk:
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def _make_chunk(self, text: str, section_title: str,
                    page_num: int, chunk_idx: int,
                    doc_meta: dict) -> Optional[TextChunk]:
        token_est = self.estimate_tokens(text)
        if token_est < CHUNK_MIN_TOKENS:
            return None

        chunk_id = f"{doc_meta['doc_id']}_C{chunk_idx:03d}"

        return TextChunk(
            chunk_id=chunk_id,
            doc_id=doc_meta["doc_id"],
            text=text,
            token_estimate=token_est,
            page_number=page_num,
            section_title=section_title,
            chunk_index=chunk_idx,
            category=doc_meta["category"],
            source_type=doc_meta["source_type"],
            source_name=doc_meta["source_name"],
            filename=doc_meta["filename"],
        )

    def _get_page_for_offset(self, offset: int, page_map: dict) -> int:
        page = 1
        for pos, pg in sorted(page_map.items()):
            if pos <= offset:
                page = pg
        return page

    def _get_overlap(self, text: str) -> str:
        tail = text[-CHUNK_OVERLAP_CHARS:].strip()
        first_period = tail.find('. ')
        if first_period != -1 and first_period < len(tail) // 2:
            tail = tail[first_period + 2:]
        return tail + " " if tail else ""

class EmbeddingEngine:
    def __init__(self):
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed.")
        logger.info(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"  Embedding dimension: {self.dim}")

    def embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        texts = [c.text for c in chunks]
        all_embeddings = []

        for i in range(0, len(texts), BATCH_EMBED_SIZE):
            batch = texts[i: i + BATCH_EMBED_SIZE]
            logger.info(
                f"  Embedding batch {i // BATCH_EMBED_SIZE + 1}/"
                f"{(len(texts) - 1) // BATCH_EMBED_SIZE + 1} "
                f"({len(batch)} chunks)"
            )
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,    # cosine similarity optimized
                batch_size=BATCH_EMBED_SIZE
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

class VectorStore:
    def __init__(self):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed.")

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}   # cosine similarity
        )
        logger.info(
            f"  ChromaDB collection '{COLLECTION_NAME}' ready. "
            f"Existing chunks: {self.collection.count()}"
        )

    def upsert_chunks(self, chunks: list[TextChunk],
                      embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0

        ids        = [c.chunk_id for c in chunks]
        documents  = [c.text for c in chunks]
        metadatas  = [
            {
                "doc_id":        c.doc_id,
                "filename":      c.filename,
                "category":      c.category,
                "source_type":   c.source_type,
                "source_name":   c.source_name,
                "page_number":   c.page_number,
                "section_title": c.section_title,
                "chunk_index":   c.chunk_index,
                "token_estimate": c.token_estimate,
                "chunk_hash":    c.chunk_hash,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_text: str, embedding_engine: "EmbeddingEngine",
              n_results: int = 5,
              category_filter: Optional[str] = None) -> dict:
        query_embedding = embedding_engine.model.encode(
            [query_text], normalize_embeddings=True
        ).tolist()

        where = {"category": category_filter} if category_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        return results

class IngestionPipeline:

    def __init__(self):
        self.extractor  = PDFExtractor()
        self.chunker    = SemanticChunker()
        self.embedder   = EmbeddingEngine()
        self.store      = VectorStore()
        self.results    = []

    def ingest_document(self, doc_record: dict) -> IngestionResult:
        doc_id    = doc_record["doc_id"]
        filepath  = Path(doc_record["processed_path"])
        filename  = doc_record["filename"]
        category  = doc_record["category"]

        logger.info(f"\n  {'─'*50}")
        logger.info(f"  Processing: {doc_id} | {filename}")

        try:
            pages = self.extractor.extract(filepath)
            if not pages:
                return IngestionResult(
                    doc_id=doc_id, filename=filename, category=category,
                    total_pages=0, total_chars=0, total_chunks=0,
                    valid_chunks=0, discarded_chunks=0,
                    chunk_strategy="none", status="failed",
                    error="No text extracted from PDF"
                )

            total_chars = sum(p.char_count for p in pages)
            logger.info(f"    Extracted: {len(pages)} pages, {total_chars:,} chars")

            # Stage 2: Chunk
            doc_meta = {
                "doc_id":      doc_id,
                "filename":    filename,
                "category":    category,
                "source_type": doc_record.get("source_type", "unknown"),
                "source_name": doc_record.get("source_name", "unknown"),
            }
            all_chunks = self.chunker.chunk_document(pages, doc_meta)
            valid_chunks = [c for c in all_chunks if c is not None]
            discarded = len(all_chunks) - len(valid_chunks)

            if not valid_chunks:
                return IngestionResult(
                    doc_id=doc_id, filename=filename, category=category,
                    total_pages=len(pages), total_chars=total_chars,
                    total_chunks=len(all_chunks), valid_chunks=0,
                    discarded_chunks=discarded,
                    chunk_strategy="none", status="failed",
                    error="All chunks were below minimum token threshold"
                )

            strategy = getattr(valid_chunks[0], 'chunk_strategy', 'unknown')

            embeddings = self.embedder.embed_chunks(valid_chunks)
            self.store.upsert_chunks(valid_chunks, embeddings)

            logger.info(
                f"    Done: {len(valid_chunks)} chunks stored "
                f"(discarded: {discarded})"
            )

            return IngestionResult(
                doc_id=doc_id, filename=filename, category=category,
                total_pages=len(pages), total_chars=total_chars,
                total_chunks=len(all_chunks), valid_chunks=len(valid_chunks),
                discarded_chunks=discarded, chunk_strategy=strategy,
                status="success"
            )

        except Exception as e:
            logger.error(f"    FAILED: {e}", exc_info=True)
            return IngestionResult(
                doc_id=doc_id, filename=filename, category=category,
                total_pages=0, total_chars=0, total_chunks=0,
                valid_chunks=0, discarded_chunks=0,
                chunk_strategy="none", status="failed", error=str(e)
            )

    def run(self) -> list[IngestionResult]:
        if not INVENTORY_FILE.exists():
            logger.error(f"Inventory file not found: {INVENTORY_FILE}")
            logger.error("Run data_validator.py first.")
            return []

        with open(INVENTORY_FILE) as f:
            inventory = json.load(f)

        all_docs = []
        for cat_data in inventory["categories"].values():
            all_docs.extend(cat_data["documents"])

        logger.info(f"\n{'='*55}")
        logger.info(f"  Module 2 — Document Ingestion Pipeline")
        logger.info(f"  Embedding model : {EMBEDDING_MODEL}")
        logger.info(f"  Documents       : {len(all_docs)}")
        logger.info(f"  Vector store    : {CHROMA_DIR}")
        logger.info(f"{'='*55}")

        start_time = time.time()
        results = []

        for doc in all_docs:
            result = self.ingest_document(doc)
            results.append(result)
            self.results.append(asdict(result))

        elapsed = time.time() - start_time
        self._save_report(results, elapsed)
        return results

    def _save_report(self, results: list[IngestionResult], elapsed: float):
        success  = [r for r in results if r.status == "success"]
        failed   = [r for r in results if r.status == "failed"]

        by_strategy = {}
        by_category = {}
        for r in success:
            by_strategy[r.chunk_strategy] = by_strategy.get(r.chunk_strategy, 0) + r.valid_chunks
            by_category[r.category]       = by_category.get(r.category, 0) + r.valid_chunks

        report = {
            "generated_at":    datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME,
            "chroma_path":     str(CHROMA_DIR),
            "summary": {
                "total_documents":     len(results),
                "successful":          len(success),
                "failed":              len(failed),
                "total_chunks_stored": self.store.count(),
                "total_chars_processed": sum(r.total_chars for r in success),
                "avg_chunks_per_doc":  round(
                    sum(r.valid_chunks for r in success) / max(len(success), 1), 1
                ),
            },
            "chunks_by_strategy": by_strategy,
            "chunks_by_category": by_category,
            "failed_documents": [
                {"doc_id": r.doc_id, "filename": r.filename, "error": r.error}
                for r in failed
            ],
            "document_details": [asdict(r) for r in results],
        }

        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n[REPORT] Saved: {REPORT_FILE}")
        return report



def smoke_test(pipeline: IngestionPipeline):
    test_queries = [
        ("What is the penalty for late claim submission?",       "insurance_policy"),
        ("What are the exclusions in a health insurance policy?","insurance_policy"),
        ("What are Key Audit Matters in financial audit?",       "financial_audit"),
        ("What are CGHS rates for knee replacement surgery?",    "medical_billing"),
        ("What is the GST applicable on hospital services?",     "medical_billing"),
    ]

    print("\n" + "="*55)
    print("  SMOKE TEST — Retrieval Quality Check")
    print("="*55)

    for query, category in test_queries:
        print(f"\n  Q: {query}")
        try:
            results = pipeline.store.query(
                query, pipeline.embedder,
                n_results=2,
                category_filter=category
            )
            docs      = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
                score = round(1 - dist, 3)   # cosine similarity
                print(f"  → [{score:.3f}] {meta.get('filename','?')[:45]} "
                      f"| p.{meta.get('page_number','?')} "
                      f"| {meta.get('section_title','?')[:40]}")
                print(f"     \"{doc[:120].strip()}...\"")
        except Exception as e:
            print(f"  [ERROR] Query failed: {e}")

def main():
    missing = []
    if not PYMUPDF_AVAILABLE and not PYPDF_AVAILABLE:
        missing.append("pymupdf  (pip install pymupdf)")
    if not CHROMA_AVAILABLE:
        missing.append("chromadb  (pip install chromadb)")
    if not ST_AVAILABLE:
        missing.append("sentence-transformers  (pip install sentence-transformers)")

    if missing:
        print("\n[ERROR] Missing dependencies:")
        for m in missing:
            print(f"  pip install {m}")
        print("\nRun: pip install pymupdf chromadb sentence-transformers")
        return

    pipeline = IngestionPipeline()
    results  = pipeline.run()

    if not results:
        return
    success = [r for r in results if r.status == "success"]
    failed  = [r for r in results if r.status == "failed"]
    total_chunks = pipeline.store.count()

    print("\n" + "="*55)
    print("  MODULE 2 — INGESTION COMPLETE")
    print("="*55)
    print(f"  Documents processed : {len(results)}")
    print(f"  Successful          : {len(success)}")
    print(f"  Failed              : {len(failed)}")
    print(f"  Total chunks stored : {total_chunks}")
    print(f"  Vector store path   : {CHROMA_DIR}")
    print(f"  Report              : {REPORT_FILE}")

    if failed:
        print(f"\n  Failed documents:")
        for r in failed:
            print(f"    ✗ {r.doc_id} — {r.error}")
    if success:
        smoke_test(pipeline)

    print("\n  ✅ Vector store ready. Next: Module 3 — RAG Query Engine\n")


if __name__ == "__main__":
    main()