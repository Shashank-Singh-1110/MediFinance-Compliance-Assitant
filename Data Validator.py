import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from pypdf import PdfReader

SYNTHETIC_DIR = Path("data/synthetic")
SCRAPED_DIR = Path("data/scraped")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")
INVENTORY_FILE = PROCESSED_DIR / "data_inventory.json"

MIN_FILE_SIZE_KB = 2  # Reject files smaller than this (synthetic PDFs are ~4-7KB)
MAX_FILE_SIZE_MB = 100  # Warn on files larger than this
MIN_PDF_PAGES = 1  # Reject PDFs with 0 extractable pages

# Maps source folder names → document category
CATEGORY_MAP = {
    # Synthetic
    "insurance_policies": "insurance_policy",
    "financial_audits": "financial_audit",
    "medical_billing": "medical_billing",
    "sebi_circulars": "financial_audit",
    "cghs_documents": "medical_billing",
    # Scraped
    "irdai_gov": "insurance_policy",
    "cag_gov": "financial_audit",
    "nha_gov": "medical_billing",
    "rbi_org": "banking_finance",
    "sebi_gov": "financial_audit",
    "health_gov": "medical_billing",
    "cghs_gov": "medical_billing",
}

# Maps category → processed subfolder
CATEGORY_TO_FOLDER = {
    "insurance_policy": "insurance_policies",
    "financial_audit": "financial_audits",
    "medical_billing": "medical_billing",
    "banking_finance": "banking_finance",
}

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"validator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataValidator")


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────
@dataclass
class DocumentRecord:
    doc_id: str  # Unique ID: CAT_001, CAT_002 ...
    original_path: str
    processed_path: str
    filename: str
    category: str
    source_type: str  # synthetic | scraped
    source_name: str  # IRDAI, CAG, synthetic, etc.
    file_size_kb: float
    page_count: int
    is_valid: bool
    validation_notes: str
    doc_hash: str
    added_at: str


# ─────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────
class PDFValidator:

    def validate(self, filepath: Path) -> tuple[bool, str, int]:
        """
        Returns (is_valid, notes, page_count)
        Checks: existence, size, PDF header, page readability
        """
        # 1. File exists
        if not filepath.exists():
            return False, "File does not exist", 0

        # 2. Size check
        size_kb = filepath.stat().st_size / 1024
        if size_kb < MIN_FILE_SIZE_KB:
            return False, f"File too small ({size_kb:.1f} KB < {MIN_FILE_SIZE_KB} KB)", 0

        if size_kb > MAX_FILE_SIZE_MB * 1024:
            logger.warning(f"  [LARGE] {filepath.name} is {size_kb / 1024:.1f} MB")

        # 3. PDF magic bytes check
        try:
            with open(filepath, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    return False, f"Not a valid PDF (header: {header})", 0
        except Exception as e:
            return False, f"Cannot read file: {e}", 0

        try:
            reader = PdfReader(str(filepath))
            page_count = len(reader.pages)

            if page_count < MIN_PDF_PAGES:
                return False, f"PDF has {page_count} pages", page_count

            # Try extracting text from first page
            text = reader.pages[0].extract_text()
            if not text or len(text.strip()) < 10:
                return True, f"Valid PDF — {page_count} pages (scanned/image-based)", page_count

            return True, f"Valid PDF — {page_count} pages, text extractable", page_count

        except Exception as e:
            return False, f"PDF read error: {e}", 0


# ─────────────────────────────────────────────
# ORGANIZER
# ─────────────────────────────────────────────
class DataOrganizer:

    def __init__(self):
        self.validator = PDFValidator()
        self.seen_hashes = {}  # hash → first filepath (deduplication)
        self.counters = {cat: 0 for cat in CATEGORY_TO_FOLDER}
        self.records = []
        self.stats = {
            "total_found": 0,
            "valid": 0,
            "invalid": 0,
            "duplicates": 0,
            "by_category": {cat: 0 for cat in CATEGORY_TO_FOLDER},
            "by_source": {},
        }

        # Create processed subdirectories
        for folder in CATEGORY_TO_FOLDER.values():
            (PROCESSED_DIR / folder).mkdir(parents=True, exist_ok=True)

    def _get_md5(self, filepath: Path) -> str:
        return hashlib.md5(filepath.read_bytes()).hexdigest()

    def _infer_category_and_source(self, filepath: Path) -> tuple[str, str, str]:
        """Returns (category, source_type, source_name)"""
        parts = filepath.parts

        # Determine source type
        source_type = "scraped"
        for part in parts:
            if "synthetic" in part.lower():
                source_type = "synthetic"
                break

        # Determine source folder name
        source_folder = ""
        for part in parts:
            if part in CATEGORY_MAP:
                source_folder = part
                break

        category = CATEGORY_MAP.get(source_folder, "financial_audit")

        # Source name from parent directory
        source_name = filepath.parent.name if source_type == "scraped" else "synthetic"
        return category, source_type, source_name

    def _make_clean_filename(self, category: str, original_name: str, doc_id: str) -> str:
        """Generate a clean, consistent filename."""
        # Strip known prefixes that came from scraper
        name = original_name
        for prefix in ["CAG_", "IRDAI_circular_", "IRDAI_regulation_",
                       "RBI_master_circular_", "RBI_master_direction_",
                       "RBI_annual_report_", "NHA_", "SEBI_", "CGHS_", "MoHFW_", "Health_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Clean up
        name = name.replace(".pdf", "").strip("_").strip()
        if len(name) > 60:
            name = name[:60].strip("_")
        if not name:
            name = "document"

        folder_prefix = CATEGORY_TO_FOLDER[category].upper()[:3]
        return f"{doc_id}_{name}.pdf"

    def process_file(self, filepath: Path) -> DocumentRecord | None:
        self.stats["total_found"] += 1

        # Infer metadata
        category, source_type, source_name = self._infer_category_and_source(filepath)

        # Validate
        is_valid, notes, pages = self.validator.validate(filepath)
        if not is_valid:
            logger.warning(f"  [INVALID] {filepath.name} — {notes}")
            self.stats["invalid"] += 1
            return None

        # Deduplicate
        doc_hash = self._get_md5(filepath)
        if doc_hash in self.seen_hashes:
            logger.info(f"  [DUP] {filepath.name} == {Path(self.seen_hashes[doc_hash]).name}")
            self.stats["duplicates"] += 1
            return None
        self.seen_hashes[doc_hash] = str(filepath)

        # Assign ID and clean filename
        self.counters[category] += 1
        cat_code = category.replace("_", "").upper()[:6]
        doc_id = f"{cat_code}_{self.counters[category]:03d}"
        clean_name = self._make_clean_filename(category, filepath.stem, doc_id)

        # Copy to processed/
        dest_folder = PROCESSED_DIR / CATEGORY_TO_FOLDER[category]
        dest_path = dest_folder / clean_name
        shutil.copy2(filepath, dest_path)

        # Update stats
        self.stats["valid"] += 1
        self.stats["by_category"][category] += 1
        self.stats["by_source"][source_name] = self.stats["by_source"].get(source_name, 0) + 1

        size_kb = filepath.stat().st_size / 1024
        record = DocumentRecord(
            doc_id=doc_id,
            original_path=str(filepath),
            processed_path=str(dest_path),
            filename=clean_name,
            category=category,
            source_type=source_type,
            source_name=source_name,
            file_size_kb=round(size_kb, 2),
            page_count=pages,
            is_valid=True,
            validation_notes=notes,
            doc_hash=doc_hash,
            added_at=datetime.now().isoformat()
        )

        logger.info(f"  [OK] {doc_id} | {clean_name[:55]} | {pages}pp | {size_kb:.0f}KB")
        return record

    def run(self) -> list[DocumentRecord]:
        logger.info(f"\n{'=' * 55}")
        logger.info("  Data Validator & Organizer — Starting")
        logger.info(f"{'=' * 55}")

        # Collect all PDF files from both sources
        all_pdfs = []

        # Synthetic docs
        if SYNTHETIC_DIR.exists():
            synthetic_pdfs = list(SYNTHETIC_DIR.rglob("*.pdf"))
            logger.info(f"\n  Found {len(synthetic_pdfs)} synthetic PDFs")
            all_pdfs.extend(synthetic_pdfs)
        else:
            logger.warning(f"  Synthetic dir not found: {SYNTHETIC_DIR}")

        # Scraped docs
        if SCRAPED_DIR.exists():
            scraped_pdfs = [
                p for p in SCRAPED_DIR.rglob("*.pdf")
                if "manifest" not in p.name.lower()
            ]
            logger.info(f"  Found {len(scraped_pdfs)} scraped PDFs")
            all_pdfs.extend(scraped_pdfs)
        else:
            logger.warning(f"  Scraped dir not found: {SCRAPED_DIR}")

        logger.info(f"\n  Total PDFs to process: {len(all_pdfs)}")
        logger.info(f"{'─' * 55}")

        # Process each file
        for pdf_path in sorted(all_pdfs):
            record = self.process_file(pdf_path)
            if record:
                self.records.append(record)

        return self.records


# ─────────────────────────────────────────────
# INVENTORY WRITER
# ─────────────────────────────────────────────
def save_inventory(records: list[DocumentRecord], stats: dict):
    """Save data_inventory.json — input spec for the RAG ingestion pipeline."""

    # Group records by category
    by_category = {}
    for rec in records:
        by_category.setdefault(rec.category, []).append(asdict(rec))

    inventory = {
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "1.0",
        "total_documents": len(records),
        "stats": stats,
        "categories": {
            cat: {
                "count": len(docs),
                "folder": str(PROCESSED_DIR / CATEGORY_TO_FOLDER.get(cat, cat)),
                "documents": docs
            }
            for cat, docs in by_category.items()
        },
        "ingestion_ready": True,
        "notes": (
            "All documents have been validated (PDF integrity, size, readability), "
            "deduplicated (MD5 hash), and organized into category folders. "
            "Feed processed_path from each document record into the chunking pipeline."
        )
    }

    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    logger.info(f"\n[INVENTORY] Saved: {INVENTORY_FILE}")
    return inventory


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    organizer = DataOrganizer()
    records = organizer.run()
    stats = organizer.stats
    inventory = save_inventory(records, stats)

    # Final report
    print("\n" + "=" * 55)
    print("  DATA VALIDATION COMPLETE")
    print("=" * 55)
    print(f"  Total PDFs Found    : {stats['total_found']}")
    print(f"  Valid & Processed   : {stats['valid']}")
    print(f"  Invalid (rejected)  : {stats['invalid']}")
    print(f"  Duplicates (skipped): {stats['duplicates']}")
    print(f"\n  By Category:")
    for cat, count in stats["by_category"].items():
        folder = CATEGORY_TO_FOLDER.get(cat, cat)
        bar = "█" * count
        print(f"    {cat:<22} {count:>3}  {bar}")
    print(f"\n  By Source:")
    for src, count in sorted(stats["by_source"].items(),
                             key=lambda x: x[1], reverse=True):
        print(f"    {src:<25} {count:>3} docs")
    print(f"\n  Output : {PROCESSED_DIR}/")
    print(f"  Inventory : {INVENTORY_FILE}")
    print("=" * 55)
    print("\n  ✅ Data layer complete. Ready for RAG ingestion pipeline.\n")


if __name__ == "__main__":
    main()