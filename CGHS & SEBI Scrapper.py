"""
MediFinance — Gap-Fill Synthetic Generator
==========================================
Generates synthetic SEBI and CGHS documents to replace
blocked scraper sources. Pure Python + ReportLab, no API keys.

Documents:
  3 x SEBI-style regulatory circulars / audit reports
  3 x CGHS-style rate lists and policy circulars

Author: MediFinance Team
"""

import os
import random
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUTPUT_DIR = "data/synthetic"

# ── Data Pools ────────────────────────────────
INDIAN_CITIES   = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune"]
CA_FIRMS        = [
    "Deloitte Haskins & Sells LLP", "Price Waterhouse & Co. LLP",
    "B S R & Co. LLP", "S.R. Batliboi & Associates LLP",
    "Walker Chandiok & Co. LLP"
]
LISTED_COMPANIES = [
    "Fortis Healthcare Ltd.", "Apollo Hospitals Enterprise Ltd.",
    "Max Healthcare Institute Ltd.", "Narayana Hrudayalaya Ltd.",
    "Aster DM Healthcare Ltd.", "Dr. Lal PathLabs Ltd.",
    "Metropolis Healthcare Ltd.", "HCG Oncology Ltd."
]
CGHS_CITIES = [
    "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad",
    "Bengaluru", "Pune", "Ahmedabad", "Lucknow", "Bhopal"
]

def random_date(start_year=2022, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%d/%m/%Y")

def get_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=styles["Title"],
            fontSize=15, textColor=colors.HexColor("#1a3c5e"),
            spaceAfter=4, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#2c6496"),
            spaceAfter=4, alignment=TA_CENTER),
        "section": ParagraphStyle("section", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#1a3c5e"),
            spaceBefore=12, spaceAfter=5,
            backColor=colors.HexColor("#eaf2fb")),
        "body": ParagraphStyle("body", parent=styles["Normal"],
            fontSize=9.5, leading=14, spaceAfter=5,
            alignment=TA_JUSTIFY),
        "clause": ParagraphStyle("clause", parent=styles["Normal"],
            fontSize=9, leading=13, leftIndent=18,
            spaceAfter=4, alignment=TA_JUSTIFY),
        "footer": ParagraphStyle("footer", parent=styles["Normal"],
            fontSize=7.5, textColor=colors.grey, alignment=TA_CENTER),
    }

def std_table_style(header_bg="#1a3c5e"):
    return TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ])

# ─────────────────────────────────────────────
# GENERATOR A: SEBI CIRCULAR
# ─────────────────────────────────────────────
def generate_sebi_circular(output_path, idx):
    s = get_styles()
    city = "Mumbai"
    date = random_date(2022, 2024)
    circ_no = f"SEBI/HO/CFD/CMD{random.randint(1,2)}/CIR/P/{date.split('/')[-1]}/{random.randint(1,200)}"

    topics = [
        {
            "subject": "Disclosure Requirements for Listed Healthcare Entities",
            "body": (
                "In exercise of powers conferred under Section 11(1) of the Securities and "
                "Exchange Board of India Act, 1992 read with Regulation 101 of SEBI (Listing "
                "Obligations and Disclosure Requirements) Regulations, 2015, SEBI hereby issues "
                "the following circular regarding enhanced disclosure norms for listed entities "
                "operating in the healthcare and pharmaceutical sector."
            ),
            "clauses": [
                ("1. Applicability", "This circular shall apply to all entities listed on recognised stock exchanges whose primary business is healthcare services, hospitals, diagnostics, or pharmaceutical manufacturing."),
                ("2. Enhanced Disclosures", "Listed healthcare entities shall disclose the following on a quarterly basis: (a) Bed occupancy rates segregated by category; (b) Revenue from government scheme patients (PM-JAY, CGHS, ECHS) as a percentage of total revenue; (c) Outstanding receivables from government schemes with ageing analysis."),
                ("3. Related Party Transactions", "Any transaction with a related party involving healthcare service delivery, medical equipment procurement, or pharmaceutical supply exceeding Rs. 1 crore shall require prior approval of the Audit Committee and disclosure within 24 hours on stock exchange."),
                ("4. Compliance Timeline", "Listed entities shall ensure compliance with this circular within 90 days from the date of issuance. Non-compliance shall attract action under Section 23E of the Securities Contracts (Regulation) Act, 1956."),
            ]
        },
        {
            "subject": "Guidelines on Corporate Governance for Healthcare Listed Companies",
            "body": (
                "SEBI, in consultation with the Ministry of Health and Family Welfare and based on "
                "the recommendations of the Corporate Governance Committee, issues guidelines to "
                "strengthen corporate governance standards specifically applicable to listed entities "
                "in the healthcare sector, given the critical nature of services provided and the "
                "involvement of public funds through government health schemes."
            ),
            "clauses": [
                ("1. Board Composition", "Listed healthcare entities with market capitalisation exceeding Rs. 500 crore shall have at least one independent director with a medical qualification (MBBS or equivalent) or a background in public health policy on their Board of Directors."),
                ("2. Audit Committee", "The Audit Committee of listed healthcare entities shall specifically review: (a) Revenue recognition policies for government scheme patients; (b) Upcoding and billing fraud risk assessment; (c) Compliance with IRDAI regulations for insurance claims."),
                ("3. Whistleblower Mechanism", "A dedicated whistleblower mechanism for reporting billing fraud, upcoding, or non-compliance with PM-JAY package rates shall be established and reported in the Annual Report under Corporate Governance section."),
                ("4. Penalty", "Non-compliance with these guidelines shall be subject to action under Section 15HB of SEBI Act, 1992 which provides for penalty up to Rs. 25 crore or three times the profits made, whichever is higher."),
            ]
        },
        {
            "subject": "Insider Trading Restrictions — Healthcare Sector Specific Provisions",
            "body": (
                "In continuation of SEBI (Prohibition of Insider Trading) Regulations, 2015 and "
                "the amendments thereto, SEBI issues this circular providing sector-specific "
                "guidance on unpublished price sensitive information (UPSI) for listed entities "
                "in the healthcare sector, including hospitals, diagnostics chains, pharmaceutical "
                "companies, and health insurance entities."
            ),
            "clauses": [
                ("1. UPSI for Healthcare", "For listed healthcare entities, Unpublished Price Sensitive Information shall specifically include: (a) Results of clinical trials in Phase II or III; (b) Government scheme empanelment or de-empanelment decisions; (c) IRDAI licence approval, suspension or cancellation; (d) Outcome of SEBI/IRDAI investigations."),
                ("2. Trading Window", "The trading window shall be closed for an additional period of 15 days before the announcement of any government scheme rate revision (PM-JAY or CGHS) where the company's revenue from such schemes exceeds 20% of total revenue."),
                ("3. Designated Persons", "All medical directors, chief medical officers, and persons heading government affairs or insurance departments shall be categorised as Designated Persons for the purpose of these regulations."),
                ("4. Compliance Officer", "The Compliance Officer shall file a half-yearly report to SEBI in the prescribed format confirming adherence to these sector-specific insider trading provisions."),
            ]
        }
    ]

    topic = topics[(idx - 1) % len(topics)]
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    story.append(Paragraph("SECURITIES AND EXCHANGE BOARD OF INDIA", s["title"]))
    story.append(Paragraph("SEBI Bhavan, Plot No. C4-A, 'G' Block, Bandra Kurla Complex, Mumbai – 400 051", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 8))

    meta = [
        ["Circular No.", circ_no, "Date", date],
        ["To", "All Listed Entities (Healthcare Sector) | All Stock Exchanges | All Depositories", "", ""],
        ["Subject", topic["subject"], "", ""],
    ]
    meta_tbl = Table(meta, colWidths=[3*cm, 9*cm, 2*cm, 4*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("SPAN", (1,1), (-1,1)),
        ("SPAN", (1,2), (-1,2)),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. BACKGROUND", s["section"]))
    story.append(Paragraph(topic["body"], s["body"]))

    story.append(Paragraph("2. PROVISIONS OF THIS CIRCULAR", s["section"]))
    for clause_title, clause_text in topic["clauses"]:
        story.append(Paragraph(f"<b>{clause_title}:</b>", s["body"]))
        story.append(Paragraph(clause_text, s["clause"]))

    story.append(Paragraph("3. APPLICABILITY & EFFECTIVE DATE", s["section"]))
    story.append(Paragraph(
        f"This circular is issued in exercise of powers conferred under Section 11(1) of the "
        f"SEBI Act, 1992 and shall come into force with immediate effect from {date}. "
        f"Stock Exchanges are directed to bring this circular to the notice of all listed companies "
        f"and also disseminate the same on their websites.",
        s["body"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(f"For and on behalf of the Board,", s["body"]))
    story.append(Spacer(1, 25))
    story.append(Paragraph(f"General Manager | Division of Corporate Finance | SEBI, {city}", s["body"]))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"SEBI Registration No.: Not Applicable — Regulatory Authority | "
        f"This circular is available on SEBI website: www.sebi.gov.in",
        s["footer"]))

    doc.build(story)
    print(f"  [OK] SEBI Circular {idx}: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────
# GENERATOR B: CGHS RATE LIST & POLICY CIRCULAR
# ─────────────────────────────────────────────
def generate_cghs_document(output_path, idx):
    s = get_styles()
    date = random_date(2022, 2024)
    city = random.choice(CGHS_CITIES)

    if idx % 2 == 1:
        _generate_cghs_rate_list(output_path, idx, s, date, city)
    else:
        _generate_cghs_policy_circular(output_path, idx, s, date, city)


def _generate_cghs_rate_list(output_path, idx, s, date, city):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    story.append(Paragraph("CENTRAL GOVERNMENT HEALTH SCHEME (CGHS)", s["title"]))
    story.append(Paragraph("Ministry of Health and Family Welfare | Government of India", s["subtitle"]))
    story.append(Paragraph(f"REVISED RATE LIST FOR CGHS EMPANELLED HOSPITALS — {city} WELLNESS CENTRE", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"This revised rate list is issued in supersession of earlier rate schedules and "
        f"shall be effective from {date}. All CGHS empanelled hospitals in {city} are "
        f"directed to adhere strictly to these rates for treatment of CGHS beneficiaries. "
        f"Non-adherence shall result in de-empanelment under Rule 14 of CGHS Rules.",
        s["body"]))

    # Rate tables by category
    categories = [
        {
            "name": "CONSULTATION CHARGES",
            "rates": [
                ["General Physician (OPD)", "Non-NABH: Rs. 200", "NABH: Rs. 250", "NABH+ Rs. 300"],
                ["Specialist Consultation", "Non-NABH: Rs. 350", "NABH: Rs. 400", "NABH+: Rs. 500"],
                ["Super Specialist", "Non-NABH: Rs. 500", "NABH: Rs. 600", "NABH+: Rs. 750"],
                ["Telemedicine Consultation", "Rs. 150 (all categories)", "", ""],
                ["ICU Monitoring (per day)", "Rs. 3,500", "Rs. 4,000", "Rs. 4,500"],
            ]
        },
        {
            "name": "DIAGNOSTIC INVESTIGATIONS",
            "rates": [
                ["Complete Blood Count (CBC)", "Rs. 150", "Rs. 175", "Rs. 200"],
                ["HbA1c (Glycosylated Haemoglobin)", "Rs. 350", "Rs. 380", "Rs. 420"],
                ["Lipid Profile (Full)", "Rs. 300", "Rs. 340", "Rs. 380"],
                ["Thyroid Profile (T3, T4, TSH)", "Rs. 400", "Rs. 440", "Rs. 500"],
                ["2D Echocardiography", "Rs. 1,200", "Rs. 1,400", "Rs. 1,600"],
                ["MRI Brain Plain", "Rs. 4,500", "Rs. 5,000", "Rs. 5,500"],
                ["CT Chest with Contrast", "Rs. 3,000", "Rs. 3,500", "Rs. 4,000"],
                ["PET CT Scan (Whole Body)", "Rs. 18,000", "Rs. 20,000", "Rs. 22,000"],
            ]
        },
        {
            "name": "SURGICAL PROCEDURES — DAY CARE",
            "rates": [
                ["Cataract with Phaco + IOL", "Rs. 12,000", "Rs. 14,000", "Rs. 16,000"],
                ["Tonsillectomy", "Rs. 10,000", "Rs. 12,000", "Rs. 14,000"],
                ["Circumcision", "Rs. 5,000", "Rs. 6,000", "Rs. 7,000"],
                ["Varicose Vein (Endovenous)", "Rs. 25,000", "Rs. 28,000", "Rs. 32,000"],
                ["Laparoscopic Hernia Repair", "Rs. 30,000", "Rs. 35,000", "Rs. 40,000"],
            ]
        },
        {
            "name": "MAJOR SURGICAL PROCEDURES",
            "rates": [
                ["CABG (Open Heart Bypass)", "Rs. 1,50,000", "Rs. 1,75,000", "Rs. 2,00,000"],
                ["Total Knee Replacement", "Rs. 1,00,000", "Rs. 1,20,000", "Rs. 1,40,000"],
                ["Total Hip Replacement", "Rs. 1,10,000", "Rs. 1,30,000", "Rs. 1,50,000"],
                ["Spinal Fusion (Lumbar)", "Rs. 1,40,000", "Rs. 1,65,000", "Rs. 1,90,000"],
                ["Liver Transplant (Cadaveric)", "Rs. 10,00,000", "Rs. 12,00,000", "Rs. 15,00,000"],
                ["Kidney Transplant", "Rs. 7,00,000", "Rs. 8,50,000", "Rs. 10,00,000"],
            ]
        },
    ]

    for cat in categories:
        story.append(Paragraph(cat["name"], s["section"]))
        header = ["Procedure / Investigation", "Non-NABH Rate", "NABH Rate", "NABH+ Rate"]
        rows = [header] + cat["rates"]
        tbl = Table(rows, colWidths=[8*cm, 3.5*cm, 3.5*cm, 3.5*cm])
        tbl.setStyle(std_table_style())
        story.append(tbl)
        story.append(Spacer(1, 6))

    story.append(Paragraph("TERMS & CONDITIONS", s["section"]))
    conditions = [
        "1. Rates mentioned are inclusive of surgeon's fee, OT charges, anaesthesia, and nursing charges. Implant costs are to be reimbursed at actual cost with bills, subject to CGHS ceiling.",
        "2. Room rent is payable separately at Rs. 1,200/day (General Ward), Rs. 2,500/day (Semi-Private), and Rs. 4,500/day (ICU) for Non-NABH hospitals.",
        "3. For procedures not listed in this schedule, rates shall be as per the CGHS package rates revised circular CGHS/MH/2023/04 dated 01/04/2023.",
        "4. All claims must be submitted within 60 days of treatment. Delayed claims shall attract 10% deduction per month of delay beyond the stipulated period.",
        "5. Hospitals found charging beyond CGHS rates shall be liable for immediate suspension of empanelment and recovery of excess amount charged with 24% interest per annum.",
    ]
    for cond in conditions:
        story.append(Paragraph(cond, s["clause"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(f"Additional Director (CGHS) | {city} | Date: {date}", s["body"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "CGHS Helpline: 1800-11-2644 (Toll Free) | Website: cghs.gov.in | "
        "This is a computer-generated document. For disputes, contact CGHS Wellness Centre.",
        s["footer"]))

    doc.build(story)
    print(f"  [OK] CGHS Rate List {idx}: {os.path.basename(output_path)}")


def _generate_cghs_policy_circular(output_path, idx, s, date, city):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    circ_no = f"CGHS/{city[:3].upper()}/{random.randint(100,999)}/MH/{date.split('/')[-1]}"
    topics = [
        {
            "subject": "Revised Procedure for Cashless Treatment at Empanelled Hospitals",
            "paras": [
                ("Background", "CGHS has been receiving representations from beneficiaries regarding delays in obtaining cashless treatment authorisation at empanelled hospitals. In order to streamline the process and ensure timely treatment, the following revised procedure is being issued with immediate effect."),
                ("Emergency Treatment", "In case of emergency, the empanelled hospital shall provide treatment immediately without waiting for prior permission. The hospital shall intimate the concerned CGHS Wellness Centre within 24 hours of admission. The Medical Officer In-Charge (MOIC) of the Wellness Centre shall issue provisional permission within 4 hours of receiving the intimation."),
                ("Planned Treatment", "For planned procedures, the beneficiary shall obtain a referral from the CGHS Wellness Centre or an authorised specialist. The hospital shall submit the treatment plan and estimated cost to the Wellness Centre. Authorisation shall be granted within 72 hours. In case of non-receipt of authorisation within 72 hours, the hospital may proceed with treatment and the same shall be deemed to be authorised."),
                ("Claim Submission", "All cashless claim documents shall be submitted in soft copy (scanned PDFs) on the CGHS portal within 15 days of discharge. Hard copies shall be retained for 3 years. Incomplete submissions shall be returned within 7 days with specific deficiencies noted."),
                ("Penalty for Hospitals", "Hospitals found to have refused cashless treatment to eligible CGHS beneficiaries without valid reason shall be issued a show-cause notice. Repeated violations shall lead to suspension of empanelment for 6 months and forfeiture of pending claims."),
            ]
        },
        {
            "subject": "Guidelines for Reimbursement of Treatment Taken Outside CGHS Covered Cities",
            "paras": [
                ("Scope", "These guidelines apply to CGHS beneficiaries who undertake treatment at non-CGHS cities or at non-empanelled hospitals due to emergency or non-availability of specialised treatment in the beneficiary's home city."),
                ("Eligibility for Reimbursement", "Reimbursement shall be permissible if: (a) The treatment is for a condition covered under CGHS; (b) A referral from a CGHS Medical Officer or empanelled specialist has been obtained (except in emergencies); (c) The treating hospital is a registered private hospital or government hospital."),
                ("Rate of Reimbursement", "Reimbursement shall be at CGHS rates applicable to non-NABH hospitals in the nearest CGHS-covered city. For specialised procedures not listed in CGHS schedule, reimbursement shall be at rates approved by the Standing Committee of the Ministry of Health."),
                ("Documentation", "Claim for reimbursement shall include: (a) Original bills and receipts; (b) Discharge summary signed by treating doctor; (c) Investigation reports; (d) Referral letter (if applicable); (e) Emergency certificate from treating hospital (for emergency cases)."),
                ("Time Limit", "Reimbursement claims shall be submitted within 6 months of treatment. Claims submitted after 6 months shall not be entertained except in cases of genuine hardship approved by the Additional Director, CGHS with specific reasons recorded in writing."),
            ]
        },
    ]

    topic = topics[(idx // 2) % len(topics)]
    story.append(Paragraph("CENTRAL GOVERNMENT HEALTH SCHEME", s["title"]))
    story.append(Paragraph("Ministry of Health and Family Welfare, Government of India", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 8))

    meta = [
        ["Circular No.", circ_no, "Date", date],
        ["Subject", topic["subject"], "", ""],
    ]
    meta_tbl = Table(meta, colWidths=[3*cm, 9*cm, 2*cm, 4*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("SPAN", (1,1), (-1,1)),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    for para_title, para_text in topic["paras"]:
        story.append(Paragraph(para_title.upper(), s["section"]))
        story.append(Paragraph(para_text, s["body"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "This circular supersedes all previous orders on the subject. "
        "All CGHS Wellness Centres, Additional Directors, and empanelled hospitals "
        "are directed to comply with these guidelines strictly.",
        s["body"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"(Dr. _____________)", s["body"]))
    story.append(Paragraph(f"Director General, CGHS | New Delhi | Date: {date}", s["body"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"Copy to: All Ministries/Departments | All Additional Directors CGHS | "
        f"NIC for uploading on cghs.gov.in",
        s["footer"]))

    doc.build(story)
    print(f"  [OK] CGHS Policy Circular {idx}: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(99)
    print("\n" + "="*55)
    print("  MediFinance Gap-Fill Synthetic Generator")
    print("  SEBI Circulars + CGHS Rate Lists & Policies")
    print("="*55)

    os.makedirs(f"{OUTPUT_DIR}/sebi_circulars", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/cghs_documents", exist_ok=True)

    print("\n[1/2] Generating 3 SEBI Circulars...")
    for i in range(1, 4):
        path = f"{OUTPUT_DIR}/sebi_circulars/SEBI_Circular_{i:02d}.pdf"
        generate_sebi_circular(path, i)

    print("\n[2/2] Generating 3 CGHS Documents...")
    for i in range(1, 4):
        path = f"{OUTPUT_DIR}/cghs_documents/CGHS_Document_{i:02d}.pdf"
        generate_cghs_document(path, i)

    print("\n" + "="*55)
    print("  Done! 6 additional documents generated.")
    print(f"  SEBI Circulars : {OUTPUT_DIR}/sebi_circulars/")
    print(f"  CGHS Documents : {OUTPUT_DIR}/cghs_documents/")
    print("="*55)


if __name__ == "__main__":
    main()