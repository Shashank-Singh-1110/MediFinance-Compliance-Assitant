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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

OUTPUT_DIR = "data/synthetic"
NUM_INSURANCE_DOCS = 10
NUM_AUDIT_DOCS = 10
NUM_BILLING_DOCS = 10

INDIAN_NAMES = [
    "Rajesh Kumar Sharma", "Priya Venkataraman", "Anil Mehta",
    "Sunita Devi Patel", "Vikram Singh Chauhan", "Meena Iyer",
    "Suresh Babu Reddy", "Kavitha Nair", "Amit Kumar Joshi",
    "Pooja Gupta", "Ramesh Chandra Mishra", "Anita Bhatt",
    "Sanjay Kumar Das", "Lakshmi Narayanan", "Deepak Verma"
]

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Bhopal", "Indore", "Nagpur", "Surat", "Kochi"
]

INDIAN_HOSPITALS = [
    "Apollo Hospitals", "Fortis Healthcare", "AIIMS New Delhi",
    "Manipal Hospital", "Narayana Health", "Max Super Speciality Hospital",
    "Medanta - The Medicity", "Kokilaben Dhirubhai Ambani Hospital",
    "Christian Medical College Vellore", "Tata Memorial Hospital",
    "Sir Ganga Ram Hospital", "Lilavati Hospital", "Jaslok Hospital",
    "Hinduja Hospital", "PD Hinduja National Hospital"
]

INSURANCE_COMPANIES = [
    "Star Health and Allied Insurance Co. Ltd.",
    "HDFC ERGO General Insurance Co. Ltd.",
    "ICICI Lombard General Insurance Co. Ltd.",
    "Niva Bupa Health Insurance Co. Ltd.",
    "United India Insurance Co. Ltd.",
    "National Insurance Co. Ltd.",
    "Oriental Insurance Co. Ltd.",
    "New India Assurance Co. Ltd.",
    "Care Health Insurance Ltd.",
    "Aditya Birla Health Insurance Co. Ltd."
]

CA_FIRMS = [
    "Deloitte Haskins & Sells LLP",
    "Price Waterhouse & Co. LLP",
    "B S R & Co. LLP (KPMG)",
    "S.R. Batliboi & Associates LLP (EY)",
    "Walker Chandiok & Co. LLP (Grant Thornton)",
    "Lodha & Co. Chartered Accountants",
    "MSKA & Associates",
    "CNK & Associates LLP"
]

MEDICAL_PROCEDURES = [
    ("Coronary Artery Bypass Graft (CABG)", "H0301", 3, 7, 185000, 320000),
    ("Total Knee Replacement (TKR)", "MS0401", 5, 10, 120000, 250000),
    ("Laparoscopic Cholecystectomy", "GS0201", 2, 4, 45000, 90000),
    ("Appendectomy (Open)", "GS0101", 2, 5, 35000, 70000),
    ("Cataract Surgery with IOL", "OP0101", 1, 2, 18000, 45000),
    ("Normal Delivery", "OB0101", 2, 4, 25000, 55000),
    ("Caesarean Section", "OB0201", 3, 6, 45000, 95000),
    ("Hemodialysis (per session)", "NP0101", 1, 1, 1800, 4500),
    ("Chemotherapy (per cycle)", "ON0101", 1, 3, 25000, 85000),
    ("MRI Brain with Contrast", "RD0201", 1, 1, 6000, 14000),
    ("Angioplasty with Stenting", "CV0201", 3, 6, 150000, 280000),
    ("Hip Replacement Surgery", "MS0501", 5, 10, 130000, 260000),
    ("Spinal Fusion Surgery", "NS0301", 5, 12, 175000, 350000),
    ("Liver Transplant", "GS0801", 14, 30, 1200000, 2500000),
    ("Kidney Transplant", "NP0201", 10, 21, 800000, 1500000),
]

ICD10_CODES = [
    ("I21.0", "Acute transmural myocardial infarction of anterior wall"),
    ("J18.9", "Pneumonia, unspecified organism"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("K80.20", "Calculus of gallbladder without cholecystitis"),
    ("M17.11", "Primary osteoarthritis, right knee"),
    ("C34.10", "Malignant neoplasm of upper lobe, bronchus or lung"),
    ("N18.6", "End stage renal disease"),
    ("I63.9", "Cerebral infarction, unspecified"),
    ("K35.80", "Other and unspecified acute appendicitis"),
    ("O82", "Encounter for cesarean delivery without indication"),
]

CGHS_RATES = {
    "Consultation (Specialist)": 350,
    "Consultation (General Physician)": 200,
    "ECG": 150,
    "2D Echo": 1200,
    "X-Ray Chest (PA View)": 200,
    "Complete Blood Count (CBC)": 180,
    "Blood Sugar (Fasting)": 80,
    "Lipid Profile": 350,
    "Thyroid Function Test (T3, T4, TSH)": 450,
    "Urine Routine & Microscopy": 100,
    "MRI Brain Plain": 5000,
    "CT Scan Abdomen with Contrast": 3500,
    "ICU Charges (per day)": 4500,
    "General Ward (per day)": 1200,
    "OT Charges (Major Surgery)": 15000,
}

def random_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%d/%m/%Y")

def random_policy_number():
    return f"IRDAI/{random.choice(['HLT','GEN','MED'])}/{random.randint(100,999)}/{random.randint(10000,99999)}/{''.join([str(random.randint(0,9)) for _ in range(4)])}"

def random_amount(low, high):
    return f"₹{random.randint(low // 1000, high // 1000) * 1000:,}"

def random_cin():
    return f"L{random.randint(10000,99999)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(1990,2010)}PLC{random.randint(100000,999999)}"

def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "doc_title": ParagraphStyle("doc_title", parent=styles["Title"],
            fontSize=16, textColor=colors.HexColor("#1a3c5e"),
            spaceAfter=6, alignment=TA_CENTER),
        "doc_subtitle": ParagraphStyle("doc_subtitle", parent=styles["Normal"],
            fontSize=11, textColor=colors.HexColor("#2c6496"),
            spaceAfter=4, alignment=TA_CENTER),
        "section_header": ParagraphStyle("section_header", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#1a3c5e"),
            spaceBefore=14, spaceAfter=6, borderPad=4,
            backColor=colors.HexColor("#eaf2fb")),
        "body": ParagraphStyle("body", parent=styles["Normal"],
            fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
            spaceAfter=6),
        "clause": ParagraphStyle("clause", parent=styles["Normal"],
            fontSize=9, leading=13, leftIndent=20,
            spaceAfter=4, alignment=TA_JUSTIFY),
        "footer": ParagraphStyle("footer", parent=styles["Normal"],
            fontSize=8, textColor=colors.grey,
            alignment=TA_CENTER),
        "bold_label": ParagraphStyle("bold_label", parent=styles["Normal"],
            fontSize=9.5, fontName="Helvetica-Bold"),
        "disclaimer": ParagraphStyle("disclaimer", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#555555"),
            leftIndent=10, rightIndent=10, alignment=TA_JUSTIFY),
    }
    return styles, custom

# GENERATOR 1: IRDAI INSURANCE POLICY DOCUMENT
def generate_insurance_policy(output_path, doc_index):
    insurer = random.choice(INSURANCE_COMPANIES)
    policyholder = random.choice(INDIAN_NAMES)
    city = random.choice(INDIAN_CITIES)
    policy_no = random_policy_number()
    issue_date = random_date(2022, 2023)
    expiry_date = random_date(2024, 2025)
    sum_insured = random.choice([300000, 500000, 1000000, 2000000, 5000000])
    premium = int(sum_insured * random.uniform(0.025, 0.045))
    age = random.randint(25, 65)
    plan_name = random.choice([
        "Comprehensive Health Shield Plan",
        "Family Floater Health Guard",
        "Senior Citizen Health Protect",
        "Critical Illness Benefit Plan",
        "Individual Mediclaim Policy"
    ])

    styles, custom = get_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)

    story = []

    story.append(Paragraph("Insurance Regulatory and Development Authority of India (IRDAI)", custom["doc_subtitle"]))
    story.append(Paragraph(f"{insurer}", custom["doc_title"]))
    story.append(Paragraph(f"{plan_name}", custom["doc_subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("POLICY SCHEDULE", custom["section_header"]))
    policy_data = [
        ["Policy Number", policy_no, "Plan Type", "Individual Health Insurance"],
        ["Policyholder Name", policyholder, "Date of Birth", f"Age: {age} Years"],
        ["Policy Issue Date", issue_date, "Policy Expiry Date", expiry_date],
        ["Sum Insured", f"Rs. {sum_insured:,}/-", "Annual Premium", f"Rs. {premium:,}/-"],
        ["GST (18%)", f"Rs. {int(premium*0.18):,}/-", "Total Premium", f"Rs. {int(premium*1.18):,}/-"],
        ["City / Location", city, "Policy Type", "Yearly Renewable"],
    ]
    tbl = Table(policy_data, colWidths=[4*cm, 6*cm, 4*cm, 5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf2fb")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("SECTION A — COVERAGE BENEFITS", custom["section_header"]))
    benefits = [
        ("1. In-Patient Hospitalisation", f"Covered up to Sum Insured of Rs. {sum_insured:,}/- per policy year. Minimum 24-hour hospitalisation required."),
        ("2. Pre-Hospitalisation Expenses", "Medical expenses incurred 60 days prior to hospitalisation are covered, subject to the ailment being covered under this policy."),
        ("3. Post-Hospitalisation Expenses", "Medical expenses incurred up to 90 days after discharge are covered up to 10% of Sum Insured or Rs. 50,000/-, whichever is lower."),
        ("4. Day Care Procedures", "135 Day Care procedures listed under Annexure II are covered without the 24-hour hospitalisation requirement."),
        ("5. AYUSH Treatment", "Treatment under Ayurveda, Yoga, Unani, Siddha, and Homeopathy in a registered AYUSH Hospital is covered up to Rs. 25,000/- per policy year."),
        ("6. Ambulance Charges", "Emergency ambulance charges covered up to Rs. 5,000/- per hospitalisation."),
        ("7. Organ Donor Expenses", "Medical expenses of the organ donor for harvesting of the organ are covered up to Rs. 1,00,000/-."),
        ("8. Domiciliary Hospitalisation", "Treatment taken at home for more than 3 days when hospitalisation is not possible is covered up to 20% of Sum Insured."),
    ]
    for title, text in benefits:
        story.append(Paragraph(f"<b>{title}:</b> {text}", custom["clause"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("SECTION B — GENERAL EXCLUSIONS", custom["section_header"]))
    exclusions = [
        "Pre-existing diseases during the first 48 months of continuous coverage from policy inception date.",
        "Any disease contracted within the first 30 days of policy commencement (not applicable on renewal).",
        "Expenses related to self-inflicted injuries, suicide attempts, or abuse of intoxicants.",
        "Cosmetic or aesthetic treatments, plastic surgery unless necessitated by an accident or illness.",
        "Treatment for obesity, weight control programs, or any complications arising therefrom.",
        "Expenses arising from war, invasion, act of foreign enemy, hostilities, civil war, or nuclear hazard.",
        "Dental treatment unless requiring hospitalisation due to an accidental injury.",
        "Experimental or unproven treatments not approved by the Drug Controller General of India (DCGI).",
        "Maternity expenses including delivery and lawful medical termination of pregnancy (unless add-on opted).",
        "Non-allopathic treatment except AYUSH as specified under Section A, Clause 5.",
    ]
    for i, exc in enumerate(exclusions, 1):
        story.append(Paragraph(f"{i}. {exc}", custom["clause"]))

    story.append(PageBreak())

    story.append(Paragraph("SECTION C — CLAIMS PROCEDURE", custom["section_header"]))
    story.append(Paragraph(
        "The Policyholder or nominee shall notify the Company's 24x7 toll-free helpline "
        "(1800-XXX-XXXX) immediately upon hospitalisation or within 24 hours of emergency admission. "
        "For planned hospitalisation, prior authorisation must be obtained at least 48 hours in advance.",
        custom["body"]))

    claim_steps = [
        ["Step", "Action", "Timeline", "Documents Required"],
        ["1", "Intimate the insurer / TPA", "Within 24 hrs of admission", "Policy number, Hospital details"],
        ["2", "Cashless authorisation (network hospitals)", "Before discharge", "Pre-auth form, Doctor's certificate"],
        ["3", "Submit claim documents (reimbursement)", "Within 30 days of discharge", "Original bills, Discharge summary, Lab reports"],
        ["4", "Claim processing by insurer", "Within 30 days of document receipt", "—"],
        ["5", "Settlement / Deficiency letter", "Within 15 days of processing", "Bank details for NEFT transfer"],
    ]
    claim_tbl = Table(claim_steps, colWidths=[1.5*cm, 5*cm, 4*cm, 6.5*cm])
    claim_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(claim_tbl)
    story.append(Spacer(1, 12))

    # Penalty & Regulatory
    story.append(Paragraph("SECTION D — REGULATORY COMPLIANCE & GRIEVANCE REDRESSAL", custom["section_header"]))
    story.append(Paragraph(
        "This policy is issued in accordance with the Insurance Act, 1938 (as amended), the IRDAI (Health Insurance) "
        "Regulations, 2016, and the Protection of Policyholders' Interests Regulations, 2017. "
        "The policy is subject to the guidelines issued by IRDAI from time to time.",
        custom["body"]))
    story.append(Paragraph(
        "<b>Grievance Redressal:</b> In case of any grievance, the policyholder may approach the Company's "
        "grievance cell at grievance@insurer.in or call 1800-XXX-XXXX. If unresolved within 30 days, "
        "the complaint may be escalated to the Insurance Ombudsman under Rule 13 of the Insurance "
        "Ombudsman Rules, 2017, or to IRDAI's Bima Bharosa portal at igms.irda.gov.in.",
        custom["body"]))

    story.append(Paragraph(
        "<b>Penalty for Late Claim Submission:</b> Claims not intimated within the stipulated timeframe "
        "may be rejected or subject to a 10% penalty on the admissible claim amount, as per IRDAI "
        "circular IRDAI/HLT/REG/CIR/194/08/2020. The Company reserves the right to call for additional "
        "documents or conduct investigation before settlement.",
        custom["body"]))

    story.append(Spacer(1, 20))
    sig_data = [
        [f"For {insurer}", "", "Policyholder Signature"],
        ["", "", ""],
        ["Authorised Signatory", f"Date: {issue_date}", "Date: ___________"],
        ["(Principal Officer)", "", ""],
    ]
    sig_tbl = Table(sig_data, colWidths=[7*cm, 3*cm, 7*cm])
    sig_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,0), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,0), "Helvetica-Bold"),
        ("LINEABOVE", (0,2), (0,2), 0.5, colors.black),
        ("LINEABOVE", (2,2), (2,2), 0.5, colors.black),
        ("TOPPADDING", (0,1), (-1,1), 20),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"IRDAI Registration No.: {random.randint(100,199)} | CIN: {random_cin()} | "
        f"Registered Office: {random.choice(INDIAN_CITIES)} | "
        "This is a computer-generated document. This policy is subject to IRDAI regulations.",
        custom["footer"]))

    doc.build(story)
    print(f"  [OK] Insurance Policy {doc_index}: {os.path.basename(output_path)}")



# GENERATOR 2: CAG / SEBI FINANCIAL AUDIT REPORT
def generate_financial_audit_report(output_path, doc_index):
    company = f"{random.choice(['Bharat', 'Hindustan', 'National', 'India', 'Apex'])} " \
              f"{random.choice(['Healthcare', 'Pharma', 'Medical Systems', 'Life Sciences', 'Diagnostics'])} " \
              f"{random.choice(['Ltd.', 'Pvt. Ltd.', 'Corporation Ltd.'])}"
    auditor = random.choice(CA_FIRMS)
    city = random.choice(INDIAN_CITIES)
    fy = random.choice(["2022-23", "2023-24"])
    revenue = random.randint(500, 5000) * 1000000
    profit = int(revenue * random.uniform(0.05, 0.18))
    audit_date = random_date(2023, 2024)
    cin = random_cin()

    styles, custom = get_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph("INDEPENDENT AUDITOR'S REPORT", custom["doc_title"]))
    story.append(Paragraph(f"To the Members of {company}", custom["doc_subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("COMPANY INFORMATION", custom["section_header"]))
    info_data = [
        ["Company Name", company, "CIN", cin],
        ["Registered Office", city, "Financial Year", f"FY {fy}"],
        ["Auditor", auditor, "Audit Date", audit_date],
        ["Reporting Framework", "Indian Accounting Standards (Ind AS)", "Audit Type", "Statutory Audit"],
    ]
    info_tbl = Table(info_data, colWidths=[4*cm, 6*cm, 3.5*cm, 5.5*cm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 10))

    # Report on Financial Statements
    story.append(Paragraph("REPORT ON THE AUDIT OF STANDALONE FINANCIAL STATEMENTS", custom["section_header"]))
    story.append(Paragraph("<b>Opinion</b>", custom["bold_label"]))
    story.append(Paragraph(
        f"We have audited the standalone financial statements of {company} ('the Company'), "
        f"which comprise the Balance Sheet as at March 31, {fy.split('-')[1]}, "
        "the Statement of Profit and Loss (including Other Comprehensive Income), "
        "the Statement of Changes in Equity and the Statement of Cash Flows for the year then ended, "
        "and notes to the financial statements, including a summary of material accounting policies "
        "and other explanatory information (hereinafter referred to as 'the standalone financial statements').",
        custom["body"]))
    story.append(Paragraph(
        "In our opinion and to the best of our information and according to the explanations given to us, "
        "the aforesaid standalone financial statements give the information required by the Companies Act, 2013 "
        "in the manner so required and give a true and fair view in conformity with the Indian Accounting "
        "Standards (Ind AS) prescribed under Section 133 of the Companies Act, 2013 read with the Companies "
        "(Indian Accounting Standards) Rules, 2015, as amended.",
        custom["body"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Basis for Opinion</b>", custom["bold_label"]))
    story.append(Paragraph(
        "We conducted our audit in accordance with the Standards on Auditing (SAs) specified under "
        "Section 143(10) of the Companies Act, 2013. Our responsibilities under those Standards are further "
        "described in the Auditor's Responsibilities for the Audit of the Financial Statements section "
        "of our report. We are independent of the Company in accordance with the Code of Ethics issued "
        "by the Institute of Chartered Accountants of India (ICAI) together with the ethical requirements "
        "that are relevant to our audit of the financial statements under the provisions of the Companies Act, "
        "2013 and the Rules thereunder, and we have fulfilled our other ethical responsibilities.",
        custom["body"]))

    story.append(Paragraph("KEY AUDIT MATTERS", custom["section_header"]))
    story.append(Paragraph(
        "Key audit matters are those matters that, in our professional judgement, were of most significance "
        "in our audit of the standalone financial statements of the current period. These matters were addressed "
        "in the context of our audit of the standalone financial statements as a whole.",
        custom["body"]))

    kam_data = [
        ["#", "Key Audit Matter", "Auditor's Response"],
        ["1", "Revenue Recognition from Healthcare Services\n"
              "Risk: Complex billing arrangements with insurance companies and TPAs create risk of incorrect revenue recognition.",
              "Tested controls over billing systems; performed substantive testing of a sample of revenue transactions against TPA settlement statements."],
        ["2", "Valuation of Medical Inventory & Consumables\n"
              "Risk: High-value pharmaceutical inventory may be misstated due to obsolescence or expiry.",
              "Attended physical inventory count; verified NRV testing for slow-moving items; reviewed expiry tracking procedures."],
        ["3", "SEBI Compliance — Related Party Transactions\n"
              "Risk: Transactions with promoter group entities may not comply with SEBI LODR Regulations, 2015.",
              "Reviewed all related party contracts; verified Audit Committee and shareholder approvals per SEBI Circular SEBI/HO/CFD/CMD1/CIR/P/2021/662."],
        ["4", "Provision for Doubtful Receivables\n"
              "Risk: Trade receivables from government schemes (PM-JAY, CGHS) may be overstated.",
              "Analysed ageing of receivables; reviewed correspondence with scheme administrators; assessed adequacy of provisions per Ind AS 109."],
    ]
    kam_tbl = Table(kam_data, colWidths=[1*cm, 7.5*cm, 8.5*cm])
    kam_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("WORDWRAP", (0,0), (-1,-1), True),
    ]))
    story.append(kam_tbl)

    story.append(PageBreak())

    story.append(Paragraph("FINANCIAL HIGHLIGHTS — STANDALONE", custom["section_header"]))
    fin_data = [
        ["Particulars", f"FY {fy} (Rs. in Lakhs)", f"FY {str(int(fy.split('-')[0])-1)+'-'+fy.split('-')[0][-2:]} (Rs. in Lakhs)", "YoY Change"],
        ["Total Revenue from Operations", f"{revenue//100000:,}", f"{int(revenue*0.88)//100000:,}", f"+{random.randint(8,18)}%"],
        ["Other Income", f"{random.randint(50,500):,}", f"{random.randint(30,300):,}", f"+{random.randint(5,20)}%"],
        ["Total Expenses", f"{int(revenue*0.83)//100000:,}", f"{int(revenue*0.75)//100000:,}", f"+{random.randint(6,15)}%"],
        ["EBITDA", f"{int(revenue*0.17)//100000:,}", f"{int(revenue*0.14)//100000:,}", f"+{random.randint(10,25)}%"],
        ["Profit Before Tax (PBT)", f"{int(profit*1.3)//100000:,}", f"{int(profit*1.1)//100000:,}", f"+{random.randint(8,20)}%"],
        ["Tax Expense", f"{int(profit*0.3)//100000:,}", f"{int(profit*0.28)//100000:,}", f"+{random.randint(5,12)}%"],
        ["Profit After Tax (PAT)", f"{profit//100000:,}", f"{int(profit*0.85)//100000:,}", f"+{random.randint(10,22)}%"],
        ["Earnings Per Share (EPS) (Rs.)", f"{random.randint(15,85):.2f}", f"{random.randint(10,70):.2f}", "—"],
    ]
    fin_tbl = Table(fin_data, colWidths=[7*cm, 4.5*cm, 4.5*cm, 3*cm])
    fin_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(fin_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("COMPLIANCE OBSERVATIONS & MANAGEMENT LETTER POINTS", custom["section_header"]))
    observations = [
        ("MLP-01 (High Priority)", "GST Reconciliation Gaps",
         "Discrepancies of Rs. 12.4 lakhs identified between GSTR-2A auto-populated data and purchase register. "
         "Management to reconcile and file amended returns under Section 39 of the CGST Act, 2017."),
        ("MLP-02 (Medium Priority)", "TDS Default on Contractor Payments",
         "TDS under Section 194C was not deducted on payments exceeding Rs. 30,000 to two service vendors. "
         "Estimated short deduction: Rs. 3.2 lakhs. Interest liability under Section 201(1A) may arise."),
        ("MLP-03 (Medium Priority)", "SEBI Insider Trading Policy",
         "The Company's Code of Conduct for Prevention of Insider Trading was not updated in line with "
         "SEBI (Prohibition of Insider Trading) (Amendment) Regulations, 2022. Update required."),
        ("MLP-04 (Low Priority)", "Fixed Asset Register",
         "Physical verification of fixed assets was not conducted during the year. Management is advised "
         "to conduct an annual physical verification as per the Companies Act, 2013 requirements."),
    ]
    for code, title, desc in observations:
        obs_data = [[f"{code} — {title}", desc]]
        obs_tbl = Table(obs_data, colWidths=[5*cm, 12*cm])
        obs_tbl.setStyle(TableStyle([
            ("FONTNAME", (0,0), (0,0), "Helvetica-Bold"),
            ("FONTNAME", (1,0), (1,0), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#eaf2fb")),
            ("PADDING", (0,0), (-1,-1), 6),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(obs_tbl)
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 16))
    story.append(Paragraph(f"For {auditor}", custom["body"]))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Partner | Membership No.: {random.randint(100000,199999)}", custom["body"]))
    story.append(Paragraph(f"UDIN: {audit_date.replace('/','')}{''.join([str(random.randint(0,9)) for _ in range(6)])}", custom["body"]))
    story.append(Paragraph(f"Place: {city} | Date: {audit_date}", custom["body"]))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "This report is prepared in accordance with the Companies Act, 2013, ICAI Standards on Auditing, "
        "and Ind AS. For SEBI-listed entities, additional requirements of SEBI LODR Regulations, 2015 apply.",
        custom["footer"]))

    doc.build(story)
    print(f"  [OK] Financial Audit Report {doc_index}: {os.path.basename(output_path)}")


# GENERATOR 3: CGHS / PM-JAY MEDICAL BILLING
def generate_medical_billing(output_path, doc_index):
    patient = random.choice(INDIAN_NAMES)
    hospital = random.choice(INDIAN_HOSPITALS)
    city = random.choice(INDIAN_CITIES)
    uhid = f"UHID{random.randint(1000000, 9999999)}"
    bill_no = f"BILL/{random.randint(2022,2024)}/{random.randint(10000,99999)}"
    admission_date = random_date(2023, 2024)
    icd_code, icd_desc = random.choice(ICD10_CODES)
    procedure_name, proc_code, min_days, max_days, min_cost, max_cost = random.choice(MEDICAL_PROCEDURES)
    los = random.randint(min_days, max_days)
    base_cost = random.randint(min_cost, max_cost)
    scheme = random.choice(["PM-JAY (Ayushman Bharat)", "CGHS", "ECHS", "State Government Scheme", "Corporate TPA"])
    tpa = random.choice(["Medi Assist India TPA", "MD India Health Insurance TPA",
                          "Paramount Health Services TPA", "Vidal Health TPA", "Genins India TPA"])
    discharge_date_obj = datetime.strptime(admission_date, "%d/%m/%Y") + timedelta(days=los)
    discharge_date = discharge_date_obj.strftime("%d/%m/%Y")

    # Generate itemized bill
    room_charges = los * random.randint(2000, 8000)
    nursing = los * random.randint(500, 2000)
    ot_charges = int(base_cost * 0.15)
    anesthesia = int(base_cost * 0.08)
    surgeon_fee = int(base_cost * 0.20)
    medicines = int(base_cost * 0.18)
    lab_tests = random.randint(5000, 25000)
    imaging = random.randint(3000, 20000)
    consumables = int(base_cost * 0.10)
    misc = random.randint(2000, 8000)
    total = room_charges + nursing + ot_charges + anesthesia + surgeon_fee + medicines + lab_tests + imaging + consumables + misc
    cghs_approved = int(total * random.uniform(0.65, 0.85))
    patient_liability = total - cghs_approved
    gst = int(total * 0.05)
    grand_total = total + gst

    styles, custom = get_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph(hospital.upper(), custom["doc_title"]))
    story.append(Paragraph(f"NABH Accredited | {city}", custom["doc_subtitle"]))
    story.append(Paragraph("FINAL HOSPITAL BILL — MEDICAL CLAIM DOCUMENT", custom["doc_subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c6496")))
    story.append(Spacer(1, 8))

    story.append(Paragraph("PATIENT & ADMISSION DETAILS", custom["section_header"]))
    pat_data = [
        ["UHID", uhid, "Bill Number", bill_no],
        ["Patient Name", patient, "Scheme", scheme],
        ["ICD-10 Diagnosis", icd_code, "Diagnosis Description", icd_desc],
        ["Procedure", procedure_name, "Procedure Code", proc_code],
        ["Date of Admission", admission_date, "Date of Discharge", discharge_date],
        ["Length of Stay", f"{los} Days", "TPA / Insurer", tpa],
    ]
    pat_tbl = Table(pat_data, colWidths=[4*cm, 5.5*cm, 4*cm, 5.5*cm])
    pat_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f5f9ff")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("ITEMIZED BILL OF CHARGES", custom["section_header"]))
    bill_data = [
        ["#", "Description of Service", "Quantity / Days", "Unit Rate (Rs.)", "Amount (Rs.)"],
        ["1", "Room & Board (General Ward / ICU)", f"{los} days", f"{room_charges//los:,}", f"{room_charges:,}"],
        ["2", "Nursing & Monitoring Charges", f"{los} days", f"{nursing//los:,}", f"{nursing:,}"],
        ["3", "Operation Theatre (OT) Charges", "1", f"{ot_charges:,}", f"{ot_charges:,}"],
        ["4", "Anaesthesia Charges", "1", f"{anesthesia:,}", f"{anesthesia:,}"],
        ["5", "Surgeon / Consultant Fees", "1", f"{surgeon_fee:,}", f"{surgeon_fee:,}"],
        ["6", "Medicines & IV Fluids", "—", "—", f"{medicines:,}"],
        ["7", "Laboratory & Pathology Tests", "—", "—", f"{lab_tests:,}"],
        ["8", "Radiology & Imaging", "—", "—", f"{imaging:,}"],
        ["9", "Surgical Consumables & Implants", "—", "—", f"{consumables:,}"],
        ["10", "Miscellaneous Charges", "—", "—", f"{misc:,}"],
        ["", "SUBTOTAL", "", "", f"{total:,}"],
        ["", "GST @ 5% (on applicable services)", "", "", f"{gst:,}"],
        ["", "GRAND TOTAL", "", "", f"{grand_total:,}"],
    ]
    bill_tbl = Table(bill_data, colWidths=[1*cm, 7*cm, 3*cm, 3.5*cm, 3.5*cm])
    bill_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTNAME", (1,-3), (-1,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#eaf2fb")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,1), (-1,-4), [colors.white, colors.HexColor("#f5f9ff")]),
        ("ALIGN", (2,0), (-1,-1), "RIGHT"),
        ("PADDING", (0,0), (-1,-1), 5),
        ("LINEABOVE", (0,-3), (-1,-3), 1, colors.HexColor("#1a3c5e")),
    ]))
    story.append(bill_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("CLAIM SETTLEMENT SUMMARY", custom["section_header"]))
    settle_data = [
        ["Total Bill Amount", f"Rs. {grand_total:,}/-"],
        [f"Amount Approved under {scheme}", f"Rs. {cghs_approved:,}/-"],
        ["Admissible Package Rate (as per NHA / CGHS Schedule)", f"Rs. {int(cghs_approved*0.95):,}/-"],
        ["Non-Admissible / Over CGHS Rate Charges", f"Rs. {grand_total - cghs_approved:,}/-"],
        ["Patient / Beneficiary Liability", f"Rs. {patient_liability:,}/-"],
        ["Amount to be paid by Scheme / TPA", f"Rs. {cghs_approved:,}/-"],
    ]
    settle_tbl = Table(settle_data, colWidths=[10*cm, 7*cm])
    settle_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#eaf2fb")),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b0c4de")),
        ("ROWBACKGROUNDS", (0,0), (-1,-2), [colors.white, colors.HexColor("#f5f9ff")]),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(settle_tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph("COMPLIANCE & BILLING NOTES", custom["section_header"]))
    notes = [
        "All charges are as per CGHS / NHA package rates effective April 2023. Charges exceeding package rates require prior approval from the Controlling Officer / TPA.",
        "ICD-10 coding performed by certified Medical Coder as per WHO International Classification of Diseases (10th Revision) guidelines adopted by MoHFW, Government of India.",
        "GST applicable on non-healthcare services as per GST Council Circular No. 32/06/2018-GST. Pure healthcare services are exempt under Notification No. 12/2017.",
        "This bill has been generated from a NABH-accredited Hospital Management System (HMS). Original receipts available for inspection.",
        "Claim documents to be submitted to TPA within 15 days of discharge. Delay beyond 30 days may result in claim rejection as per policy terms.",
    ]
    for i, note in enumerate(notes, 1):
        story.append(Paragraph(f"{i}. {note}", custom["clause"]))

    story.append(Spacer(1, 16))
    sig_data2 = [
        ["Medical Superintendent / CEO", "Chief Finance Officer", "Patient / Attendee Signature"],
        ["", "", ""],
        [f"{hospital}", f"Date: {discharge_date}", "Date: ___________"],
    ]
    sig_tbl2 = Table(sig_data2, colWidths=[6*cm, 5.5*cm, 5.5*cm])
    sig_tbl2.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,2), (-1,2), "Helvetica"),
        ("TOPPADDING", (0,1), (-1,1), 20),
        ("LINEABOVE", (0,2), (0,2), 0.5, colors.black),
        ("LINEABOVE", (1,2), (1,2), 0.5, colors.black),
        ("LINEABOVE", (2,2), (2,2), 0.5, colors.black),
    ]))
    story.append(sig_tbl2)

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"NABH Accreditation No.: {random.randint(1000,9999)} | NHA Empanelled Hospital | "
        "Ayushman Bharat PM-JAY Empanelment Valid | CGHS Approved Hospital | "
        "This is a system-generated bill and is subject to audit.",
        custom["footer"]))

    doc.build(story)
    print(f"  [OK] Medical Billing Document {doc_index}: {os.path.basename(output_path)}")

def main():
    random.seed(42)
    print("\n" + "="*60)
    print("  MediFinance Synthetic Data Generator — India Context")
    print("="*60)

    dirs = [
        f"{OUTPUT_DIR}/insurance_policies",
        f"{OUTPUT_DIR}/financial_audits",
        f"{OUTPUT_DIR}/medical_billing",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"\n[1/3] Generating {NUM_INSURANCE_DOCS} Insurance Policy Documents...")
    for i in range(1, NUM_INSURANCE_DOCS + 1):
        path = f"{OUTPUT_DIR}/insurance_policies/IRDAI_Health_Policy_{i:02d}.pdf"
        generate_insurance_policy(path, i)

    print(f"\n[2/3] Generating {NUM_AUDIT_DOCS} Financial Audit Reports...")
    for i in range(1, NUM_AUDIT_DOCS + 1):
        path = f"{OUTPUT_DIR}/financial_audits/Financial_Audit_Report_{i:02d}.pdf"
        generate_financial_audit_report(path, i)

    print(f"\n[3/3] Generating {NUM_BILLING_DOCS} Medical Billing Documents...")
    for i in range(1, NUM_BILLING_DOCS + 1):
        path = f"{OUTPUT_DIR}/medical_billing/Medical_Bill_{i:02d}.pdf"
        generate_medical_billing(path, i)

    total = NUM_INSURANCE_DOCS + NUM_AUDIT_DOCS + NUM_BILLING_DOCS
    print("\n" + "="*60)
    print(f"  SUCCESS: {total} documents generated.")
    print(f"  Location: ./{OUTPUT_DIR}/")
    print("="*60)
    print("\n  Breakdown:")
    print(f"    Insurance Policies  : {NUM_INSURANCE_DOCS} PDFs")
    print(f"    Financial Audits    : {NUM_AUDIT_DOCS} PDFs")
    print(f"    Medical Billing     : {NUM_BILLING_DOCS} PDFs")


if __name__ == "__main__":
    main()