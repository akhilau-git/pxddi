import sys
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="AuditDDI_SIH_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    # Color Palette - Modern Medical Tech Dark Theme
    BG_COLOR = RGBColor(11, 19, 43)        # #0B132B Deep Midnight Navy
    CARD_BG = RGBColor(22, 36, 71)         # #162447 Slate Navy Card
    CARD_BORDER = RGBColor(38, 59, 107)    # #263B6B Card Border
    ACCENT_CYAN = RGBColor(0, 212, 255)    # #00D4FF Electric Cyan
    ACCENT_EMERALD = RGBColor(6, 214, 160) # #06D6A0 Safety Mint / Green
    ACCENT_CORAL = RGBColor(255, 107, 107) # #FF6B6B Alert Coral
    ACCENT_GOLD = RGBColor(255, 193, 7)    # #FFC107 Metric Gold
    TEXT_WHITE = RGBColor(255, 255, 255)   # Crisp White
    TEXT_LIGHT = RGBColor(215, 226, 245)   # High legibility soft blue-white
    TEXT_MUTED = RGBColor(140, 160, 190)   # Secondary muted text
    TAG_BG = RGBColor(15, 45, 82)          # Tag pill background

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, category_tag, slide_title, slide_subtitle=None, slide_num=""):
        # Top Accent Line
        top_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.04))
        top_line.fill.solid()
        top_line.fill.fore_color.rgb = ACCENT_CYAN
        top_line.line.fill.background()

        # Category Pill / Tag
        tag_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.55), Inches(3.2), Inches(0.35))
        tag_box.fill.solid()
        tag_box.fill.fore_color.rgb = TAG_BG
        tag_box.line.color.rgb = ACCENT_CYAN
        tag_box.line.width = Pt(1)
        tf = tag_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"SIH 2024-25  |  {category_tag.upper()}"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN

        # Slide Number
        if slide_num:
            num_box = slide.shapes.add_textbox(Inches(11.333), Inches(0.55), Inches(1.2), Inches(0.35))
            np = num_box.text_frame.paragraphs[0]
            np.text = slide_num
            np.alignment = PP_ALIGN.RIGHT
            np.font.size = Pt(11)
            np.font.bold = True
            np.font.color.rgb = TEXT_MUTED

        # Main Title
        tb = slide.shapes.add_textbox(Inches(0.75), Inches(0.95), Inches(11.8), Inches(0.65))
        p = tb.text_frame.paragraphs[0]
        p.text = slide_title
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE

        # Subtitle
        if slide_subtitle:
            p2 = tb.text_frame.add_paragraph()
            p2.text = slide_subtitle
            p2.font.size = Pt(11.5)
            p2.font.color.rgb = TEXT_MUTED

    def add_card(slide, left, top, width, height, title="", title_color=ACCENT_CYAN, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        if title:
            # Header strip inside card
            tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.1), Inches(width - 0.3), Inches(0.45))
            p = tb.text_frame.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = title_color
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)

    # Accent decorative top bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_CYAN
    bar.line.fill.background()

    # Tag
    tag = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.1), Inches(4.2), Inches(0.4))
    tag.fill.solid()
    tag.fill.fore_color.rgb = TAG_BG
    tag.line.color.rgb = ACCENT_CYAN
    tag.line.width = Pt(1)
    tf = tag.text_frame
    p = tf.paragraphs[0]
    p.text = "SMART INDIA HACKATHON  |  INNOVATION IDEA"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    # Title Box
    title_box = s1.shapes.add_textbox(Inches(0.75), Inches(1.7), Inches(11.8), Inches(2.2))
    p_title = title_box.text_frame.paragraphs[0]
    p_title.text = "AuditDDI: Auditable Multimodal Graph AI"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

    p_sub = title_box.text_frame.add_paragraph()
    p_sub.text = "Next-Generation Cold-Start Drug-Drug Interaction Prediction via Molecular GNNs & Evolutionary Protein Modeling"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = ACCENT_CYAN

    # Problem Statement Code & Domain Banner
    banner = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.9), Inches(11.733), Inches(0.65))
    banner.fill.solid()
    banner.fill.fore_color.rgb = CARD_BG
    banner.line.color.rgb = ACCENT_CYAN
    banner.line.width = Pt(1)
    bf = banner.text_frame
    bp = bf.paragraphs[0]
    bp.text = "Theme: MedTech / BioTech / HealthTech  |  Focus Area: Polypharmacy Safety & Preclinical Pharmacology"
    bp.font.size = Pt(12)
    bp.font.bold = True
    bp.font.color.rgb = TEXT_WHITE
    bp.alignment = PP_ALIGN.CENTER

    # Team & Institute Grid (4 Columns)
    cols = [
        ("Team Details", ["Team Name: [Your Team Name]", "Leader: [Team Leader Name]", "Members: [Member 1, 2, 3, 4, 5]"]),
        ("Institution", ["Institute: [Your College Name]", "Department: [CSE / AI / BioTech]", "City / State: [Your City]"]),
        ("Mentor / Guide", ["Faculty Guide: [Guide Name]", "Designation: [Professor / HoD]", "Domain: HealthTech / AI"]),
        ("Submission Category", ["Category: Student Innovation", "Mode: Software / AI Pipeline", "Target: Hospital & Pharma R&D"])
    ]
    for i, (col_title, items) in enumerate(cols):
        left = 0.8 + i * 3.0
        c = add_card(s1, left, 4.8, 2.75, 2.0, col_title, ACCENT_CYAN)
        tb = s1.shapes.add_textbox(Inches(left + 0.12), Inches(5.35), Inches(2.5), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, it in enumerate(items):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = it
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 2: THE PROBLEM & CLINICAL URGENCY
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(s2, "Problem Statement", "The Polypharmacy Crisis & The 'Cold-Start' Blindspot", 
               "Adverse Drug-Drug Interactions (DDIs) cause >10% of hospitalizations worldwide, yet current tools fail on novel molecules.", "02 / 08")

    # 3 Problem Pillars
    pillars = [
        ("1. The Polypharmacy & ADE Threat", ACCENT_CORAL, [
            "Patients with co-morbidities (Diabetes + Hypertension + Oncology) take 5 to 10+ medications simultaneously.",
            "Adverse Drug Events (ADEs) cause 1.3M+ ER visits/year and >$30B in preventable healthcare costs.",
            "In India, multi-specialist prescriptions often lead to unmonitored dangerous drug clashes."
        ]),
        ("2. The 'Cold-Start' Failure of Lookup DBs", ACCENT_GOLD, [
            "Clinical tools (Medscape, DrugBank, 1mg) rely on published historical databases.",
            "Zero Interaction Data: When a new drug is approved or an unstudied combination is tried, lookup databases return 'No interaction found' — giving false security.",
            "Cannot predict interactions for newly synthesized or repurposed drug candidates."
        ]),
        ("3. The Black-Box & Leakage Trap in AI", ACCENT_CYAN, [
            "Existing research AI models (standard GNNs) suffer from severe data leakage (inflating metrics up to 99% in labs, then failing in clinics).",
            "Lack of Explainability: Clinicians cannot trust black-box risk scores without biological evidence or structural reasoning.",
            "Uncalibrated Risk: Models output overconfident probabilities without uncertainty bounds."
        ])
    ]

    for i, (title, color, bullets) in enumerate(pillars):
        left = 0.8 + i * 4.0
        add_card(s2, left, 1.8, 3.75, 4.0, title, color)
        tb = s2.shapes.add_textbox(Inches(left + 0.15), Inches(2.4), Inches(3.45), Inches(3.2))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = f"- {b}"
            p.font.size = Pt(11.5)
            p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(8)

    # Bottom Callout Card
    bot = add_card(s2, 0.8, 6.0, 11.733, 0.9, "", bg_color=TAG_BG, border_color=ACCENT_CYAN)
    tb_bot = s2.shapes.add_textbox(Inches(1.0), Inches(6.1), Inches(11.333), Inches(0.7))
    tf_b = tb_bot.text_frame
    p = tf_b.paragraphs[0]
    p.text = "THE CORE CLINICAL NEED:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p2 = tf_b.add_paragraph()
    p2.text = "An auditable, multimodal system that predicts interaction risks for NOVEL (unseen) drugs, grounds predictions in multi-omics biology, and abstains when uncertain."
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 3: PROPOSED SOLUTION & CORE INNOVATION
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(s3, "Proposed Innovation", "AuditDDI: Multimodal Graph AI & Evolutionary Protein Modeling",
               "A breakthrough architecture combining 2D/3D chemical topology with protein sequence language models and auditable memory.", "03 / 08")

    features = [
        ("Dual-Stream Molecular GNN", ACCENT_CYAN, [
            "Edge-Aware GATv2: Encodes chemical topology, bond types, hybridization, and stereochemistry.",
            "1024-bit Morgan ECFP: Captures high-level functional groups and circular chemical motifs.",
            "Eliminates reliance on drug IDs; operates purely on raw molecular SMILES."
        ]),
        ("ESM-2 Evolutionary Protein Modeling", ACCENT_EMERALD, [
            "UniProt & ESM-2 Integration: Uses Meta's protein language model to read amino acid FASTA sequences directly.",
            "Solves Cold-Target Generalization: Accurately predicts interactions for brand-new enzyme/receptor targets without prior data.",
            "CrossModalBioAttention aligns molecular graphs with target protein binding domains."
        ]),
        ("Multi-Omics Biological Grounding", ACCENT_GOLD, [
            "PharmGKB: Pharmacogenomic enzyme & metabolism pathways (CYP450).",
            "FAERS Real-World Signals: FDA post-marketing clinical adverse event severity.",
            "BindingDB & PDB: 3D macromolecular crystal complexes and target affinity profiles."
        ]),
        ("Auditable Neighbor Memory (RAG-DDI)", ACCENT_CORAL, [
            "Analog Evidence Retrieval: Finds nearest verified chemical analogs in memory to explain WHY two drugs interact.",
            "Conformal Uncertainty: Rather than hallucinating risky guesses, the model quantifies uncertainty and alerts clinical pharmacists.",
            "Strict Commutative Invariance: Guaranteeing f(Drug A, Drug B) === f(Drug B, Drug A) identically."
        ])
    ]

    for i, (title, color, bullets) in enumerate(features):
        row = i // 2
        col = i % 2
        left = 0.8 + col * 6.0
        top = 1.8 + row * 2.6
        add_card(s3, left, top, 5.75, 2.35, title, color)
        tb = s3.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.45), Inches(5.45), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = f"- {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 4: TECHNICAL ARCHITECTURE & WORKFLOW
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(s4, "Technical Architecture", "End-to-End Multimodal Deep Learning Pipeline",
               "From raw chemical SMILES & protein FASTA sequences to cost-calibrated clinical interaction risk.", "04 / 08")

    # 4 Pipeline Stage Cards
    stages = [
        ("STAGE 1: Ingestion & Featurization", ACCENT_CYAN, [
            "Input: Drug SMILES (Drug A, Drug B) + UniProt target sequences.",
            "RDKit Parser: Extracts atomic numbers, hybridization, formal charge, bond stereochemistry.",
            "1024-bit Morgan Fingerprints (radius=2) + ESM-2 residue tokenization.",
            "Multi-omics linking (PharmGKB, FAERS, BindingDB, PDB complexes)."
        ]),
        ("STAGE 2: Dual-Stream Encoding", ACCENT_GOLD, [
            "Edge-Aware GATv2: Multi-head graph attention propagating atom & bond features across chemical space.",
            "ECFP Dense Projection: Compresses high-dimensional circular fingerprints to embedding space.",
            "ESM-2 Protein Encoder: Transforms target amino acid sequence into contextual 320-dim biological vectors."
        ]),
        ("STAGE 3: Cross-Modal Attention & Fusion", ACCENT_EMERALD, [
            "CrossModalBioAttention: Dynamically aligns drug molecular graph with biological target enzymes.",
            "Commutative Pair Fusion: Uses symmetric operations (eA + eB and |eA - eB|) ensuring order independence.",
            "RAG Memory Layer: Retrieves top-k nearest historical analogs for auditable reasoning."
        ]),
        ("STAGE 4: Calibration & Clinical Output", ACCENT_CORAL, [
            "Cost-Sensitive Youden Thresholding: Calibrated for clinical safety (J_cost = 2.0*TPR - FPR).",
            "Out-of-Sample Platt & Temperature Scaling: True calibrated probability scores.",
            "Conformal Prediction Set: Safe abstention flag when confidence falls below 95% threshold."
        ])
    ]

    for i, (stitle, color, bullets) in enumerate(stages):
        left = 0.8 + i * 3.0
        add_card(s4, left, 1.8, 2.8, 4.8, stitle, color)
        tb = s4.shapes.add_textbox(Inches(left + 0.12), Inches(2.4), Inches(2.55), Inches(4.0))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = f"- {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(6)

    # Bottom summary tag
    bot4 = add_card(s4, 0.8, 6.75, 11.733, 0.45, "", bg_color=TAG_BG, border_color=ACCENT_CYAN)
    tb_b4 = s4.shapes.add_textbox(Inches(1.0), Inches(6.8), Inches(11.333), Inches(0.35))
    p = tb_b4.text_frame.paragraphs[0]
    p.text = "Zero Data Leakage: Training, validation, and post-hoc calibration partitions are strictly isolated with Murcko scaffold splits."
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 5: COMPETITIVE ADVANTAGE & NOVELTY MATRIX
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(s5, "Novelty & Comparison", "Competitive Benchmark: Why AuditDDI Outperforms",
               "Comparison against static clinical lookup databases and state-of-the-art research AI models.", "05 / 08")

    # Table Layout
    rows, cols = 7, 4
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.5)
    table_shape = s5.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(2.7)
    table.columns[2].width = Inches(2.7)
    table.columns[3].width = Inches(3.133)

    headers = ["Capabilities & Metrics", "Traditional Databases (DrugBank, Medscape)", "Standard AI Models (Decagon, DeepDDI)", "AuditDDI (Our Innovation)"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN if col_idx == 3 else TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

    matrix_data = [
        ("Cold-Start (Unseen Drugs)", "FAIL (Zero data for novel molecules)", "POOR (~50% AUROC on unseen splits)", "EXCELLENT (Generalizes via ESM-2 & Fingerprints)"),
        ("Chemical Scaffold Generalization", "N/A (Dictionary lookup only)", "FAIL (Suffers from scaffold memorization)", "ROBUST (Murcko Scaffold-Disjoint Audited)"),
        ("Biological Grounding", "Text notes only (Non-computational)", "Single network (PPI only)", "MULTIMODAL (PharmGKB, FAERS, PDB, UniProt)"),
        ("Auditability & Explainability", "Static reference links", "Black-Box (Unexplained probability)", "AUDITABLE (RAG-DDI Analog Memory Retrieval)"),
        ("Uncertainty & Clinical Safety", "Binary warning only", "Overconfident raw logits", "CONFORMAL (Calibrated Abstention on OOD)"),
        ("Inference Speed & Deployment", "Fast lookup (<10ms)", "Heavy graph queries (>500ms)", "HIGH SPEED (<45ms/pair, Dockerized FastAPI)")
    ]

    for r_idx, row_data in enumerate(matrix_data):
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = TAG_BG if c_idx == 3 else (CARD_BG if r_idx % 2 == 0 else RGBColor(16, 26, 52))
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = TEXT_WHITE
                p.alignment = PP_ALIGN.LEFT
            elif c_idx == 3:
                p.font.bold = True
                p.font.color.rgb = ACCENT_EMERALD
                p.alignment = PP_ALIGN.CENTER
            else:
                p.font.color.rgb = TEXT_MUTED
                p.alignment = PP_ALIGN.CENTER

    # Bottom summary
    bot5 = add_card(s5, 0.8, 6.45, 11.733, 0.75, "", bg_color=CARD_BG, border_color=ACCENT_EMERALD)
    tb_b5 = s5.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.333), Inches(0.6))
    p = tb_b5.text_frame.paragraphs[0]
    p.text = "Key Innovation Moat: AuditDDI is the FIRST system to combine ESM-2 protein language modeling with auditable neighbor memory and strict conformal risk abstention for cold-start DDIs."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 6: FEASIBILITY, VALIDATION & TECHNICAL READINESS
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(s6, "Feasibility & Results", "Experimental Validation & Production Readiness",
               "Rigorously audited on 639 curated drugs, 218 passing automated tests, and real-time API serving.", "06 / 08")

    # 3 Stat Cards on Top
    stat_cards = [
        ("0.9516 AUROC", "Validated Accuracy", "Leak-free Transductive Holdout AUROC verified on multi-omics graph", ACCENT_CYAN),
        ("<45 ms / Pair", "Real-Time Inference", "Blazing fast CPU/GPU response time suitable for live prescription screening", ACCENT_EMERALD),
        ("218 Automated Tests", "Software Reliability", "Comprehensive test suite covering leak detection, symmetry & APIs", ACCENT_GOLD)
    ]

    for i, (stat, subtitle, desc, col) in enumerate(stat_cards):
        left = 0.8 + i * 4.0
        add_card(s6, left, 1.8, 3.75, 1.6, "", bg_color=CARD_BG, border_color=col)
        tb = s6.shapes.add_textbox(Inches(left + 0.15), Inches(1.9), Inches(3.45), Inches(1.4))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = stat
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_WHITE
        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = TEXT_MUTED

    # Bottom 2 Wide Columns: Engineering Architecture & Clinical Safety Guardrails
    add_card(s6, 0.8, 3.6, 5.75, 3.4, "Production Architecture (Ready Today)", ACCENT_CYAN)
    tb_arch = s6.shapes.add_textbox(Inches(0.95), Inches(4.1), Inches(5.45), Inches(2.7))
    tf_a = tb_arch.text_frame
    tf_a.word_wrap = True
    arch_bullets = [
        "FastAPI Backend: Asynchronous, typed endpoints (/health, /ready, /predict, /explain).",
        "Interactive Pharmacist Dashboard: Clean web interface with dynamic molecular structure rendering and instant risk scoring.",
        "Containerized & Cloud-Agnostic: Complete Docker & Docker Compose setup, runs on-premise or cloud with read-only root security.",
        "High Throughput: Supports batch prescription screening of 50+ drug combinations in under 2 seconds."
    ]
    for j, b in enumerate(arch_bullets):
        p = tf_a.paragraphs[0] if j == 0 else tf_a.add_paragraph()
        p.text = f"- {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s6, 6.8, 3.6, 5.75, 3.4, "Clinical Safety & Audit Guardrails", ACCENT_EMERALD)
    tb_guard = s6.shapes.add_textbox(Inches(6.95), Inches(4.1), Inches(5.45), Inches(2.7))
    tf_g = tb_guard.text_frame
    tf_g.word_wrap = True
    guard_bullets = [
        "Strict Out-of-Distribution (OOD) Guard: Rejects invalid SMILES, disconnected ions, and chemically unviable structures before graph construction.",
        "Symmetric Guarantee: Commutative forward pass mathematically guarantees f(A, B) == f(B, A), preventing order-dependent prescription errors.",
        "Conformal Abstention: Flags high-uncertainty drug pairs for manual clinical pharmacist review instead of making hazardous guesses.",
        "Cost-Sensitive Tuning: Youden index set with 2x penalty on false negatives to prioritize patient safety."
    ]
    for j, b in enumerate(guard_bullets):
        p = tf_g.paragraphs[0] if j == 0 else tf_g.add_paragraph()
        p.text = f"- {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 7: IMPACT, BENEFICIARIES & NATIONAL ALIGNMENT
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7)
    add_header(s7, "Impact & Benefits", "Healthcare Impact & National Digital Health Alignment",
               "Empowering clinicians, protecting multi-morbid patients, and accelerating drug discovery.", "07 / 08")

    stakeholders = [
        ("Clinicians & Hospitals", ACCENT_CYAN, [
            "Real-time prescription copilot alerting doctors during multi-specialty rounds.",
            "Reduces preventable drug-related ICU admissions and lengths of hospital stay.",
            "Audit trail provides medicolegal documentation and transparent reasoning."
        ]),
        ("Patients & Elderly Care", ACCENT_EMERALD, [
            "Direct protection for elderly polypharmacy patients taking 5+ daily chronic medications.",
            "Prevents silent drug clashes (e.g. fatal arrhythmias from QT prolongation, gastrointestinal bleeding).",
            "Improves quality of life and medication adherence."
        ]),
        ("Pharmaceutical R&D", ACCENT_GOLD, [
            "De-risks drug discovery: Screens novel lead candidates for interaction liabilities before costly clinical trials.",
            "Accelerates combination oncology therapies and antiviral cocktail design.",
            "Significant cost reduction in preclinical pharmacovigilance screening."
        ]),
        ("National Alignment: ABDM", ACCENT_CORAL, [
            "Seamless integration with Ayushman Bharat Digital Mission (ABDM) through FHIR/HL7 APIs.",
            "Intelligent decision support for e-Sanjeevani telemedicine and Jan Aushadhi Kendras.",
            "Contributes to National Pharmacovigilance Programme of India (PvPI)."
        ])
    ]

    for i, (title, color, bullets) in enumerate(stakeholders):
        row = i // 2
        col = i % 2
        left = 0.8 + col * 6.0
        top = 1.8 + row * 2.6
        add_card(s7, left, top, 5.75, 2.35, title, color)
        tb = s7.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.45), Inches(5.45), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = f"- {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 8: ROADMAP & FUTURE SCOPE
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8)
    add_header(s8, "Roadmap & Scaling", "Future Scope & Implementation Roadmap",
               "A structured, milestone-driven execution plan from prototype to national deployment.", "08 / 08")

    phases = [
        ("Phase 1: Current State (Hackathon MVP)", ACCENT_CYAN, [
            "Milestone: Completed & Audited",
            "- Dual-stream GNN + 1024-bit Morgan ECFP pipeline.",
            "- UniProt & ESM-2 protein sequence embedding integration.",
            "- 0.9516 AUROC validated on leak-free holdout splits.",
            "- Containerized FastAPI service + Interactive Doctor Dashboard.",
            "- 218 passing automated tests & rigorous audit fixes."
        ]),
        ("Phase 2: EHR & ABDM Integration (Months 1-6)", ACCENT_GOLD, [
            "Milestone: Clinical Pilot & Hospital EHR Linkage",
            "- Build FHIR / HL7 standard connectors for Hospital Information Systems (HIS).",
            "- Multi-drug regimen screener: Check N-way (3 to 10 drug) combinations simultaneously.",
            "- Pilot trial in tertiary hospital cardiology & oncology inpatient departments.",
            "- Mobile application for rural healthcare workers & pharmacists."
        ]),
        ("Phase 3: National Scale & Pharma API (Months 6-18)", ACCENT_EMERALD, [
            "Milestone: National Telemedicine & Pharma Adoption",
            "- Deploy as a verified clinical copilot module within e-Sanjeevani telemedicine.",
            "- Preclinical screening SaaS portal for Indian pharmaceutical manufacturers.",
            "- Real-world evidence gathering in collaboration with Indian Pharmacopoeia Commission.",
            "- Prospective clinical validation study for publication in peer-reviewed journals."
        ])
    ]

    for i, (title, color, bullets) in enumerate(phases):
        left = 0.8 + i * 4.0
        add_card(s8, left, 1.8, 3.75, 4.2, title, color)
        tb = s8.shapes.add_textbox(Inches(left + 0.15), Inches(2.4), Inches(3.45), Inches(3.5))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = b
            if j == 0:
                p.font.size = Pt(11)
                p.font.bold = True
                p.font.color.rgb = color
            else:
                p.font.size = Pt(10.5)
                p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(5)

    # Bottom Callout
    bot8 = add_card(s8, 0.8, 6.15, 11.733, 0.9, "", bg_color=TAG_BG, border_color=ACCENT_CYAN)
    tb_b8 = s8.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.333), Inches(0.7))
    tf_b8 = tb_b8.text_frame
    p = tf_b8.paragraphs[0]
    p.text = "CONCLUSION & VISION:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p2 = tf_b8.add_paragraph()
    p2.text = "AuditDDI transforms drug safety from retrospective crisis management to proactive, auditable AI prediction — safeguarding millions of vulnerable patients across India and the globe."
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE

    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    out = "AuditDDI_SIH_Presentation.pptx"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    create_deck(out)
