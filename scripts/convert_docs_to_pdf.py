"""
Script to convert all Project Markdown Documentation into Publication-Grade Academic PDFs using ReportLab.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_DIR = os.path.join(PROJECT_ROOT, "docs", "markdown")
PDF_DIR = os.path.join(PROJECT_ROOT, "docs", "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

DOCUMENTS = [
    (os.path.join(MD_DIR, "PROJECT_REPORT.md"), os.path.join(PDF_DIR, "PROJECT_REPORT.pdf"), "TerrAgro: Soil Classification Academic Report"),
    (os.path.join(MD_DIR, "VIVA_PREPARATION_GUIDE.md"), os.path.join(PDF_DIR, "VIVA_PREPARATION_GUIDE.pdf"), "Case Study 139: Viva-Voce Examination Guide"),
    (os.path.join(MD_DIR, "ML_TOPICS_EXPLAINED.md"), os.path.join(PDF_DIR, "ML_TOPICS_EXPLAINED.pdf"), "Machine Learning Topics & Concepts Guide"),
    (os.path.join(MD_DIR, "PROJECT_FILES_EXPLAINED.md"), os.path.join(PDF_DIR, "PROJECT_FILES_EXPLAINED.pdf"), "Project Files & Directory Architecture Guide"),
    (os.path.join(MD_DIR, "ML_MODELS_EXPLAINED.md"), os.path.join(PDF_DIR, "ML_MODELS_EXPLAINED.pdf"), "Machine Learning Models Deep Dive Guide"),
    (os.path.join(PROJECT_ROOT, "README.md"), os.path.join(PDF_DIR, "README.pdf"), "TerrAgro Project Overview & Documentation")
]

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw running header and footer with total page count.
    """
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 11 * 72 - 25, getattr(self, 'doc_title', 'TerrAgro ML Case Study 139'))
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * 72 - 28, 8.5 * 72 - 36, 11 * 72 - 28)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 36, 20, page_str)
        self.drawString(36, 20, "TerrAgro: Soil Classification Using Machine Learning | Case Study 139")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * 72 - 36, 30)
        
        self.restoreState()

def clean_inline_md(text):
    """Converts inline markdown formatting to reportlab XML tags safely."""
    if not text:
        return ""
    
    # 1. Strip images & badges
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    
    # 2. Extract and protect inline code `code`
    code_placeholders = []
    def save_code(match):
        code_content = match.group(1)
        # Escape xml chars in code
        code_content = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        idx = len(code_placeholders)
        code_placeholders.append(f'<font name="Courier" color="#1e293b"><b>{code_content}</b></font>')
        return f"__CODE_PH_{idx}__"
    
    text = re.sub(r'`(.*?)`', save_code, text)

    # 3. Clean LaTeX math formatting $math$ -> simple readable text
    text = re.sub(r'\$\$(.*?)\$\$', r'<b>\1</b>', text)
    text = re.sub(r'\$(.*?)\$', r'<i>\1</i>', text)
    text = text.replace(r'\mathbf', '').replace(r'\text', '').replace(r'\ln', 'ln').replace(r'\sum', 'Σ').replace(r'\arg\max', 'argmax')
    text = text.replace(r'\mu', 'μ').replace(r'\sigma', 'σ').replace(r'\lambda', 'λ').replace(r'\eta', 'η').replace(r'\Delta', 'Δ')
    text = text.replace(r'\circ', '°').replace(r'\cdot', '·').replace(r'\times', '×').replace(r'\pm', '±').replace(r'\le', '≤').replace(r'\ge', '≥')
    text = text.replace(r'\sqrt', '√').replace('{', '').replace('}', '').replace('\\', '')

    # 4. XML escape standalone & < > (before adding real tags)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 5. Markdown Links [text](url) -> <u>text</u>
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'<u>\1</u>', text)
    
    # 6. Bold & Italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Only match _italic_ surrounded by spaces/punctuation, not inside variable names
    text = re.sub(r'(?<=\s)_(.*?)_(?=\s|[.,;:!?]|$)', r'<i>\1</i>', text)
    
    # 7. Restore code placeholders
    for idx, ph_tag in enumerate(code_placeholders):
        text = text.replace(f"__CODE_PH_{idx}__", ph_tag)

    return text.strip()

def build_pdf_from_markdown(md_path, pdf_path, doc_title):
    print(f"📄 Converting: {md_path} -> {pdf_path}...")
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1b4332'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2d6a4f'),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1b4332'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2d6a4f'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=5
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=3
    )
    
    quote_style = ParagraphStyle(
        'Quote',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12.5,
        leftIndent=15,
        rightIndent=15,
        textColor=colors.HexColor('#2d6a4f'),
        spaceBefore=4,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0f172a'),
        leftIndent=10,
        spaceAfter=2
    )
    
    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0
    )
    
    td_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b')
    )

    story = []
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        
        # Calculate column widths
        num_cols = len(table_rows[0])
        avail_width = 8.5 * 72 - 72 # 540 pt
        col_width = avail_width / num_cols
        
        flowable_table_data = []
        for row_idx, row in enumerate(table_rows):
            row_cells = []
            for col_idx, cell in enumerate(row):
                style = th_style if row_idx == 0 else td_style
                p = Paragraph(clean_inline_md(cell), style)
                row_cells.append(p)
            flowable_table_data.append(row_cells)
            
        t = Table(flowable_table_data, colWidths=[col_width] * num_cols)
        t_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d6a4f')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ]
        # Alternating row colors
        for r_idx in range(1, len(flowable_table_data)):
            if r_idx % 2 == 0:
                t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor('#f8fafc')))
        t.setStyle(TableStyle(t_style))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 6))
        table_rows = []

    def flush_code_block():
        nonlocal code_lines
        if not code_lines:
            return
        code_text = "<br/>".join([clean_inline_md(cl) if cl.strip() else "&nbsp;" for cl in code_lines])
        p = Paragraph(code_text, code_style)
        
        # Wrap in shaded table box
        box_table = Table([[p]], colWidths=[540])
        box_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(Spacer(1, 3))
        story.append(box_table)
        story.append(Spacer(1, 5))
        code_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Code block toggle
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                flush_code_block()
            else:
                in_code_block = True
                if in_table:
                    in_table = False
                    flush_table()
            continue

        if in_code_block:
            code_lines.append(line.rstrip("\n"))
            continue

        # Markdown Tables
        if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
            cells_in_row = [c.strip() for c in stripped.strip("|").split("|")]
            # Check if separator row like | :--- | :---: |
            if all(re.match(r'^:?-+:?$', c) for c in cells_in_row if c):
                continue
            in_table = True
            table_rows.append(cells_in_row)
            continue
        else:
            if in_table:
                in_table = False
                flush_table()

        # Empty lines
        if not stripped:
            continue

        # Horizontal rule
        if stripped in ["---", "***", "___"]:
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceAfter=5))
            continue

        # Headings
        if stripped.startswith("# "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(clean_inline_md(stripped[2:]), title_style))
            continue
        elif stripped.startswith("## "):
            story.append(Spacer(1, 4))
            story.append(Paragraph(clean_inline_md(stripped[3:]), h1_style))
            continue
        elif stripped.startswith("### "):
            story.append(Spacer(1, 3))
            story.append(Paragraph(clean_inline_md(stripped[4:]), h2_style))
            continue
        elif stripped.startswith("#### "):
            story.append(Spacer(1, 2))
            story.append(Paragraph(clean_inline_md(stripped[5:]), h3_style))
            continue

        # Blockquotes
        if stripped.startswith(">"):
            quote_text = clean_inline_md(stripped.lstrip("> "))
            story.append(Paragraph(quote_text, quote_style))
            continue

        # Bullet points and numbers
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = f"• {clean_inline_md(stripped[2:])}"
            story.append(Paragraph(bullet_text, bullet_style))
            continue
        elif re.match(r'^\d+\.\s', stripped):
            num_match = re.match(r'^(\d+\.)\s(.*)', stripped)
            if num_match:
                prefix, content = num_match.groups()
                item_text = f"<b>{prefix}</b> {clean_inline_md(content)}"
                story.append(Paragraph(item_text, bullet_style))
            continue

        # Regular paragraph
        story.append(Paragraph(clean_inline_md(stripped), body_style))

    if in_table:
        flush_table()
    if in_code_block:
        flush_code_block()

    # Build with custom canvas for page numbers
    def add_title_to_canvas(c, d):
        c.doc_title = doc_title

    doc.build(story, canvasmaker=NumberedCanvas, onFirstPage=add_title_to_canvas, onLaterPages=add_title_to_canvas)
    print(f"✅ Generated: {pdf_path}")

def main():
    print("🚀 Starting Markdown to PDF Batch Conversion...")
    for md_path, pdf_path, title in DOCUMENTS:
        if os.path.exists(md_path):
            try:
                build_pdf_from_markdown(md_path, pdf_path, title)
            except Exception as e:
                print(f"❌ Error converting {os.path.basename(md_path)}: {e}")
        else:
            print(f"⚠️ File not found: {md_path}")
    print("\n🎉 All PDF documents generated successfully!")

if __name__ == "__main__":
    main()
