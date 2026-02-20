"""API routes for Instruction Manual Builder."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from shared.data_loaders import get_available_components, load_wireframe_data

router = APIRouter(prefix="/api")

# Resolve data dir from the package root (instruction_manual_generator/), not routes/
_PACKAGE_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


@router.get("/assemblies")
async def list_assemblies():
    """List available assembly configuration files."""
    assemblies = []
    for f in sorted(DATA_DIR.glob("fmb_assembly*.json")):
        assemblies.append({
            "name": f.stem,
            "filename": f.name,
        })
    return assemblies


@router.get("/assembly/{name}")
async def get_assembly(name: str):
    """Load a specific assembly configuration."""
    assembly_file = DATA_DIR / f"{name}.json"
    if not assembly_file.exists():
        raise HTTPException(status_code=404, detail=f"Assembly '{name}' not found")

    with open(assembly_file, "r") as f:
        return json.load(f)


@router.get("/wireframe/{component_name}")
async def get_wireframe(component_name: str):
    """Load wireframe data for a component."""
    data = load_wireframe_data(DATA_DIR, component_name)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Wireframe data for '{component_name}' not found",
        )
    return data


@router.get("/components")
async def list_components():
    """List all available components with wireframe data."""
    return get_available_components(DATA_DIR)


@router.get("/model/{component_name}")
async def get_model(component_name: str):
    """Serve the OBJ CAD file for a component."""
    models_dir = DATA_DIR / "models"
    obj_file = models_dir / f"{component_name}.obj"
    if not obj_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file for '{component_name}' not found",
        )
    return FileResponse(
        obj_file,
        media_type="text/plain",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@router.post("/export-pdf")
async def export_pdf(data: dict):
    """Generate a PDF instruction manual from rendered images.

    Expects JSON body:
    {
        "title": "FMB Assembly 1",
        "components": [{"name": "base", "count": 1, "image_b64": "..."}],
        "steps": [{"description": "Step 1: ...", "image_b64": "..."}]
    }
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    )
    from reportlab.lib import colors

    title = data.get("title", "Assembly Manual")
    components = data.get("components", [])
    steps = data.get("steps", [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ManualTitle",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=8 * mm,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=4 * mm,
        spaceBefore=4 * mm,
    )
    step_desc_style = ParagraphStyle(
        "StepDesc",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=2 * mm,
        alignment=TA_CENTER,
    )
    comp_label_style = ParagraphStyle(
        "CompLabel",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
    )

    story = []

    # Title
    story.append(Paragraph(title, title_style))

    # Components section
    if components:
        story.append(Paragraph("Components:", section_style))

        # Build component grid (up to 4 per row)
        comp_cells = []
        for comp in components:
            img_data = base64.b64decode(comp["image_b64"])
            img = Image(io.BytesIO(img_data), width=35 * mm, height=35 * mm)
            label = Paragraph(
                f"{comp['name']}<br/>x{comp['count']}", comp_label_style
            )
            comp_cells.append([img, label])

        # Arrange in rows of 4
        rows = []
        for i in range(0, len(comp_cells), 4):
            row_imgs = []
            row_labels = []
            for j in range(4):
                if i + j < len(comp_cells):
                    row_imgs.append(comp_cells[i + j][0])
                    row_labels.append(comp_cells[i + j][1])
                else:
                    row_imgs.append("")
                    row_labels.append("")
            rows.append(row_imgs)
            rows.append(row_labels)

        page_width = A4[0] - 30 * mm
        col_width = page_width / 4
        comp_table = Table(rows, colWidths=[col_width] * 4)
        comp_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 6 * mm))

    # Steps section - 2 columns
    if steps:
        step_rows = []
        page_width = A4[0] - 30 * mm
        img_width = (page_width / 2) - 4 * mm
        img_height = img_width * 0.75

        for i in range(0, len(steps), 2):
            desc_row = []
            img_row = []

            for j in range(2):
                if i + j < len(steps):
                    step = steps[i + j]
                    desc_row.append(
                        Paragraph(step["description"], step_desc_style)
                    )
                    img_data = base64.b64decode(step["image_b64"])
                    img = Image(
                        io.BytesIO(img_data),
                        width=img_width,
                        height=img_height,
                    )
                    img_row.append(img)
                else:
                    desc_row.append("")
                    img_row.append("")

            step_rows.append(desc_row)
            step_rows.append(img_row)

        col_width = page_width / 2
        step_table = Table(step_rows, colWidths=[col_width] * 2)
        step_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ]))
        story.append(step_table)

    doc.build(story)
    buffer.seek(0)

    filename = f"{title.replace(' ', '_')}_manual.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/save-manual")
async def save_manual(config: dict):
    """Save manual configuration to data/manuals/ directory."""
    manuals_dir = DATA_DIR / "manuals"
    manuals_dir.mkdir(exist_ok=True)

    name = config.get("title", "untitled").replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"

    filepath = manuals_dir / filename
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    return {"success": True, "filename": filename}


@router.get("/manuals")
async def list_manuals():
    """List saved manual configurations."""
    manuals_dir = DATA_DIR / "manuals"
    if not manuals_dir.exists():
        return []

    manuals = []
    for f in sorted(manuals_dir.glob("*.json"), reverse=True):
        manuals.append({
            "name": f.stem,
            "filename": f.name,
        })
    return manuals


@router.get("/manual/{name}")
async def get_manual(name: str):
    """Load a saved manual configuration."""
    manuals_dir = DATA_DIR / "manuals"
    manual_file = manuals_dir / f"{name}.json"
    if not manual_file.exists():
        raise HTTPException(status_code=404, detail=f"Manual '{name}' not found")

    with open(manual_file, "r") as f:
        return json.load(f)
