"""Self-contained offline HTML report builder (Part E)."""

import base64
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chebnet_kaist.constants import MODEL_LABELS
from chebnet_kaist.evaluation.plots.style import DATASET_NAME


def _embed_image(path: Path) -> str:
    """Return a base64 data URI for a PNG file."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _figure_block(path: Path, caption: str) -> str:
    """Build an HTML figure block with embedded image and caption."""
    if not path.exists():
        return f'<p class="missing">Figure not available: {html.escape(str(path.name))}</p>'
    uri = _embed_image(path)
    return f"""
<figure>
  <img src="{uri}" alt="{html.escape(caption)}" loading="lazy"/>
  <figcaption>{html.escape(caption)}</figcaption>
</figure>"""


def generate_key_findings(rows: list[dict[str, Any]], model_name: str) -> list[str]:
    """Auto-generate 3-5 bullet points from metrics."""
    bullets: list[str] = []
    class_rows = [r for r in rows if r["class"] not in ("Macro Avg", "Weighted Avg")]

    macro = next((r for r in rows if r["class"] == "Macro Avg"), None)
    if macro:
        bullets.append(f"Macro F1 of {macro['f1']:.4f} on the validation split.")
        bullets.append(f"Macro ROC-AUC of {macro['roc_auc']:.4f}.")

    best = max(class_rows, key=lambda r: r["f1"])
    worst = min(class_rows, key=lambda r: r["f1"])
    bullets.append(f"Strongest class: {best['class']} (F1={best['f1']:.4f}).")
    bullets.append(
        f"Weakest class: {worst['class']} (F1={worst['f1']:.4f}, recall={worst['recall']:.4f})."
    )

    normal = next((r for r in class_rows if r["class"] == "Normal"), None)
    if normal and normal["recall"] < 0.95:
        bullets.append(f"Normal class recall is {normal['recall']:.4f} — monitor false negatives.")

    label = MODEL_LABELS.get(model_name, model_name.upper())
    bullets.insert(0, f"{label} evaluated on {DATASET_NAME} (graph-level classification).")
    return bullets[:5]


def build_html_report(
    model_name: str,
    output_path: Path,
    metrics_html_path: Path,
    figure_sections: list[tuple[str, list[tuple[Path, str]]]],
    key_findings: list[str],
    config_summary: dict[str, Any],
) -> None:
    """Write a fully self-contained HTML evaluation report.

    Args:
        model_name: Model identifier (gatv2 or chebnet).
        output_path: Destination HTML path.
        metrics_html_path: Path to the styled metrics table HTML.
        figure_sections: List of ``(section_title, [(png_path, caption), ...])``.
        key_findings: Auto-generated bullet points.
        config_summary: Hyperparameter summary dict.
    """
    label = MODEL_LABELS.get(model_name, model_name.upper())
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metrics_table = ""
    if metrics_html_path.exists():
        # Extract table body from metrics HTML
        content = metrics_html_path.read_text(encoding="utf-8")
        start = content.find("<table>")
        end = content.find("</table>") + len("</table>")
        if start >= 0 and end > start:
            metrics_table = content[start:end]

    findings_html = "".join(f"<li>{html.escape(b)}</li>" for b in key_findings)

    config_lines = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in config_summary.items()
    )

    sections_html = ""
    for title, figures in figure_sections:
        figs = "".join(_figure_block(p, cap) for p, cap in figures)
        sections_html += f"""
<section>
  <h2>{html.escape(title)}</h2>
  <div class="figures">{figs}</div>
</section>"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(label)} — Evaluation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Georgia,serif;background:#fff;color:#1e293b;
      max-width:960px;margin:0 auto;padding:32px 24px;line-height:1.6}}
header{{border-bottom:2px solid #e2e8f0;padding-bottom:20px;margin-bottom:28px}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:6px}}
.meta{{color:#64748b;font-size:0.9em}}
h2{{font-size:1.15em;margin:28px 0 14px;color:#0f172a;border-left:4px solid #2563eb;padding-left:10px}}
ul.findings{{margin:12px 0 12px 20px}}
ul.findings li{{margin-bottom:6px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #e2e8f0;padding:8px 12px;text-align:center;font-size:0.9em}}
th{{background:#f1f5f9}}
.config td:first-child{{font-weight:600;text-align:left;background:#f8fafc}}
figure{{margin:16px 0;text-align:center}}
figure img{{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:4px}}
figcaption{{font-size:0.85em;color:#64748b;margin-top:8px;font-style:italic}}
.missing{{color:#94a3b8;font-style:italic}}
.figures{{display:flex;flex-direction:column;gap:8px}}
</style>
</head><body>
<header>
  <h1>{html.escape(label)} — Evaluation Report</h1>
  <p class="meta">Dataset: {DATASET_NAME} &nbsp;|&nbsp; Date: {date_str}</p>
</header>

<section>
  <h2>Configuration</h2>
  <table class="config">{config_lines}</table>
</section>

<section>
  <h2>Key Findings</h2>
  <ul class="findings">{findings_html}</ul>
</section>

<section>
  <h2>Performance Summary</h2>
  {metrics_table}
</section>

{sections_html}

<footer style="margin-top:40px;padding-top:16px;border-top:1px solid #e2e8f0;
  color:#94a3b8;font-size:0.8em;text-align:center">
  Generated by gnn-canids evaluation pipeline. Self-contained report — no external dependencies.
</footer>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
