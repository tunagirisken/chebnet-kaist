"""Summary metrics tables: CSV, HTML, and JSON (Part A5)."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from chebnet_kaist.constants import CLASS_NAMES
from sklearn.metrics import classification_report, roc_auc_score

COLUMNS = ["Precision", "Recall", "F1", "Support", "ROC-AUC"]


def build_summary_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    """Build per-class and aggregate metric rows.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        class_names: Optional class labels (default: 5-class KAIST names).
        num_classes: Optional class count (default: len(class_names)).

    Returns:
        List of row dicts ready for export.
    """
    names = class_names or CLASS_NAMES
    n_cls = num_classes or len(names)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    report = classification_report(
        y_true_arr,
        y_pred,
        labels=list(range(n_cls)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    y_bin = np.column_stack([(y_true_arr == idx).astype(int) for idx in range(n_cls)])

    rows: list[dict[str, Any]] = []
    for name in names:
        idx = names.index(name)
        try:
            auc = roc_auc_score(y_bin[:, idx], y_prob_arr[:, idx])
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
                "roc_auc": auc,
            }
        )

    for agg in ("macro avg", "weighted avg"):
        label = "Macro Avg" if agg == "macro avg" else "Weighted Avg"
        try:
            auc = roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average=agg.split()[0])
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "class": label,
                "precision": report[agg]["precision"],
                "recall": report[agg]["recall"],
                "f1": report[agg]["f1-score"],
                "support": int(report[agg]["support"]),
                "roc_auc": auc,
            }
        )
    return rows


def _cell_color(value: float) -> str:
    """Return background color for HTML table cell."""
    if np.isnan(value):
        return "#f8fafc"
    if value >= 0.90:
        return "#dcfce7"
    if value >= 0.80:
        return "#fef9c3"
    return "#fee2e2"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write plain CSV for thesis tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class"] + COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["class"],
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    row["support"],
                    f"{row['roc_auc']:.4f}" if not np.isnan(row["roc_auc"]) else "",
                ]
            )


def write_json(rows: list[dict[str, Any]], path: Path, extra: dict | None = None) -> None:
    """Write metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": rows}
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_html(rows: list[dict[str, Any]], path: Path, model_name: str) -> None:
    """Write color-coded HTML metrics table."""
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = ["precision", "recall", "f1", "roc_auc"]
    best: dict[str, float] = {}
    for col in metric_cols:
        vals = [r[col] for r in rows if r["class"] in CLASS_NAMES and not np.isnan(r[col])]
        best[col] = max(vals) if vals else 0.0

    html_rows = []
    for row in rows:
        cells = [f"<td><strong>{row['class']}</strong></td>"]
        for col, label in zip(metric_cols, ["Precision", "Recall", "F1", "ROC-AUC"]):
            val = row[col]
            bold = (
                "font-weight:700;"
                if (row["class"] in CLASS_NAMES and not np.isnan(val) and val == best.get(col))
                else ""
            )
            bg = _cell_color(val) if col != "support" else "#ffffff"
            text = f"{val:.4f}" if not np.isnan(val) else "N/A"
            cells.append(f'<td style="background:{bg};{bold}">{text}</td>')
        cells.append(f"<td>{row['support']:,}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{model_name} Metrics</title>
<style>
body{{font-family:Georgia,serif;margin:40px;background:#fff;color:#1e293b}}
h1{{font-size:1.4em;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}
th,td{{border:1px solid #e2e8f0;padding:10px 14px;text-align:center}}
th{{background:#f1f5f9;font-size:0.85em;text-transform:uppercase;letter-spacing:0.05em}}
</style></head><body>
<h1>{model_name.upper()} — Classification Metrics</h1>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Support</th></tr>
{"".join(html_rows)}
</table>
<p style="margin-top:16px;font-size:0.9em;color:#64748b">
Green &ge; 0.90 &nbsp;|&nbsp; Yellow &ge; 0.80 &nbsp;|&nbsp; Red &lt; 0.80
</p>
</body></html>"""

    path.write_text(html, encoding="utf-8")
