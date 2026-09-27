"""Evaluate the trained model on the held-out eval_set (never seen during training).

Usage (from the project root):  python -m scripts.evaluate

Writes reports/eval.json (machine-readable) and reports/eval.md (for the README):
accuracy, per-class precision/recall, confusion matrix, and accuracy-vs-coverage
at several confidence thresholds.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.model import ResNetClassifier
from app.preprocess import preprocess

EVAL_DIR = Path("data/raw/eval_set")
LABELS_CSV = Path("data/raw/eval_labels.csv")
MANIFEST = Path("models/manifest.json")
REPORTS_DIR = Path("reports")
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def load_eval_rows() -> list[dict]:
    with open(LABELS_CSV) as f:
        return list(csv.DictReader(f))  # columns: filename, true_label


def main():
    clf = ResNetClassifier(MANIFEST)
    rows = load_eval_rows()

    # Run all eval tiles through the model in one batch (210 small tiles fit in memory).
    batch = torch.stack([preprocess(Image.open(EVAL_DIR / r["filename"])) for r in rows])
    probs = clf.predict(batch)

    report = build_report(rows, probs, clf.classes, {
        "model_name": clf.model_name,
        "model_version": clf.model_version,
        "weights_sha256": clf.weights_sha,
        "preprocess_version": clf.preprocess_version,
    })
    write_report(report, "eval")


def build_report(rows: list[dict], probs: np.ndarray, classes: list[str], model_info: dict) -> dict:
    """Metrics for any model: rows are the eval CSV rows, probs[i] is the distribution for rows[i]."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([class_to_idx[r["true_label"]] for r in rows])
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)  # top-1 confidence
    n = len(classes)

    # Confusion matrix: rows = true class, columns = predicted class.
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float((y_true == y_pred).mean())
    per_class, f1s, supports = {}, [], []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        predicted_as_i = cm[:, i].sum()
        actually_i = cm[i, :].sum()
        precision = tp / predicted_as_i if predicted_as_i else 0.0  # never predicted -> 0
        recall = tp / actually_i if actually_i else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        supports.append(actually_i)
        per_class[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(actually_i),
        }

    # Macro-F1: plain mean of per-class F1, so every class counts equally (a bad class
    # drags it down even if it's rare). Weighted-F1: mean weighted by class size.
    # Micro-F1: pools all tiles; with exactly one label and one prediction per tile,
    # micro precision = micro recall = accuracy, so micro-F1 always equals accuracy here.
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports))
    micro_f1 = accuracy

    # Accuracy vs coverage: if we only auto-accept predictions with conf >= t,
    # what fraction do we accept (coverage), and how accurate are those?
    coverage_table = []
    for t in THRESHOLDS:
        accepted = conf >= t
        coverage_table.append({
            "threshold": t,
            "coverage": round(float(accepted.mean()), 4),
            "accuracy_on_accepted": round(float((y_true[accepted] == y_pred[accepted]).mean()), 4)
            if accepted.any() else None,
            "sent_to_review": int((~accepted).sum()),
        })

    # The worst mistakes: wrong but confident. These are what a threshold can't catch.
    confident_errors = sorted(
        [
            {"filename": r["filename"], "true": r["true_label"], "pred": classes[p], "conf": round(float(c), 4)}
            for r, p, t, c in zip(rows, y_pred, y_true, conf) if p != t
        ],
        key=lambda e: -e["conf"],
    )

    return {
        **model_info,
        "n_tiles": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {"classes": classes, "rows_true_cols_pred": cm.tolist()},
        "accuracy_vs_coverage": coverage_table,
        "errors": confident_errors,
    }


def write_report(report: dict, name: str) -> None:
    """Write reports/<name>.json and reports/<name>.md, and print the markdown."""
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (REPORTS_DIR / f"{name}.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    classes = r["confusion_matrix"]["classes"]
    lines = [
        f"# Eval report: {r['model_name']} {r['model_version']}",
        "",
        f"Held-out eval_set: {r['n_tiles']} tiles. **Accuracy: {r['accuracy']:.2%}**",
        "",
        f"Macro-F1: {r['macro_f1']:.4f} | Weighted-F1: {r['weighted_f1']:.4f} | "
        f"Micro-F1: {r['micro_f1']:.4f} (= accuracy for single-label classification)",
        "",
        "## Per-class",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for c, m in r["per_class"].items():
        lines.append(f"| {c} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")

    lines += ["", "## Confusion matrix (rows = true, cols = predicted)",
              "| true \\ pred | " + " | ".join(classes) + " |",
              "|---" * (len(classes) + 1) + "|"]
    for name, row in zip(classes, r["confusion_matrix"]["rows_true_cols_pred"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")

    lines += ["", "## Accuracy vs coverage",
              "| threshold | coverage | accuracy on accepted | sent to review |",
              "|---|---|---|---|"]
    for row in r["accuracy_vs_coverage"]:
        lines.append(f"| {row['threshold']} | {row['coverage']:.2%} | {row['accuracy_on_accepted']} | {row['sent_to_review']} |")

    lines += ["", f"## Errors ({len(r['errors'])}), most confident first",
              "| file | true | predicted | conf |", "|---|---|---|---|"]
    for e in r["errors"]:
        lines.append(f"| {e['filename']} | {e['true']} | {e['pred']} | {e['conf']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
