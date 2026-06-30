"""Rule reward for structured T cell reactivity classification.

reward separates the two:

    score = format_score + correctness_score

  - format_score   = FORMAT_BONUS if (has_output_tag and has_reason) else 0
                     (independent of whether the label is right)
  - correctness_score = CORRECT_BASE * CLASS_WEIGHTS[expected] if correct else 0

This way the format gradient is sourced from ~100% of well-formatted samples
regardless of which label appears

FORMAT_BONUS is set below CORRECT_BASE * min(CLASS_WEIGHTS) so that "good
format + wrong label" never out-earns "right label".
"""

from __future__ import annotations

import re


LABELS = ("Reactive", "Non-Reactive")
LABEL_RE = re.compile(r"\b(Non-Reactive|Reactive)\b", re.IGNORECASE)
OUTPUT_RE = re.compile(r"<output>\s*(Non-Reactive|Reactive)\s*</output>", re.IGNORECASE | re.DOTALL)
REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(
    r"(?:final\s+(?:answer|label)|label|classification|prediction)\s*[:：-]\s*(Non-Reactive|Reactive)\b",
    re.IGNORECASE,
)

CORRECT_BASE = 1.0
CLASS_WEIGHTS = {"Reactive": 1.0, "Non-Reactive": 1.0}
FORMAT_BONUS = 0.3


def _normalize_label(text: str | None) -> str | None:
    text = (text or "").strip()
    for label in LABELS:
        if text.lower() == label.lower():
            return label
    return None


def _extract_label(solution_str: str) -> tuple[str | None, str, int]:
    solution_str = solution_str or ""
    output_matches = OUTPUT_RE.findall(solution_str)
    if output_matches:
        return _normalize_label(output_matches[-1]), "output_tag", len(output_matches)

    final_matches = FINAL_RE.findall(solution_str)
    if final_matches:
        return _normalize_label(final_matches[-1]), "final_label", len(final_matches)

    matches = LABEL_RE.findall(solution_str)
    if not matches:
        return None, "unparseable", 0
    return _normalize_label(matches[-1]), "last_label_mention", len(matches)


def _has_valid_reason(solution_str: str) -> bool:
    match = REASON_RE.search(solution_str or "")
    return bool(match and match.group(1).strip())


def _confusion_indicators(expected: str | None, predicted: str | None) -> dict[str, float]:
    cells = {"cm_R_R": 0.0, "cm_R_N": 0.0, "cm_N_R": 0.0, "cm_N_N": 0.0}
    if expected not in LABELS or predicted not in LABELS:
        return cells
    g = "R" if expected == "Reactive" else "N"
    p = "R" if predicted == "Reactive" else "N"
    cells[f"cm_{g}_{p}"] = 1.0
    return cells


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info
    expected = _normalize_label(ground_truth)
    predicted, parse_source, label_count = _extract_label(solution_str)
    has_output_tag = parse_source == "output_tag"
    has_reason = _has_valid_reason(solution_str)
    is_correct = (
        expected is not None and predicted is not None and predicted == expected
    )
    confusion = _confusion_indicators(expected, predicted)

    format_score = FORMAT_BONUS if (has_output_tag and has_reason) else 0.0

    if is_correct:
        correctness_score = CORRECT_BASE * CLASS_WEIGHTS[expected]
    else:
        correctness_score = 0.0

    if expected is None:
        format_score = 0.0
        correctness_score = 0.0

    score = format_score + correctness_score

    result = {
        "score": score,
        "format_score": format_score,
        "correctness_score": correctness_score,
        "predicted": predicted or "",
        "expected": expected or "",
        "parse_source": parse_source,
        "label_count": label_count,
        "has_reason": has_reason,
        "has_output_tag": has_output_tag,
        "is_correct": float(is_correct),
    }
    result.update(confusion)
    return result
