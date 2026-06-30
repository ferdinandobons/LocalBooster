"""Lightweight benchmark scoring helpers.

The scorers are intentionally conservative. They are good enough for local benchmark slices and
avoid depending on upstream benchmark graders or heavyweight symbolic packages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ScoreResult:
    grader: str
    expected: str
    prediction: str
    correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "grader": self.grader,
            "expected": self.expected,
            "prediction": self.prediction,
            "correct": self.correct,
        }


def score_response(response: str, expected: str, grader: str = "auto") -> ScoreResult:
    """Score a model response against a benchmark answer."""

    grader = infer_grader(expected) if grader == "auto" else grader
    prediction = extract_prediction(response)
    if grader == "choice":
        correct = grade_choice(prediction, expected)
    elif grader == "contains":
        correct = normalize_text(expected) in normalize_text(response)
    elif grader == "math":
        correct = grade_math(prediction, expected)
    elif grader == "exact":
        correct = normalize_text(prediction) == normalize_text(expected)
    else:
        raise ValueError(f"unknown grader {grader!r}")
    return ScoreResult(
        grader=grader,
        expected=expected,
        prediction=prediction,
        correct=correct,
    )


def infer_grader(expected: str) -> str:
    stripped = expected.strip()
    if re.fullmatch(r"[A-Ea-e]", stripped):
        return "choice"
    if re.search(r"\\frac|\\pi|\^|[0-9]", stripped):
        return "math"
    return "exact"


def extract_prediction(response: str) -> str:
    """Extract the most likely final answer span from a response."""

    response = strip_markup_lines(response.strip())
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    patterns = [
        r"(?<![</])(?:final\s+answer|answer|therefore|thus)\s*(?:is|:)?\s*(.+)",
        r"####\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return clean_prediction(matches[-1])

    nonempty_lines = [line.strip() for line in response.splitlines() if line.strip()]
    if nonempty_lines:
        return clean_prediction(nonempty_lines[-1])
    return response


def strip_markup_lines(response: str) -> str:
    lines = []
    for line in response.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"</?[A-Za-z][A-Za-z0-9_-]*>", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    index = start + len(marker)
    depth = 1
    chars = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        index += 1
    return clean_prediction("".join(chars)) if chars else None


def clean_prediction(value: str) -> str:
    value = value.strip()
    value = value.split("\n", 1)[0].strip()
    value = re.split(r"(?i)\s+(?:because|since|where)\s+", value, maxsplit=1)[0]
    return value.strip(" .,:;")


def grade_choice(prediction: str, expected: str) -> bool:
    match = re.search(r"\b([A-Ea-e])\b", prediction)
    return bool(match and match.group(1).upper() == expected.strip().upper())


def grade_math(prediction: str, expected: str) -> bool:
    prediction_norm = normalize_math(prediction)
    expected_norm = normalize_math(expected)
    if prediction_norm == expected_norm:
        return True

    prediction_number = parse_number(prediction_norm)
    expected_number = parse_number(expected_norm)
    if prediction_number is not None and expected_number is not None:
        return prediction_number == expected_number

    return False


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_math(value: str) -> str:
    value = value.strip()
    value = value.replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "")
    value = value.replace("\\!", "")
    value = value.replace("\\cdot", "*")
    value = value.replace("\\times", "*")
    value = value.replace("\\pi", "pi")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = replace_latex_fractions(value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", "", value)
    value = value.strip(" .,:;")
    return value.lower()


def replace_latex_fractions(value: str) -> str:
    fraction_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = fraction_pattern.sub(r"\1/\2", value)
    return value


def parse_number(value: str) -> Fraction | None:
    value = value.replace(",", "")
    try:
        return Fraction(value)
    except ValueError:
        return None
