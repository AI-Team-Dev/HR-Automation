"""LLM-powered scoring utilities using Grok 4 Fast."""

from __future__ import annotations

import json
from typing import Any, Dict

from loguru import logger

from .grok_client import GrokClient
from .prompts import build_scoring_prompt


def _extract_json_object(raw_response: str) -> Any:
    """Best-effort extraction of a JSON value from a model response string.

    Handles common cases such as markdown code fences and extra prose by
    locating the first "{" and the last "}" and attempting to parse the
    substring as JSON.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove the initial ```[lang]? line
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

    # First, try direct JSON load
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try to grab the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:  # pragma: no cover - best effort
            raise ValueError("Model response is not valid JSON") from exc

    raise ValueError("Model response does not contain a JSON object")


def _parse_scoring_response(raw_response: str) -> Dict[str, Any]:
    """Parse the Grok scoring response and normalise core fields.

    The LLM is allowed to return a rich JSON object with many fields
    (candidate details, skills/education/experience analysis, summary,
    recommendations, etc.). This function:

    - Ensures the top-level JSON is an object.
    - Validates the presence of ``match_score``, ``shortlisted``, and
      ``reasons``.
    - Normalises these three core fields and writes them back into the
      original dict.
    - Returns the full enriched dict so callers get all additional
      information produced by the model.
    """
    try:
        data = _extract_json_object(raw_response)
    except ValueError as exc:
        raise ValueError("Model response is not valid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("Model response JSON must be an object")

    # Allow match_score/shortlisted either at top level or in overall_evaluation
    overall_eval = data.get("overall_evaluation") or {}

    match_score = data.get("match_score", overall_eval.get("match_score"))
    shortlisted = data.get("shortlisted", overall_eval.get("shortlisted"))
    reasons = data.get("reasons")

    # Derive reasons from decision_explanation if not explicitly provided
    if reasons is None:
        decision = data.get("decision_explanation") or {}
        parts = []
        summary = decision.get("summary")
        if summary:
            parts.append(str(summary))
        positives = decision.get("positive_factors") or []
        negatives = decision.get("negative_factors") or []
        if positives:
            parts.append("Positive factors: " + "; ".join(map(str, positives)))
        if negatives:
            parts.append("Negative factors: " + "; ".join(map(str, negatives)))
        if parts:
            reasons = "\n".join(parts)

    if match_score is None or shortlisted is None or reasons is None:
        raise ValueError("Model response JSON must contain match_score, shortlisted, and reasons fields")

    try:
        match_score_num = float(match_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("match_score must be a number") from exc

    if isinstance(reasons, list):
        reasons_str = "\n".join(str(r) for r in reasons)
    else:
        reasons_str = str(reasons)

    # Normalise core fields but keep everything else the model returned.
    data["match_score"] = max(0.0, min(100.0, match_score_num))
    data["shortlisted"] = bool(shortlisted)
    data["reasons"] = reasons_str

    return data


def score_candidate(cv_text: str, jd_text: str) -> Dict[str, Any]:
    """Score a candidate resume against a job description using Grok 4 Fast.

    This function builds an instruction-heavy prompt, sends it to Grok, parses
    the JSON response, and returns a normalized dictionary with the keys:
    ``match_score``, ``shortlisted``, and ``reasons``.

    If the initial response is not valid JSON, the function will perform a
    single retry with a stronger JSON-enforcement instruction.
    """
    client = GrokClient()

    # First attempt
    prompt = build_scoring_prompt(cv_text=cv_text, jd_text=jd_text)
    raw_response = client.run(prompt)

    try:
        return _parse_scoring_response(raw_response)
    except ValueError as exc:
        logger.warning(f"Invalid JSON from Grok on first attempt: {exc}")

    # Retry once with explicit JSON enforcement
    retry_prompt = (
        build_scoring_prompt(cv_text=cv_text, jd_text=jd_text)
        + "\n\nYour previous response did not comply with the JSON-only requirement. "
        "Now respond again with ONLY a single valid JSON object following the "
        "specified schema. Do not include any extra text."
    )
    raw_retry_response = client.run(retry_prompt)

    try:
        return _parse_scoring_response(raw_retry_response)
    except ValueError as exc:
        logger.error(f"Invalid JSON from Grok after retry: {exc}")
        raise RuntimeError("Grok model failed to return valid JSON even after retry") from exc
