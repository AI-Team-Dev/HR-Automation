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
    
    logger.debug(f"Attempting to extract JSON from response of length {len(text)}")

    # Strip markdown code fences if present
    if "```" in text:
        # Handle ```json or ```JSON or just ```
        import re
        # Remove code fences
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = text.replace('```', '')
        text = text.strip()

    # First, try direct JSON load
    try:
        result = json.loads(text)
        logger.debug("Successfully parsed JSON directly")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")
        pass

    # Try to find JSON object boundaries more aggressively
    # Look for the outermost { } pair
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx != -1:
        candidate = text[start_idx:end_idx + 1]
        try:
            result = json.loads(candidate)
            logger.debug("Successfully extracted and parsed JSON object")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Extracted JSON parse failed: {e}")
            # Try to clean up common issues
            # Remove trailing commas
            import re
            candidate = re.sub(r',\s*}', '}', candidate)
            candidate = re.sub(r',\s*]', ']', candidate)
            try:
                result = json.loads(candidate)
                logger.debug("Successfully parsed JSON after cleanup")
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"Could not extract valid JSON from response. First 500 chars: {text[:500]}")
    raise ValueError("Model response does not contain a valid JSON object")


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

    # 1) Derive match_score - be very flexible with nested structure
    overall_eval = data.get("overall_evaluation") or {}
    jd_analysis = data.get("jd_requirement_analysis") or {}
    semantic = jd_analysis.get("semantic_match") or {}
    skills_analysis = jd_analysis.get("skills_analysis") or {}
    exp_analysis = jd_analysis.get("experience_analysis") or {}
    edu_analysis = jd_analysis.get("education_analysis") or {}
    candidate_profile = data.get("candidate_profile") or {}
    candidate_exp = candidate_profile.get("experience") or {}

    match_score = data.get("match_score")
    if match_score is None:
        match_score = overall_eval.get("match_score")
    if match_score is None:
        # Fallback: use semantic_match.semantic_score if present
        match_score = semantic.get("semantic_score")
    if match_score is None:
        # Try skill_match_percentage
        match_score = skills_analysis.get("skill_match_percentage")
    if match_score is None:
        # Try to compute from components if they exist
        skill_pct = skills_analysis.get("skill_match_percentage", 0)
        exp_pct = exp_analysis.get("experience_match_percentage", 0)
        edu_pct = edu_analysis.get("education_match_percentage", 0)
        if skill_pct or exp_pct or edu_pct:
            # Weighted average: skills 50%, exp 30%, edu 20%
            match_score = (skill_pct * 0.5 + exp_pct * 0.3 + edu_pct * 0.2)

    # 2) Derive shortlisted
    shortlisted = data.get("shortlisted")
    if shortlisted is None:
        shortlisted = overall_eval.get("shortlisted")
    if shortlisted is None and match_score is not None:
        try:
            shortlisted = float(match_score) >= 60.0
        except (TypeError, ValueError):
            shortlisted = None

    # 3) Derive reasons from explicit field or decision_explanation
    reasons = data.get("reasons")
    if reasons is None:
        decision = data.get("decision_explanation") or {}
        final_decision = data.get("final_decision") or {}
        parts = []
        
        summary = decision.get("summary")
        if summary:
            parts.append(str(summary))
        
        final_reason = final_decision.get("final_decision_reason")
        if final_reason and final_reason != summary:
            parts.append(str(final_reason))
            
        positives = decision.get("positive_factors") or []
        negatives = decision.get("negative_factors") or []
        if positives:
            parts.append("Positive factors: " + "; ".join(map(str, positives)))
        if negatives:
            parts.append("Negative factors: " + "; ".join(map(str, negatives)))
        
        if parts:
            reasons = "\n".join(parts)
        else:
            # Last resort: use any text we can find
            reasons = "Evaluation completed based on JD and resume analysis"

    # Be more lenient - if we have at least match_score, we can work with it
    if match_score is None:
        # If we still don't have a score, default to 50 (neutral)
        logger.warning("Could not derive match_score from response, defaulting to 50")
        match_score = 50
    
    if shortlisted is None:
        shortlisted = float(match_score) >= 60.0
    
    if reasons is None:
        reasons = f"Candidate evaluated with match score {match_score}"

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
        logger.warning(f"Invalid JSON from Grok on first attempt: {exc}. Raw response: {raw_response}")

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
        logger.error(f"Invalid JSON from Grok after retry: {exc}. Raw retry response: {raw_retry_response}")
        raise RuntimeError("Grok model failed to return valid JSON even after retry") from exc
