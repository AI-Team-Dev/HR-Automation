"""AI Engine for HR Automation System - Strict Minimal JSON Output.

This module provides the core AI engine for extracting structured candidate
information from resumes and comparing them with job descriptions.
Outputs a strict minimal JSON for shortlisting decisions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from loguru import logger

from .grok_client import GrokClient
from .date_utils import get_current_date_str


def build_minimal_scoring_prompt(cv_text: str, jd_text: str) -> str:
    """Build a prompt for minimal JSON output focused on shortlisting.
    
    Args:
        cv_text: The candidate's resume text
        jd_text: The job description text
    
    Returns:
        The formatted prompt string for the LLM
    """
    date_str = get_current_date_str()
    
    return (
        "You are an AI engine inside an HR Automation System.\n"
        f"IMPORTANT: Today's date is {date_str}. Use this date for all experience calculations.\n"
        "When you see 'Present', 'Current', 'Till Date', or similar terms in employment dates, "
        f"calculate the duration up to {date_str}.\n\n"
        "Your job is to extract structured candidate information from resumes, "
        "compare it with a given Job Description, and output a STRICT minimal JSON "
        "containing only what is required for shortlisting decisions.\n\n"
        
        "Follow these rules:\n"
        "1. Output ONLY valid JSON. No markdown. No explanation. No extra text.\n"
        "2. Use the EXACT JSON schema provided below. Do not add or rename fields.\n"
        "3. All arrays must exist, even if empty.\n"
        "4. BE CONCISE - Keep all text fields short and precise.\n"
        "5. Extract ONLY actual skills mentioned in resume (not courses/certifications).\n"
        "6. List certifications/courses separately in certifications array.\n"
        "7. Evaluate skills matching with JD: matched_skills, missing_skills, mandatory_skill_gaps.\n"
        "8. The shortlisted field is true ONLY if match_score >= threshold.\n\n"
        
        "This is the ONLY JSON structure you are allowed to output:\n"
        "{\n"
        '  "candidate_details": {\n'
        '    "name": "",\n'
        '    "email": "",\n'
        '    "phone": "",\n'
        '    "location": "",\n'
        '    "linkedin": "",\n'
        '    "other_urls": []\n'
        '  },\n'
        '  "education": [\n'
        '    {\n'
        '      "degree": "",\n'
        '      "year": "",\n'
        '      "institution": ""\n'
        '    }\n'
        '  ],\n'
        '  "experience": {\n'
        '    "total_years": "",\n'
        '    "relevant_experience_summary": ""\n'
        '  },\n'
        '  "experience_timeline": [\n'
        '    {\n'
        '      "role": "",\n'
        '      "company": "",\n'
        '      "start_date": "",\n'
        '      "end_date": "",\n'
        '      "normalized_start_date": "",\n'
        '      "normalized_end_date": ""\n'
        '    }\n'
        '  ],\n'
        '  "total_employment_gap": {\n'
        '    "years": 0,\n'
        '    "months": 0,\n'
        '    "summary": ""\n'
        '  },\n'
        '  "all_skills": [],\n'
        '  "certifications": [],\n'
        '  "skills_evaluation": {\n'
        '    "matched_skills": [],\n'
        '    "missing_skills": [],\n'
        '    "mandatory_skill_gaps": []\n'
        '  },\n'
        '  "match_result": {\n'
        '    "match_score": 0,\n'
        '    "shortlisted": false,\n'
        '    "threshold": 60,\n'
        '    "reason_for_decision": ""\n'
        '  },\n'
        '  "suggestions": ""\n'
        "}\n\n"
        
        "EXTRACTION AND SCORING RULES:\n"
        "- Extract ONLY technical and soft skills into all_skills (NO certifications/courses)\n"
        "- Extract certifications/courses separately into certifications array\n"
        "- Keep all_skills concise - only actual skills mentioned in resume\n"
        "- Compare with JD to identify matched_skills, missing_skills, mandatory_skill_gaps\n"
        "- Skills Match: 50% weight (mandatory skills have highest impact)\n"
        "- Experience Match: 30% weight (compare required vs actual years)\n"
        "- Education Match: 20% weight (exact > related > unrelated)\n"
        "- match_score must be 0-100\n"
        "- shortlisted = true if match_score >= 60, false otherwise\n\n"
        "EMPLOYMENT TIMELINE AND TOTAL GAP RULES:\n"
        "- experience_timeline must list individual roles with start_date and end_date extracted from the resume.\n"
        "- Normalize dates into normalized_start_date and normalized_end_date using YYYY-MM format (or empty string for ongoing roles).\n"
        "- Sort experience_timeline from most recent to oldest based on normalized dates.\n"
        "- Use the ordered experience_timeline to identify all gaps between consecutive roles and calculate their durations in years and months.\n"
        "- Sum all individual gaps to produce total_employment_gap with numeric years/months and a concise summary string (e.g., 'Total Experience Gap: 2 years 5 months'). Do not output per-job gap details.\n\n"
        
        "REASON FOR DECISION MUST BE PRECISE:\n"
        "- State exact numbers: 'Required: X years, Candidate has: Y years'\n"
        "- For education: 'Required: [degree], Candidate has: [degree]'\n"
        "- For skills: 'Matched X of Y mandatory skills'\n"
        "- Keep it factual and brief, no verbose explanations\n\n"
        
        "SUGGESTIONS RULE:\n"
        "- If shortlisted = false → suggestions must tell the candidate what to improve.\n"
        "- If shortlisted = true → suggestions must say: 'Proceed to the next interview round.'\n\n"
        
        "CRITICAL:\n"
        "- Output ONLY the JSON object, nothing else\n"
        "- Start with { and end with }\n"
        "- NO text before or after the JSON\n"
        "- NO markdown code blocks\n"
        "- NO comments\n\n"
        
        "RESUME:\n" + cv_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n"
    )


def extract_minimal_json(raw_response: str) -> Dict[str, Any]:
    """Extract and validate the minimal JSON from the model response.
    
    Args:
        raw_response: Raw text response from the model
    
    Returns:
        Validated minimal JSON dictionary
    
    Raises:
        ValueError: If JSON extraction or validation fails
    """
    text = raw_response.strip()
    
    # Remove markdown code fences if present
    if "```" in text:
        import re
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = text.replace('```', '')
        text = text.strip()
    
    # Try direct JSON parsing
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            candidate = text[start_idx:end_idx + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                # Clean up common issues
                import re
                candidate = re.sub(r',\s*}', '}', candidate)
                candidate = re.sub(r',\s*]', ']', candidate)
                data = json.loads(candidate)
        else:
            raise ValueError("No valid JSON object found in response")
    
    # Validate and ensure all required fields exist
    return validate_minimal_structure(data)


def validate_minimal_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and ensure the minimal JSON structure is complete.
    
    Args:
        data: Raw JSON dictionary from the model
    
    Returns:
        Validated and complete minimal JSON dictionary
    """
    # Initialize the structure with defaults
    result = {
        "candidate_details": {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "linkedin": "",
            "other_urls": []
        },
        "education": [],
        "experience": {
            "total_years": "",
            "relevant_experience_summary": ""
        },
        "experience_timeline": [],
        "total_employment_gap": {
            "years": 0,
            "months": 0,
            "summary": "",
        },
        "all_skills": [],
        "certifications": [],
        "skills_evaluation": {
            "matched_skills": [],
            "missing_skills": [],
            "mandatory_skill_gaps": []
        },
        "match_result": {
            "match_score": 0,
            "shortlisted": False,
            "threshold": 60,
            "reason_for_decision": ""
        },
        "suggestions": ""
    }
    
    # Update with actual data from model response
    if isinstance(data, dict):
        # Candidate details
        if "candidate_details" in data and isinstance(data["candidate_details"], dict):
            for key in result["candidate_details"]:
                if key in data["candidate_details"]:
                    if key == "other_urls":
                        result["candidate_details"][key] = data["candidate_details"][key] if isinstance(data["candidate_details"][key], list) else []
                    else:
                        result["candidate_details"][key] = str(data["candidate_details"][key] or "")
        
        # Education
        if "education" in data and isinstance(data["education"], list):
            result["education"] = []
            for edu in data["education"]:
                if isinstance(edu, dict):
                    result["education"].append({
                        "degree": str(edu.get("degree", "")),
                        "year": str(edu.get("year", "")),
                        "institution": str(edu.get("institution", ""))
                    })
        
        # Experience
        if "experience" in data and isinstance(data["experience"], dict):
            result["experience"]["total_years"] = str(data["experience"].get("total_years", ""))
            result["experience"]["relevant_experience_summary"] = str(data["experience"].get("relevant_experience_summary", ""))

        # Experience timeline (optional, for employment gap analysis)
        if "experience_timeline" in data and isinstance(data["experience_timeline"], list):
            normalized_timeline: List[Dict[str, Any]] = []
            for item in data["experience_timeline"]:
                if isinstance(item, dict):
                    normalized_timeline.append(
                        {
                            "role": str(item.get("role", "")),
                            "company": str(item.get("company", "")),
                            "start_date": str(item.get("start_date", "")),
                            "end_date": str(item.get("end_date", "")),
                            "normalized_start_date": str(item.get("normalized_start_date", "")),
                            "normalized_end_date": str(item.get("normalized_end_date", "")),
                        }
                    )
            result["experience_timeline"] = normalized_timeline
        
        # All skills
        if "all_skills" in data and isinstance(data["all_skills"], list):
            result["all_skills"] = [str(skill) for skill in data["all_skills"]]
        
        # Certifications
        if "certifications" in data and isinstance(data["certifications"], list):
            result["certifications"] = [str(cert) for cert in data["certifications"]]

        # Total employment gap: always derive it from the experience_timeline dates
        total_gap = _calculate_total_employment_gap(result.get("experience_timeline", []))
        result["total_employment_gap"] = total_gap
        
        # Skills evaluation
        if "skills_evaluation" in data and isinstance(data["skills_evaluation"], dict):
            se = data["skills_evaluation"]
            result["skills_evaluation"]["matched_skills"] = se.get("matched_skills", []) if isinstance(se.get("matched_skills"), list) else []
            result["skills_evaluation"]["missing_skills"] = se.get("missing_skills", []) if isinstance(se.get("missing_skills"), list) else []
            result["skills_evaluation"]["mandatory_skill_gaps"] = se.get("mandatory_skill_gaps", []) if isinstance(se.get("mandatory_skill_gaps"), list) else []
        
        # Match result
        if "match_result" in data and isinstance(data["match_result"], dict):
            mr = data["match_result"]
            try:
                score = float(mr.get("match_score", 0))
                result["match_result"]["match_score"] = max(0, min(100, score))
            except (TypeError, ValueError):
                result["match_result"]["match_score"] = 0
            
            result["match_result"]["shortlisted"] = bool(mr.get("shortlisted", False))
            result["match_result"]["threshold"] = int(mr.get("threshold", 60))
            result["match_result"]["reason_for_decision"] = str(mr.get("reason_for_decision", ""))
        
        # Suggestions
        result["suggestions"] = str(data.get("suggestions", ""))
    
    # Ensure consistency between match_score and shortlisted
    if result["match_result"]["match_score"] >= result["match_result"]["threshold"]:
        if not result["match_result"]["shortlisted"]:
            result["match_result"]["shortlisted"] = True
        if not result["suggestions"]:
            result["suggestions"] = "Proceed to the next interview round."
    else:
        if result["match_result"]["shortlisted"]:
            result["match_result"]["shortlisted"] = False
        if not result["suggestions"]:
            result["suggestions"] = "Focus on acquiring the missing mandatory skills and gaining more relevant experience."
    
    return result


def _calculate_total_employment_gap(experience_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate the total employment gap from a normalized experience timeline.

    This helper assumes that each item in experience_timeline may contain
    "normalized_start_date" and "normalized_end_date" in "YYYY-MM" format.
    If normalized dates are missing, it falls back to raw "start_date" and
    "end_date" when they appear to be in a parseable year-month form.

    The function:
    - Sorts experiences from most recent to oldest using end/start dates
    - Computes gaps between consecutive roles
    - Sums all gaps into total years and months
    - Returns a structure with numeric years/months and a concise summary
    """

    def _parse_ym(value: str) -> Optional[datetime]:
        value = (value or "").strip().replace(",", "")
        if not value:
            return None

        # Try strict numeric year-month formats first
        for fmt in ("%Y-%m", "%Y/%m", "%Y.%m"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        # Common textual month formats: "Jan 2020", "January 2020", etc.
        for fmt in ("%b %Y", "%B %Y", "%b-%Y", "%B-%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        # If only year is given, treat as January of that year
        try:
            if len(value) == 4 and value.isdigit():
                return datetime.strptime(value + "-01", "%Y-%m")
        except ValueError:
            pass
        return None

    # Normalize timeline entries into parseable start/end datetimes
    normalized: List[Dict[str, Any]] = []
    for item in experience_timeline or []:
        if not isinstance(item, dict):
            continue

        n_start = item.get("normalized_start_date") or item.get("start_date") or ""
        n_end = item.get("normalized_end_date") or item.get("end_date") or ""

        start_dt = _parse_ym(str(n_start))
        end_dt = _parse_ym(str(n_end))

        # If no end date (e.g., ongoing role), use start date as both to avoid
        # treating current employment as a gap contributor here.
        if start_dt is None and end_dt is None:
            continue
        if start_dt is None and end_dt is not None:
            start_dt = end_dt
        if end_dt is None and start_dt is not None:
            end_dt = start_dt

        normalized.append({
            "role": item.get("role", ""),
            "start": start_dt,
            "end": end_dt,
        })

    if len(normalized) < 2:
        return {
            "years": 0,
            "months": 0,
            "summary": "Total Experience Gap: 0 years 0 months",
        }

    # Sort from oldest to most recent by start then end date
    normalized.sort(key=lambda x: (x["start"], x["end"]))

    total_months_gap = 0

    for idx in range(len(normalized) - 1):
        older = normalized[idx]
        newer = normalized[idx + 1]

        end_older = older["end"]
        start_newer = newer["start"]

        # Gap is the time between end_older and start_newer; if start_newer is
        # on or before end_older, there is no positive gap to add.
        if start_newer <= end_older:
            continue

        # Convert to full-month difference
        year_diff = start_newer.year - end_older.year
        month_diff = start_newer.month - end_older.month
        months_gap = year_diff * 12 + month_diff

        if months_gap > 0:
            total_months_gap += months_gap

    years = total_months_gap // 12
    months = total_months_gap % 12

    summary = f"Total Experience Gap: {years} years {months} months"

    return {
        "years": years,
        "months": months,
        "summary": summary,
    }


def process_candidate(cv_text: str, jd_text: str) -> Dict[str, Any]:
    """Process a candidate's resume against a job description.
    
    This is the main entry point for the AI engine. It builds the prompt,
    calls the LLM, and returns the minimal structured JSON output.
    
    Args:
        cv_text: The candidate's resume text
        jd_text: The job description text
    
    Returns:
        Minimal structured JSON dictionary for shortlisting decision
    
    Raises:
        RuntimeError: If the LLM fails to return valid JSON after retries
    """
    client = GrokClient()
    
    # First attempt
    prompt = build_minimal_scoring_prompt(cv_text, jd_text)
    logger.info("Sending candidate evaluation request to LLM")
    raw_response = client.run(prompt)
    
    try:
        result = extract_minimal_json(raw_response)
        logger.info(f"Successfully processed candidate: {result.get('candidate_details', {}).get('name', 'Unknown')}")
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Invalid JSON on first attempt: {exc}")
    
    # Retry with stronger enforcement
    retry_prompt = (
        prompt + 
        "\n\nYour previous response was not valid JSON. "
        "Output ONLY the JSON object with the exact structure shown above. "
        "No other text. Start with { and end with }"
    )
    
    logger.info("Retrying with JSON enforcement")
    raw_retry_response = client.run(retry_prompt)
    
    try:
        result = extract_minimal_json(raw_retry_response)
        logger.info(f"Successfully processed candidate on retry: {result.get('candidate_details', {}).get('name', 'Unknown')}")
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error(f"Failed to get valid JSON after retry: {exc}")
        # Return a default structure with error indication
        return {
            "candidate_details": {
                "name": "Error Processing",
                "email": "",
                "phone": "",
                "location": "",
                "linkedin": "",
                "other_urls": []
            },
            "education": [],
            "experience": {
                "total_years": "Unable to process",
                "relevant_experience_summary": "Error: Could not extract information from resume"
            },
            "all_skills": [],
            "certifications": [],
            "skills_evaluation": {
                "matched_skills": [],
                "missing_skills": [],
                "mandatory_skill_gaps": []
            },
            "match_result": {
                "match_score": 0,
                "shortlisted": False,
                "threshold": 60,
                "reason_for_decision": "Failed to process resume - invalid response from AI model"
            },
            "suggestions": "Please resubmit the resume in a clearer format or contact support."
        }


# Convenience function for backward compatibility
def score_candidate_minimal(cv_text: str, jd_text: str) -> Dict[str, Any]:
    """Score a candidate with minimal JSON output.
    
    This function provides backward compatibility while using the new
    minimal JSON structure.
    
    Args:
        cv_text: The candidate's resume text
        jd_text: The job description text
    
    Returns:
        Minimal structured JSON dictionary for shortlisting decision
    """
    return process_candidate(cv_text, jd_text)
