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
        
        # All skills
        if "all_skills" in data and isinstance(data["all_skills"], list):
            result["all_skills"] = [str(skill) for skill in data["all_skills"]]
        
        # Certifications
        if "certifications" in data and isinstance(data["certifications"], list):
            result["certifications"] = [str(cert) for cert in data["certifications"]]
        
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
