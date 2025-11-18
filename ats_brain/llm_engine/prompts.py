"""Prompt builders for Grok-based candidate scoring."""

from __future__ import annotations
from datetime import datetime


def build_scoring_prompt(cv_text: str, jd_text: str) -> str:
    """Build a structured prompt instructing Grok to score a candidate.

    The model must:
    - Analyse the resume (CV) and job description.
    - Extract key skills, experience, and education.
    - Apply weighted scoring rules:
      * Skill Match – 50%% of the overall score.
        - Mandatory / must-have skills get the highest weight.
        - Good-to-have / nice-to-have skills get a lower weight.
      * Experience Match – 30%% of the overall score.
        - Compare required experience in the JD with the candidate's experience.
      * Education Match – 20%% of the overall score.
        - Exact matching degree/field = full points.
        - Closely related field = partial points.
        - Unrelated field = low points.
    - Compute a final numeric match score between 0 and 100 based on these
      weighted components.
    - Decide whether to shortlist the candidate using this rule:
      * If overall match score >= 60 then shortlisted = true.
      * Otherwise shortlisted = false.
    - Provide detailed reasons for the decision, referencing skills,
      experience, and education.

    The model must return **only valid JSON**, with no prose, markdown, or
    commentary outside the JSON object. The JSON must contain at least:
    - ``match_score``: number (0-100) representing the overall weighted score.
    - ``shortlisted``: boolean derived from the 60+ rule.
    - ``reasons``: string or array of strings describing the reasoning.

    Additionally, the JSON should be professionally structured with these
    top-level sections (field names are mandatory; inner content can be
    strings, numbers, or arrays as appropriate):

    - ``evaluation_metadata``: {
        ``model_version``, ``timestamp``, ``job_id``, ``candidate_id``
      }
    - ``overall_evaluation``: {
        ``match_score`` (0-100 number),
        ``shortlisted`` (boolean),
        ``threshold`` (number, usually 60)
      }
    - ``candidate_contact_details``: {
        ``name``, ``phone_number``, ``email``, ``location``,
        ``linkedin_url``, ``portfolio_url``
      }
    - ``candidate_profile``: {
        ``education``: {
            ``highest_degree``,
            ``details``: list of objects with degree/institution/year/grade
          },
        ``experience``: {
            ``total_experience``, ``current_role``, ``domains``,
            ``technologies``
          }
      }
    - ``jd_requirement_analysis``: {
        ``experience_analysis``: {
            ``required_experience``,
            ``candidate_experience``,
            ``match_level`` ("Full" | "Partial" | "No Match"),
            ``experience_match_percentage``,
            ``explanation``
          },
        ``education_analysis``: {
            ``required``,
            ``candidate``,
            ``match_level`` (e.g. "Match" | "Related" | "Unrelated"),
            ``education_match_percentage``,
            ``explanation``
          },
        ``skills_analysis``: {
            ``mandatory_skills`` (list),
            ``matched_skills`` (list),
            ``unmatched_skills`` (list),
            ``skill_match_percentage``
          },
        ``semantic_match``: {
            ``semantic_score`` (0-100 number),
            ``overall_similarity`` (string),
            ``explanation``
          }
      }
    - ``decision_explanation``: {
        ``summary``,
        ``positive_factors`` (list of strings),
        ``negative_factors`` (list of strings)
      }
    - ``recommendations``: {
        For shortlisted = true:
          * focus on next steps such as "proceed to next round of initial screening",
            interview scheduling, and communication tone;
          * DO NOT include any "areas_for_improvement" field.
        For shortlisted = false:
          * include ``areas_for_improvement`` listing upskilling suggestions
            for skills or experience gaps;
          * include guidance on sending a professional rejection e-mail with
            upskilling advice.
      }
    - ``final_decision``: {
        ``final_decision_reason`` (string summarising why the candidate is or
        is not moving forward)
      }
    """

    return (
        "You are an AI assistant helping to evaluate how well a candidate's resume "
        "matches a given job description.\n\n"
        "Carefully read BOTH documents below and then respond with a single JSON "
        "object only. Do not include any text before or after the JSON. Do not "
        "use markdown. Do not include comments. The JSON must be strictly valid "
        "and parseable.\n\n"
        "Your task:\n"
        "1. Analyse the RESUME (CV) and JOB DESCRIPTION.\n"
        "2. Identify key hard skills, soft skills, years of experience, "
        "   and education/qualifications. Explicitly distinguish mandatory or must-have "
        "   skills from good-to-have/optional skills based on the wording of the JD.\n"
        "3. Compute an overall match score between 0 and 100 using these weights:\n"
        "   - Skill Match = 50% of the score (mandatory skills highest impact).\n"
        "   - Experience Match = 30% of the score (compare required vs candidate years).\n"
        "     When the JD specifies an explicit range (e.g. 0-2 years), treat:\n"
        "       - candidate years inside the range as a full match,\n"
        "       - slightly above/below the range as a partial match, and\n"
        "       - far outside the range as a no_match for experience.\n"
        "   - Education Match = 20% of the score (exact match > related > unrelated).\n"
        "4. Apply the shortlisting rule:\n"
        "   - shortlisted = true  if match_score >= 60.\n"
        "   - shortlisted = false if match_score < 60.\n"
        "5. Provide clear, concise reasons for the score and shortlist decision, "
        "   explicitly mentioning skills, experience, and education factors.\n\n"
        "Output format (this JSON schema is mandatory at the top level):\n"
        "{\n"
        "  \"evaluation_metadata\": { ... },\n"
        "  \"overall_evaluation\": { ... },\n"
        "  \"candidate_contact_details\": { ... },\n"
        "  \"candidate_profile\": { ... },\n"
        "  \"jd_requirement_analysis\": { ... },\n"
        "  \"decision_explanation\": { ... },\n"
        "  \"recommendations\": { ... },\n"
        "  \"final_decision\": { ... }\n"
        "}\n\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the JSON object, nothing else.\n"
        "- Start your response with { and end with }\n"
        "- Do NOT include any text before or after the JSON.\n"
        "- Do NOT wrap the JSON in markdown code blocks (no ```).\n"
        "- Do NOT include comments in the JSON.\n"
        "- Ensure the JSON is valid and can be parsed by a strict JSON parser.\n"
        "- Do not include trailing commas.\n\n"
        "RESUME (CV):\n" + cv_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n"
    )


def build_llm_prompt(cv_text: str, jd_text: str) -> str:
    """Build the prompt for the LLM to score a candidate against a JD with enhanced precision.
    
    Args:
        cv_text: The candidate's CV/resume text
        jd_text: The job description text
    
    Returns:
        The formatted prompt string for the LLM
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return (
        "You are an expert recruiter evaluating a candidate's CV against a job description.\n"
        f"Today's date is {current_date}. Use this to calculate precise experience duration.\n\n"
        
        "CRITICAL INSTRUCTIONS FOR PRECISE EVALUATION:\n"
        "1. EXPERIENCE CALCULATION:\n"
        "   - Calculate EXACT duration in format: 'X years Y months'\n"
        "   - If dates show 'Present', 'Current', or 'Till Date', use today's date\n"
        "   - Example: Aug 2023 to Present (today {current_date}) = calculate exact months\n"
        "   - Compare calculated experience with JD requirement precisely\n"
        "   - State clearly: 'Required: X years, Candidate has: Y years Z months'\n\n"
        
        "2. EDUCATION MATCHING:\n"
        "   - COMPLETE MATCH: Exact degree and field as required\n"
        "   - PARTIAL MATCH: Related degree or field (e.g., CS degree for IT role)\n"
        "   - NO MATCH: Unrelated degree or missing required qualification\n"
        "   - State clearly: 'Required: [degree], Candidate has: [degree] - [MATCH LEVEL]'\n\n"
        
        "3. SKILLS ANALYSIS:\n"
        "   - List ALL mandatory skills from JD\n"
        "   - For EACH skill, state: ✓ MATCHED or ✗ NOT FOUND\n"
        "   - Calculate percentage: (matched skills / total required skills) × 100\n"
        "   - Distinguish between mandatory and optional skills\n\n"
        
        "4. DETAILED REASONING:\n"
        "   - In 'reasons' field, provide specific details:\n"
        "     * Skills: 'Matched X of Y mandatory skills (list them)'\n"
        "     * Experience: 'Has X years Y months vs required Z years'\n"
        "     * Education: '[Degree] is [COMPLETE/PARTIAL/NO] match for requirement'\n"
        "   - State clearly WHY shortlisted or rejected based on these factors\n\n"
        
        "Scoring Weights:\n"
        "- Skills Match: 50% (focus on mandatory skills)\n"
        "- Experience Match: 30% (exact comparison)\n"
        "- Education Match: 20% (degree relevance)\n\n"
        
        "Shortlisting Rule: Overall score >= 60% = SHORTLISTED\n\n"
        
        "Output format (EXACT JSON structure - follow this precisely):\n"
        "{\n"
        "  \"evaluation_metadata\": {\n"
        "    \"model_version\": \"llm-v1\",\n"
        "    \"timestamp\": \"" + current_date + "\",\n"
        "    \"job_id\": \"\",\n"
        "    \"candidate_id\": \"\"\n"
        "  },\n\n"
        "  \"overall_evaluation\": {\n"
        "    \"match_score\": [0-100 number],\n"
        "    \"shortlisted\": [true if score >= 60, false otherwise],\n"
        "    \"threshold\": 60\n"
        "  },\n\n"
        "  \"candidate_contact_details\": {\n"
        "    \"name\": \"[full name]\",\n"
        "    \"phone_number\": \"[phone]\",\n"
        "    \"email\": \"[email]\",\n"
        "    \"location\": \"[city, state/country]\",\n"
        "    \"linkedin_url\": \"[LinkedIn URL or empty string]\",\n"
        "    \"portfolio_url\": \"[Portfolio/GitHub URL or empty string]\"\n"
        "  },\n\n"
        "  \"candidate_profile\": {\n"
        "    \"education\": {\n"
        "      \"highest_degree\": \"[highest degree name]\",\n"
        "      \"details\": [\n"
        "        {\n"
        "          \"degree\": \"[degree name]\",\n"
        "          \"institution\": \"[university/college]\",\n"
        "          \"year\": [graduation year as number],\n"
        "          \"grade\": \"[CGPA/percentage]\"\n"
        "        }\n"
        "      ]\n"
        "    },\n"
        "    \"experience\": {\n"
        "      \"total_experience\": \"[X years Y months calculated from dates]\",\n"
        "      \"current_role\": \"[current job title]\",\n"
        "      \"domains\": [\"list of industry domains\"],\n"
        "      \"technologies\": [\"list of all technologies used\"]\n"
        "    }\n"
        "  },\n\n"
        "  \"jd_requirement_analysis\": {\n"
        "    \"experience_analysis\": {\n"
        "      \"required_experience\": \"[from JD, e.g., 2+ years]\",\n"
        "      \"candidate_experience\": \"[X years Y months]\",\n"
        "      \"match_level\": \"Full\" or \"Partial\" or \"No Match\",\n"
        "      \"experience_match_percentage\": [0-100],\n"
        "      \"explanation\": \"[detailed comparison]\"\n"
        "    },\n"
        "    \"education_analysis\": {\n"
        "      \"required\": \"[education requirement from JD]\",\n"
        "      \"candidate\": \"[candidate's education]\",\n"
        "      \"match_level\": \"Match\" or \"Related\" or \"Unrelated\",\n"
        "      \"education_match_percentage\": [0-100],\n"
        "      \"explanation\": \"[why it matches or doesn't]\"\n"
        "    },\n"
        "    \"skills_analysis\": {\n"
        "      \"mandatory_skills\": [\"list ALL required skills from JD\"],\n"
        "      \"matched_skills\": [\"skills candidate has\"],\n"
        "      \"unmatched_skills\": [\"skills candidate lacks\"],\n"
        "      \"skill_match_percentage\": [0-100]\n"
        "    },\n"
        "    \"semantic_match\": {\n"
        "      \"semantic_score\": [0-100],\n"
        "      \"overall_similarity\": \"High\" or \"Moderate to High\" or \"Moderate\" or \"Low\",\n"
        "      \"explanation\": \"[overall fit analysis]\"\n"
        "    }\n"
        "  },\n\n"
        "  \"decision_explanation\": {\n"
        "    \"summary\": \"[concise overall assessment]\",\n"
        "    \"positive_factors\": [\n"
        "      \"[list of candidate strengths]\"\n"
        "    ],\n"
        "    \"negative_factors\": [\n"
        "      \"[list of gaps or weaknesses]\"\n"
        "    ]\n"
        "  },\n\n"
        "  \"recommendations\": {\n"
        "    \"strengths\": [\n"
        "      \"[candidate's key strengths]\"\n"
        "    ],\n"
        "    \"areas_for_improvement\": [\n"
        "      \"[skills or areas to develop]\"\n"
        "    ]\n"
        "  },\n\n"
        "  \"final_decision\": {\n"
        "    \"final_decision_reason\": \"[clear explanation of decision]\"\n"
        "  },\n\n"
        "  \"match_score\": [same as overall_evaluation.match_score],\n"
        "  \"shortlisted\": [same as overall_evaluation.shortlisted],\n"
        "  \"reasons\": \"[Detailed: Skills - matched X/Y mandatory skills. Experience - has X years Y months vs Z required. Education - [degree] is [match level] for role.]\"\n"
        "}\n\n"
        
        "CRITICAL OUTPUT RULES:\n"
        "- Output ONLY valid JSON, nothing else\n"
        "- Start with { and end with }\n"
        "- NO text before or after the JSON\n"
        "- NO markdown code blocks (no ```)\n"
        "- NO comments in the JSON\n"
        "- NO trailing commas\n"
        "- BE PRECISE with numbers, dates, and calculations\n\n"
        
        "RESUME (CV):\n" + cv_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n"
    )
