"""Job description parsing with skill extraction, experience detection, and embedding computation."""
from typing import Dict, Any, Optional
import re
from loguru import logger

from utils.text_cleaner import normalize_text, split_sentences
from utils.skill_extractor import extract_skills
from utils.scoring import compute_embedding, serialize_embedding


EXPERIENCE_REGEX = re.compile(
    r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\.?)",
    re.IGNORECASE
)

EDUCATION_REGEX = re.compile(
    r"(?:bachelor|master|phd|doctorate|bs|ms|ph\.?d\.?)\s+(?:degree|in|of)",
    re.IGNORECASE
)


def extract_experience(jd_text: str) -> Optional[str]:
    """
    Extract experience requirement from JD text.
    
    Args:
        jd_text: Job description text
        
    Returns:
        Experience string (e.g., "5+ years") or None
    """
    matches = EXPERIENCE_REGEX.findall(jd_text)
    if matches:
        # Get the highest mentioned years
        years = [int(m) for m in matches]
        max_years = max(years)
        return f"{max_years}+ years"
    return None


def extract_education(jd_text: str) -> Optional[str]:
    """
    Extract education requirement from JD text.
    
    Args:
        jd_text: Job description text
        
    Returns:
        Education requirement string or None
    """
    match = EDUCATION_REGEX.search(jd_text)
    if match:
        # Extract surrounding context
        start = max(0, match.start() - 30)
        end = min(len(jd_text), match.end() + 30)
        context = jd_text[start:end]
        return context.strip()
    return None


def parse_job_description(
    jd_text: str,
    title: Optional[str] = None,
    compute_embedding_vector: bool = True
) -> Dict[str, Any]:
    """
    Parse job description: extract skills, experience, education, and compute embedding.
    
    Args:
        jd_text: Raw job description text
        title: Optional job title
        compute_embedding_vector: Whether to compute and include embedding (default: True)
        
    Returns:
        Dictionary with keys:
            - title: Job title
            - experience: Experience requirement (e.g., "5+ years")
            - education: Education requirement (if found)
            - skills: List of extracted skills
            - summary: Summary text (first 2 sentences)
            - jd_text: Normalized full JD text
            - embedding: Base64-encoded embedding string (if compute_embedding_vector=True)
            - embedding_method: "sbert" or "tfidf" (if embedding computed)
    """
    if not jd_text:
        jd_text = ""
    
    # Normalize text
    normalized_text = normalize_text(jd_text)
    
    # Extract components
    experience = extract_experience(normalized_text)
    education = extract_education(normalized_text)
    skills = extract_skills(normalized_text)
    
    # Create summary (first 2 sentences)
    sentences = split_sentences(normalized_text)
    summary = ". ".join(sentences[:2]) if sentences else (title or "")
    
    result: Dict[str, Any] = {
        "title": title,
        "experience": experience,
        "education": education,
        "skills": skills,
        "summary": summary,
        "jd_text": normalized_text,
    }
    
    # Compute embedding if requested
    if compute_embedding_vector:
        try:
            embedding_array, method = compute_embedding(normalized_text)
            if embedding_array is not None:
                embedding_str = serialize_embedding(embedding_array)
                result["embedding"] = embedding_str
                result["embedding_method"] = method
            else:
                result["embedding"] = None
                result["embedding_method"] = method
        except Exception as e:
            logger.warning("Failed to compute JD embedding: {}", e)
            result["embedding"] = None
            result["embedding_method"] = None
    
    return result
