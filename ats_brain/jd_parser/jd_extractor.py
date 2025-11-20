"""Job description parsing with skill extraction, experience detection, and embedding computation."""
from typing import Dict, Any, Optional
import re
from loguru import logger

from utils.text_cleaner import normalize_text, split_sentences
from utils.skill_extractor import extract_skills
from utils.scoring import compute_embedding, serialize_embedding
from utils.location_parser import extract_jd_locations


EXPERIENCE_REGEX = re.compile(
    r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\.?)",
    re.IGNORECASE
)

EDUCATION_REGEX = re.compile(
    r"(?:bachelor|master|phd|doctorate|bs|ms|ph\.?d\.?)\s+(?:degree|in|of)",
    re.IGNORECASE
)


def extract_job_title(jd_text: str) -> Optional[str]:
    if not jd_text:
        return None

    lines = [ln.strip() for ln in jd_text.split("\n") if ln.strip()]
    head = lines[:25]

    SECTION_PREFIXES = {
        "skills", "responsibilities", "requirements", "qualifications",
        "about", "company", "summary", "benefits", "location",
        "job description", "about us", "who we are", "key skills",
        "desired skills", "mandatory skills"
    }

    TITLE_HINT_WORDS = {
        "engineer", "developer", "designer", "manager", "analyst", "consultant",
        "administrator", "specialist", "lead", "intern", "scientist", "architect",
        "coordinator", "executive", "technician", "officer"
    }

    patterns = [
        r"(?:Job\s*Title|Position|Role|Title)\s*[:\-]\s*([A-Z][A-Za-z0-9\s/&,+-]{3,80})",
        r"(?:Looking for|Seeking|Hiring)\s+(?:a|an)?\s*([A-Z][A-Za-z0-9\s/&,+-]{3,80})",
        r"^([A-Z][A-Za-z0-9\s/&,+-]{3,80})\s*(?:Position|Job|Role)\b",
    ]

    def _postprocess_title(raw: str) -> Optional[str]:
        t = raw.strip()
        for delim in [",", ".", " - ", " – ", " — ", " to ", " with ", " in ", " for "]:
            if delim in t:
                t = t.split(delim, 1)[0].strip()
        t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^A-Za-z/&+\-\s]", "", t)
        if 3 <= len(t) <= 80:
            return t
        return None

    # --- Step 1: Regex-based detection ---
    for ln in head:
        low = ln.lower()
        if any(low.startswith(p) for p in SECTION_PREFIXES):
            continue
        for pat in patterns:
            m = re.search(pat, ln, re.IGNORECASE)
            if m:
                title = _postprocess_title(m.group(1))
                if title and not any(k in title.lower() for k in SECTION_PREFIXES):
                    return title

    # --- Step 2: Heuristic detection ---
    for ln in head:
        low = ln.lower()
        if any(p in low for p in SECTION_PREFIXES):
            continue
        words = ln.split()
        if 2 <= len(words) <= 6 and any(w.lower() in TITLE_HINT_WORDS for w in words):
            clean_title = _postprocess_title(ln)
            if clean_title and not re.search(r"skills?", clean_title, re.IGNORECASE):
                return clean_title

    # --- Step 3: Fallback: first decent line with title-like pattern ---
    for candidate in head:
        low = candidate.lower()
        if any(p in low for p in SECTION_PREFIXES):
            continue
        if len(candidate.split()) <= 8 and any(
            kw in low for kw in TITLE_HINT_WORDS
        ):
            clean_title = _postprocess_title(candidate)
            if clean_title:
                return clean_title

    return None


def extract_experience(jd_text: str) -> Optional[str]:
    """
    Extract a conservative minimum years requirement from common patterns.
    Returns like "3-5 years" for ranges or "3+ years" for minimum styles, else None.
    """
    if not jd_text:
        return None
    t = str(jd_text)
    m = re.search(r"(\d+)\s*[–\-to]+\s*(\d+)\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))} years"
    m = re.search(r"at\s*least\s*(\d+)\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}+ years"
    m = re.search(r"minimum\s*(\d+)\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}+ years"
    m = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}+ years"
    return None


def extract_education(jd_text: str) -> Optional[str]:
    """
    Extract the sentence containing an education requirement keyword.
    Returns the sentence text for downstream parsing.
    """
    if not jd_text:
        return None
    sentences = split_sentences(jd_text)
    # Accept explicit levels and common acronyms; also accept generic 'degree in <discipline>'
    edu_pat = re.compile(
        r"\b(bachelor|master|phd|doctorate|bs|ms|b\.tech|m\.tech|btech|mtech|b\.e|m\.e|be|me)\b|\bdegree\s+in\b",
        re.IGNORECASE,
    )
    for s in sentences:
        if edu_pat.search(s):
            return s.strip()
    m = EDUCATION_REGEX.search(jd_text)
    if m:
        start = max(0, m.start() - 30)
        end = min(len(jd_text), m.end() + 30)
        return jd_text[start:end].strip()
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
    # Location requirements are extracted from the raw JD text to preserve
    # phrases like "Mumbai only", lists such as "India (Mumbai, Pune, ...)",
    # and other location hints. This is additive and does not affect existing
    # parsing logic.
    locations = extract_jd_locations(jd_text)
    
    # Create summary (first 2 sentences)
    sentences = split_sentences(normalized_text)
    summary = ". ".join(sentences[:2]) if sentences else (title or "")
    
    if not title:
        # Use raw text (with newlines) for better title detection
        title = extract_job_title(jd_text)

    result: Dict[str, Any] = {
        "title": title,
        "experience": experience,
        "education": education,
        "skills": skills,
        "summary": summary,
        "jd_text": normalized_text,
        "locations": locations,
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
