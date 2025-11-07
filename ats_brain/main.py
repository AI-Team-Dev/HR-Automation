"""FastAPI application for ATS Brain - Simplified Job Description and Resume matching API."""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from typing import Dict, Any, Optional, Union
from loguru import logger
import time
from datetime import datetime, timezone
import json
import base64
import numpy as np
import uuid
import re

from core.database import (
    get_job_posting,
    get_resume_blob,
    insert_parsed_data,
    get_parsed_jd,
    get_resume_metadata,
)
from core.config import settings
from jd_parser.jd_extractor import parse_job_description
from resume_parser.resume_extractor import extract_resume_text
from utils.scoring import (
    load_sbert_model,
    is_sbert_available,
    compute_embedding,
    serialize_embedding,
    deserialize_embedding,
    score_texts,
    cosine_similarity,
)
from utils.skill_extractor import extract_skills

# Configure logging
logger.add("ats_brain.log", rotation="10 MB", level="INFO")

app = FastAPI(
    title="ATS Brain - Simple Matching API",
    version="3.0.0",
    description="""
    ## 🎯 Super Simple JD & Resume Matching
    
    **ONE endpoint does everything:**
    - Upload JD → Database OR Local (text/file)
    - Upload Resume → Database OR Local (text/file)
    - Get Match Result → Immediately see similarity score
    - Results stored in database automatically
    
    ### How to use:
    1. Click `/match` endpoint below
    2. Choose JD: Enter `job_id` (DB) OR paste text OR upload file
    3. Choose Resume: Enter `resume_id` (DB) OR paste text OR upload file
    4. Click Execute → See results immediately!
    """
)


def extract_job_title_from_text(jd_text: str) -> Optional[str]:
    """Extract job title from JD text using common patterns."""
    if not jd_text:
        return None
    # If the JD is a single line with sections (e.g., "Title  Location: ... Skills: ..."),
    # cut off everything after the first section label.
    sec_label = re.search(r"\b(Location|Experience|Skills?|Responsibilities|Requirements|Summary|About|Company)\s*:", jd_text, re.IGNORECASE)
    if sec_label:
        prefix = jd_text[:sec_label.start()].strip()
        if prefix:
            # Use the last non-empty line from the prefix as candidate
            pre_lines = [ln.strip() for ln in prefix.split("\n") if ln.strip()]
            if pre_lines:
                candidate = pre_lines[-1]
                candidate = re.sub(r"\s+", " ", candidate)
                if 3 <= len(candidate) <= 100 and candidate[0].isalpha():
                    return candidate

    # Fallback for compact single-line JDs: capture leading role phrase
    m_head = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /&+\-]{3,80})(?:\s{2,}|\s+(?:Location|Experience|Skills?|Responsibilities|Requirements|Summary|About|Company)\s*:)", jd_text, re.IGNORECASE)
    if m_head:
        cand = re.sub(r"\s+", " ", m_head.group(1)).strip()
        if cand and not re.match(r"^skills?\b", cand, re.IGNORECASE) and 3 <= len(cand) <= 100:
            return cand

    # Look for common explicit labels in the first ~20 lines
    lines = [ln.strip() for ln in jd_text.split("\n") if ln.strip()]
    head = lines[:20]
    # Avoid picking section headers as titles
    SECTION_PREFIXES = {
        "skills", "responsibilities", "requirements", "qualifications",
        "about", "company", "summary", "benefits", "location",
        "job description", "about us", "who we are", "key skills",
        "experience", "education"
    }
    patterns = [
        r"(?:Job\s*Title|Position|Role|Title)\s*[:\-]\s*([A-Z][A-Za-z0-9\s/&,+-]{3,80})",
        r"(?:Looking for|Seeking|Hiring)\s+(?:a|an)?\s*([A-Z][A-Za-z0-9\s/&,+-]{3,80})",
        r"^([A-Z][A-Za-z0-9\s/&,+-]{3,80})\s*(?:Position|Job|Role)\b",
    ]
    def _postprocess_title(raw: str) -> Optional[str]:
        t = raw.strip()
        # Stop at common delimiters/clauses
        for delim in [",", ".", " - ", " – ", " — ", " to ", " with ", " in ", " for "]:
            if delim in t:
                t = t.split(delim, 1)[0].strip()
        # Remove leading determiners
        t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.IGNORECASE)
        # Normalize spaces
        t = re.sub(r"\s+", " ", t)
        # Heuristic: prefer trailing capitalized role phrase
        tokens = t.split()
        keep = []
        for tok in reversed(tokens):
            # Accept TitleCase or ALLCAPS tokens, or role connectors like "/" & "-"
            clean = re.sub(r"[^A-Za-z/&-]", "", tok)
            if not clean:
                break
            if clean[0].isupper():
                keep.append(tok)
                continue
            break
        if keep:
            candidate = " ".join(reversed(keep)).strip()
            candidate = re.sub(r"\s+", " ", candidate)
            if 3 <= len(candidate) <= 80:
                t = candidate
        # Drop leading fluff adjectives if still present
        FLUFF = {"highly","motivated","passionate","dynamic","results-oriented","skilled","talented","seasoned","enthusiastic","proactive","driven"}
        parts = t.split()
        while parts and parts[0].lower() in FLUFF:
            parts = parts[1:]
        t = " ".join(parts)
        if 3 <= len(t) <= 80:
            return t
        return None

    for ln in head:
        low = ln.lower()
        if any(low.startswith(p + ":") for p in SECTION_PREFIXES):
            continue
        for pat in patterns:
            m = re.search(pat, ln, re.IGNORECASE)
            if m:
                title = m.group(1).strip(" -,:\t")
                title = _postprocess_title(title) or title
                if 3 <= len(title) <= 100:
                    return title
    # Heuristic: first non-section capitalized line that looks like a title
    if head:
        for candidate in head:
            low = candidate.lower()
            if any(low.startswith(p + ":") for p in SECTION_PREFIXES):
                continue
            first_line = _postprocess_title(candidate)
            if first_line and 5 <= len(first_line) <= 100 and first_line[0].isalpha():
                # Guard against single skill lines like "Python"
                if not re.match(r"^(skills?|tools?)\b", low):
                    return first_line
    return None

# Discipline extraction
DISCIPLINE_SYNONYMS = {
    "computer science": {"cs", "cse", "computer science", "cseng", "computing", "software engineering"},
    "information technology": {"it", "information technology"},
    "data science": {"data science", "data analytics", "analytics", "ai", "ml", "artificial intelligence", "machine learning"},
    "electronics": {"ece", "electronics", "electronics and communication", "electronics & communication"},
    "electrical": {"electrical", "eee", "electrical engineering"},
    "mechanical": {"mechanical", "mech", "mechanical engineering"},
    "civil": {"civil", "civil engineering"},
    "statistics": {"statistics", "statistical"},
    "mathematics": {"mathematics", "maths", "applied mathematics"},
}

def _norm_discipline_token(tok: str) -> str:
    t = (tok or "").strip().lower()
    for std, syns in DISCIPLINE_SYNONYMS.items():
        if t in syns:
            return std
    return t

def extract_discipline(text: str) -> Optional[str]:
    t = (text or "").lower()
    # direct pattern search
    for std, syns in DISCIPLINE_SYNONYMS.items():
        for s in syns:
            if re.search(rf"\b{re.escape(s)}\b", t):
                return std
    return None

# Experience utilities
def parse_years_requirement(text: str) -> Optional[int]:
    """Return minimum years required from a JD text snippet.
    Supports formats like '4-5 years', '4 – 5 yrs', '4+ years', 'at least 3 years'.
    Interprets any form as a minimum threshold.
    """
    if not text:
        return None
    t = str(text)
    m = re.search(r"(\d+)\s*[–\-to]+\s*(\d+)\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"at\s*least\s*(\d+)\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def parse_years_from_resume(text: str) -> Optional[int]:
    """Extract a conservative estimate of candidate total years by taking the maximum years found."""
    if not text:
        return None
    years = [int(x) for x in re.findall(r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)", text, re.IGNORECASE)]
    return max(years) if years else None


# -----------------------------------------------
# Skill normalization and weighting utilities
# -----------------------------------------------

ALIASES = {
    "postgres": "postgresql",
    "postgre": "postgresql",
    "ms sql": "sql-server",
    "mssql": "sql-server",
    "sql server": "sql-server",
    "sklearn": "scikit-learn",
    "np": "numpy",
    "tf": "tensorflow",
    "torch": "pytorch",
    "ml": "machine learning",
    "nlp": "nlp",
    "ms excel": "excel",
}


def _canon_skill(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)


# -----------------------------------------------
# Education parsing utilities
# -----------------------------------------------
EDU_PATTERNS = [
    ("phd", re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b", re.IGNORECASE)),
    ("master", re.compile(r"\b(m\.?sc|m\.?tech|m\.?e|masters?|post\s*graduate|mba)\b", re.IGNORECASE)),
    ("bachelor", re.compile(r"\b(b\.?tech|b\.?e|b\.?sc|bachelors?)\b", re.IGNORECASE)),
    ("diploma", re.compile(r"\b(diploma)\b", re.IGNORECASE)),
]

EDU_RANK = {"diploma": 0, "bachelor": 1, "master": 2, "phd": 3}

def extract_education_level(text: str) -> Optional[str]:
    t = text or ""
    for level, pat in EDU_PATTERNS:
        if pat.search(t):
            return level
    return None


def normalize_skills(skills: list) -> list:
    if not skills:
        return []
    return sorted(list({ _canon_skill(x) for x in skills if x }))


def detect_required_skills(jd_text: str, all_skills: list) -> set:
    if not jd_text or not all_skills:
        return set()
    required_markers = ["must have", "required", "mandatory", "at least"]
    sentences = [s.strip() for s in jd_text.split('.') if s.strip()]
    skills_set = set(normalize_skills(all_skills))
    required_sk = set()
    for s in sentences:
        if any(m in s.lower() for m in required_markers):
            s_sk = extract_skills(s)
            for sk in normalize_skills(s_sk):
                if sk in skills_set:
                    required_sk.add(sk)
    return required_sk

def compute_auto_weights(jd_text: str, jd_parsed: Dict[str, Any], required_skills: set) -> tuple:
    """Derive weights (semantic, skill, experience) from JD emphasis.

    Heuristics:
    - More required/mandatory skills -> boost skill weight
    - Explicit experience years -> boost experience weight by years
    - Many skills listed -> small boost to skill weight
    - Always normalize to sum=1
    """
    w_sem, w_skill, w_exp = 0.55, 0.35, 0.10

    text_lc = (jd_text or "").lower()
    skills = jd_parsed.get("skills", []) or []
    exp_text = jd_parsed.get("experience") or ""

    # Skill emphasis
    if len(required_skills) >= 2:
        w_skill += 0.10
    if len(required_skills) >= 4:
        w_skill += 0.05
    if any(k in text_lc for k in ["must have", "mandatory", "required:"]):
        w_skill += 0.05
    if len(skills) >= 12:
        w_skill += 0.05

    # Experience emphasis
    m = re.search(r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)", str(exp_text), re.IGNORECASE)
    if m:
        yrs = int(m.group(1))
        if yrs >= 5:
            w_exp += 0.10
        elif yrs >= 3:
            w_exp += 0.05

    # Normalize to sum=1
    total = w_sem + w_skill + w_exp
    if total <= 0:
        return 0.55, 0.35, 0.10
    w_sem, w_skill, w_exp = (w_sem/total, w_skill/total, w_exp/total)
    return w_sem, w_skill, w_exp

def compute_auto_weights_4f(
    jd_text: str,
    jd_parsed: Dict[str, Any],
    required_skills: set,
    req_level: Optional[str],
    req_discipline: Optional[str],
    jd_title: Optional[str]
) -> tuple:
    """Auto-derive 4-factor weights: (semantic, skills, experience, education).

    Heuristics:
    - Technical roles (keywords in title/JD) -> boost skills, keep semantic strong.
    - Managerial/communication-heavy -> boost semantic.
    - Explicit years in JD -> boost experience.
    - Explicit degree/discipline -> boost education.
    Always normalized to sum=1.
    """
    # Base weights
    w_sem, w_skill, w_exp, w_edu = 0.45, 0.35, 0.10, 0.10

    text_lc = (jd_text or "").lower()
    title_lc = (jd_title or "").lower()

    # Role type detection
    TECH_KEYS = ["engineer", "developer", "data", "backend", "frontend", "cloud", "devops", "ml", "ai", "analytics", "database", "etl"]
    MGR_KEYS = ["manager", "lead", "head", "director", "communications", "stakeholder", "coordination", "project"]
    if any(k in title_lc for k in TECH_KEYS) or any(k in text_lc for k in TECH_KEYS):
        w_skill += 0.10
        w_sem += 0.05
    if any(k in title_lc for k in MGR_KEYS) or any(k in text_lc for k in MGR_KEYS):
        w_sem += 0.10

    # Strong emphasis on required skills
    if len(required_skills) >= 3:
        w_skill += 0.05
    if len(required_skills) >= 5:
        w_skill += 0.05

    # Years & education emphasis
    if parse_years_requirement(jd_parsed.get("experience")) is not None:
        w_exp += 0.10
    if (req_level is not None) or (req_discipline is not None):
        w_edu += 0.05

    # Normalize
    total = w_sem + w_skill + w_exp + w_edu
    if total <= 0:
        return 0.45, 0.35, 0.10, 0.10
    return (w_sem/total, w_skill/total, w_exp/total, w_edu/total)
    
    # Common patterns for job titles
    patterns = [
        r"(?:Job Title|Position|Role|Title)[\s:]+([A-Z][A-Za-z\s&]+)",
        r"(?:Looking for|Seeking|Hiring)[\s]+(?:a|an)?[\s]+([A-Z][A-Za-z\s&]+)(?:[,\s]+with|\.)",
        r"^([A-Z][A-Za-z\s&]{5,50})(?:[\s-]+Position|Job|Role)",
    ]
    
    # Check first few lines
    lines = jd_text.split('\n')[:5]
    for line in lines:
        line = line.strip()
        if len(line) < 5 or len(line) > 100:
            continue
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if 5 <= len(title) <= 100:
                    return title
    
    # Heuristic: if text starts with a title followed by a comma, take the part before the first comma
    first_line_full = jd_text.strip().split('\n', 1)[0]
    if "," in first_line_full:
        candidate = first_line_full.split(",", 1)[0].strip()
        if 3 <= len(candidate) <= 80 and candidate[0].isalpha():
            return candidate

    # Fallback: Use first significant line
    first_line = lines[0].strip() if lines else ""
    if 5 <= len(first_line) <= 100 and first_line[0].isupper():
        return first_line
    
    return None


def extract_candidate_name_from_text(resume_text: str) -> Optional[str]:
    """Extract candidate name from resume text (usually first line or header)."""
    if not resume_text:
        return None
    
    # Check first few lines for name patterns
    lines = [ln.strip() for ln in resume_text.split('\n') if ln.strip()]
    header = lines[:15]
    excluded_keywords = {"email","phone","address","summary","objective","experience","education","skills","linkedin","github"}
    # 1) Explicit labels
    for ln in header:
        m = re.search(r"^(?:name)\s*[:\-]?\s*([A-Za-z][A-Za-z\s'.-]{3,50})$", ln, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"\s+", " ", cand)
            return cand
    # 2) Pure name-looking line (2-4 tokens, title case or caps, few symbols)
    for ln in header:
        if any(kw in ln.lower() for kw in excluded_keywords):
            continue
        if 3 <= len(ln) <= 60 and all(ch.isalpha() or ch.isspace() or ch in "'.-" for ch in ln):
            parts = [w for w in ln.split() if w]
            if 2 <= len(parts) <= 4 and sum(1 for w in parts if w[0].isalpha()) == len(parts):
                # Prefer title case or ALL CAPS words
                if all((w.istitle() or w.isupper()) for w in parts):
                    return " ".join(parts)
    # 3) Infer from email local-part if present
    m = re.search(r"([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "\n".join(header))
    if m:
        local = m.group(1)
        local = re.sub(r"\d+", " ", local)
        local = re.sub(r"[._-]+", " ", local)
        tokens = [t.capitalize() for t in local.split() if t.isalpha()]
        if 2 <= len(tokens) <= 3:
            return " ".join(tokens)
    return None


def _generate_match_reasoning(
    match_percentage: float,
    jd_skills: list,
    resume_skills: list,
    matched_skills: list,
    jd_experience: Optional[str],
    resume_text: str
) -> Dict[str, Any]:
    """Generate reasoning for match score with detailed explanations."""
    reasons = []
    strengths = []
    weaknesses = []
    
    # Skill matching analysis
    total_jd_skills = len(jd_skills)
    matched_count = len(matched_skills)
    missing_skills = list(set(jd_skills) - set(matched_skills))
    
    if total_jd_skills > 0:
        skill_match_ratio = (matched_count / total_jd_skills) * 100
    else:
        skill_match_ratio = 0
    
    # Experience analysis
    experience_match = True  # If JD doesn't specify, candidate is considered satisfying
    min_required = parse_years_requirement(jd_experience)
    if min_required is not None:
        cand_years = parse_years_from_resume(resume_text)
        if cand_years is not None and cand_years >= min_required:
            experience_match = True
            strengths.append(f"Meets/exceeds experience requirement ({cand_years} years)")
        else:
            experience_match = False
            if cand_years is not None:
                weaknesses.append(f"Below required experience: {cand_years} years vs {min_required}+ years required")
    
    # Overall match categorization
    if match_percentage >= 80:
        match_level = "HIGH"
        reasons.append(f"Strong match ({match_percentage:.1f}%) - Candidate closely aligns with job requirements")
    elif match_percentage >= 60:
        match_level = "MODERATE"
        reasons.append(f"Moderate match ({match_percentage:.1f}%) - Candidate has some relevant qualifications")
    elif match_percentage >= 40:
        match_level = "LOW"
        reasons.append(f"Low match ({match_percentage:.1f}%) - Candidate has limited alignment with requirements")
    else:
        match_level = "VERY_LOW"
        reasons.append(f"Very low match ({match_percentage:.1f}%) - Significant gaps between candidate and job requirements")
    
    # Skill-specific reasoning
    if skill_match_ratio >= 80:
        strengths.append(f"Excellent skill match: {matched_count}/{total_jd_skills} required skills found ({skill_match_ratio:.0f}%)")
    elif skill_match_ratio >= 60:
        strengths.append(f"Good skill match: {matched_count}/{total_jd_skills} required skills found ({skill_match_ratio:.0f}%)")
    elif skill_match_ratio >= 40:
        weaknesses.append(f"Partial skill match: Only {matched_count}/{total_jd_skills} required skills found ({skill_match_ratio:.0f}%)")
        if missing_skills:
            weaknesses.append(f"Missing key skills: {', '.join(missing_skills[:5])}")
    else:
        weaknesses.append(f"Poor skill match: Only {matched_count}/{total_jd_skills} required skills found ({skill_match_ratio:.0f}%)")
        if missing_skills:
            weaknesses.append(f"Missing critical skills: {', '.join(missing_skills[:5])}")
    
    # Additional resume skills
    extra_skills = list(set(resume_skills) - set(jd_skills))
    if extra_skills:
        strengths.append(f"Additional relevant skills: {', '.join(extra_skills[:5])}")
    
    # Summary
    if match_level == "HIGH":
        summary = f"Strong candidate match. Skill alignment is excellent ({skill_match_ratio:.0f}%), and semantic similarity is high ({match_percentage:.1f}%)."
    elif match_level == "MODERATE":
        summary = f"Moderate candidate match. Some skill gaps exist ({len(missing_skills)} missing skills), but core qualifications are present."
    else:
        summary = f"Weak candidate match. Significant skill gaps ({len(missing_skills)} missing skills) and low semantic similarity ({match_percentage:.1f}%)."
    
    return {
        "match_level": match_level,
        "reasons": reasons,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "summary": summary,
        "skill_match_ratio": round(skill_match_ratio, 1),
        "missing_skills": missing_skills,
        "extra_skills": extra_skills[:10],
        "experience_match": experience_match,
    }


@app.on_event("startup")
async def startup_event():
    """Preload SBERT model at startup."""
    logger.info("Starting ATS Brain service...")
    model = load_sbert_model()
    if model:
        logger.info("SBERT model loaded successfully")
    else:
        logger.warning("SBERT model not available, will use TF-IDF fallback")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API documentation immediately."""
    return RedirectResponse(url="/docs", status_code=307)


@app.post("/match")
async def match_jd_resume(
    # JD Options (choose ONE)
    job_id: Optional[str] = Form(None, description="JD from Database (job_id)"),
    jd_text: Optional[str] = Form(None, description="JD as Text (paste JD here)"),
    jd_file: Optional[Union[UploadFile, str]] = File(None, description="JD as File (upload file)"),
    
    # Resume Options (choose ONE)
    resume_id: Optional[str] = Form(None, description="Resume from Database (resume_id)"),
    resume_file: Optional[Union[UploadFile, str]] = File(None, description="Resume as File (upload file)"),

    # Advanced control
    use_ocr: Optional[bool] = Form(True, description="Use OCR fallback for scanned PDFs when text is short")
):
    """
    # 🎯 Match JD with Resume - ONE Simple Endpoint
    
    **How to use:**
    
    ### Step 1: Provide JD
    Choose ONE option:
    - **From Database:** Enter `job_id` (leave others empty)
    - **Paste Text:** Paste JD text in `jd_text` (leave others empty)
    - **Upload File:** Click "Choose File" for `jd_file` (leave others empty)
    
    ### Step 2: Provide Resume
    Choose ONE option:
    - **From Database:** Enter `resume_id` (leave file empty)
    - **Upload File:** Click "Choose File" for `resume_file` (leave resume_id empty)
    
    ### Step 3: Click Execute
    - See match result immediately!
    - Results are automatically stored in database
    - Job title and candidate name are extracted automatically
    
    **What you get:**
    - ✅ Match percentage (0-100%)
    - ✅ Matched skills list
    - ✅ Detailed analysis (strengths, weaknesses, reasoning)
    - ✅ Match level (HIGH/MODERATE/LOW/VERY_LOW)
    """
    
    try:
        start_time = time.perf_counter()
        # ========================================================================
        # Process JD
        # ========================================================================
        jd_text_final = ""
        jd_title = None
        jd_source = ""
        job_id_final = None
        
        # Handle empty strings/placeholders from form - convert to None
        if job_id and job_id.strip():
            try:
                job_id_int = int(job_id.strip())
            except (ValueError, AttributeError):
                job_id_int = None
        else:
            job_id_int = None

        # Treat Swagger placeholder 'string' or 'null' as empty for jd_text
        if jd_text and jd_text.strip().lower() in {"string", "null"}:
            jd_text = None

        # Normalize JD file input; tolerate empty-string parts from curl/clients
        jd_file_obj: Optional[UploadFile]
        if isinstance(jd_file, str):
            jd_file_obj = None if jd_file.strip() == "" else None
        else:
            jd_file_obj = jd_file if (jd_file and jd_file.filename and jd_file.filename.strip()) else None

        # Validate JD sources: exactly one if provided
        jd_sources_count = int(bool(job_id_int)) + int(bool(jd_text and jd_text.strip())) + int(bool(jd_file_obj))
        if jd_sources_count == 0:
            raise HTTPException(
                status_code=422,
                detail="At least one JD source is required: job_id (database), jd_text (text), or jd_file (file upload)"
            )
        if jd_sources_count > 1:
            raise HTTPException(
                status_code=422,
                detail="Provide only one JD source: use either job_id, jd_text, or jd_file"
            )

        if job_id_int:
            # Option 1: Database
            job = get_job_posting(job_id_int)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job posting with job_id={job_id_int} not found")
            
            jd_text_final = job.get("JDText") or ""
            if not jd_text_final:
                raise HTTPException(status_code=400, detail="Job description text is empty in database")
            
            jd_title = job.get("JobTitle")
            if not jd_title:
                jd_title = extract_job_title_from_text(jd_text_final)
            jd_source = "database"
            job_id_final = job_id_int
            
        elif jd_text and jd_text.strip():
            # Option 2: Text input
            jd_text_final = jd_text.strip()
            if len(jd_text_final) < 50:
                raise HTTPException(status_code=400, detail="JD text is too short (minimum 50 characters)")
            
            jd_title = extract_job_title_from_text(jd_text_final)
            jd_source = "text"
            
        elif jd_file_obj:
            # Option 3: File upload
            file_bytes = await jd_file_obj.read()
            if len(file_bytes) == 0:
                raise HTTPException(status_code=400, detail="JD file is empty")
            
            # Extract text from file
            if jd_file_obj.content_type and "text" in jd_file_obj.content_type:
                jd_text_final = file_bytes.decode('utf-8', errors='ignore')
            else:
                # PDF or DOCX
                try:
                    resume_data = extract_resume_text(file_bytes, use_ocr_fallback=False)
                    jd_text_final = resume_data.get("text", "")
                except Exception:
                    try:
                        jd_text_final = file_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        raise HTTPException(status_code=400, detail="Unable to extract text from JD file")
            
            if not jd_text_final or len(jd_text_final.strip()) < 50:
                raise HTTPException(status_code=400, detail="JD text extraction failed or text is too short")
            
            jd_text_final = jd_text_final.strip()
            jd_title = extract_job_title_from_text(jd_text_final)
            jd_source = "file"
        else:
            # Unreachable due to source count validation, kept for safety
            raise HTTPException(
                status_code=422,
                detail="At least one JD source is required: job_id (database), jd_text (text), or jd_file (file upload)"
            )
        
        # ========================================================================
        # Process Resume
        # ========================================================================
        resume_text_final = ""
        candidate_name = None
        resume_personal: Dict[str, Any] = {}
        resume_source = ""
        resume_id_final = None
        
        # Handle empty strings/placeholders from form - convert to None
        if resume_id and resume_id.strip():
            try:
                resume_id_int = int(resume_id.strip())
            except (ValueError, AttributeError):
                resume_id_int = None
        else:
            resume_id_int = None

        # Treat Swagger placeholder 'string' or 'null' as empty for resume_id
        # (already handled by int conversion), and for completeness normalize resume_file

        # Normalize Resume file input; tolerate empty-string parts from curl/clients
        resume_file_obj: Optional[UploadFile]
        if isinstance(resume_file, str):
            resume_file_obj = None if resume_file.strip() == "" else None
        else:
            resume_file_obj = resume_file if (resume_file and resume_file.filename and resume_file.filename.strip()) else None

        # Validate Resume sources: exactly one required
        resume_sources_count = int(bool(resume_id_int)) + int(bool(resume_file_obj))
        if resume_sources_count == 0:
            raise HTTPException(
                status_code=422,
                detail="At least one Resume source is required: resume_id (database) or resume_file (file upload)"
            )
        if resume_sources_count > 1:
            raise HTTPException(
                status_code=422,
                detail="Provide only one Resume source: use either resume_id or resume_file"
            )

        if resume_id_int:
            # Option 1: Database
            resume_bytes = get_resume_blob(resume_id_int)
            if resume_bytes is None:
                raise HTTPException(status_code=404, detail=f"Resume with resume_id={resume_id_int} not found")
            
            resume_meta = get_resume_metadata(resume_id_int)
            resume_source = "database"
            resume_id_final = resume_id_int
            
            # Extract resume text (retry with OCR if too short)
            resume_data = extract_resume_text(resume_bytes, use_ocr_fallback=False)
            resume_text_final = resume_data.get("text", "")
            if use_ocr and (not resume_text_final or len(resume_text_final.strip()) < 100):
                try:
                    resume_data = extract_resume_text(resume_bytes, use_ocr_fallback=True)
                    resume_text_final = resume_data.get("text", resume_text_final)
                except Exception:
                    pass
            resume_skills = resume_data.get("skills", [])
            resume_personal = resume_data.get("personal_details", {}) or {}
            # Prefer extracting candidate name from original resume text; fallback to DB metadata
            candidate_name = extract_candidate_name_from_text(resume_data.get("raw_text", resume_text_final))
            if not candidate_name and resume_meta:
                candidate_name = resume_meta.get("CandidateName")
            
        elif resume_file_obj:
            # Option 3: File upload
            resume_bytes = await resume_file_obj.read()
            if len(resume_bytes) == 0:
                raise HTTPException(status_code=400, detail="Resume file is empty")
            
            # Extract resume text
            resume_data = extract_resume_text(resume_bytes, use_ocr_fallback=False)
            resume_text_final = resume_data.get("text", "")
            resume_skills = resume_data.get("skills", [])
            resume_personal = resume_data.get("personal_details", {}) or {}
            
            if not resume_text_final or len(resume_text_final.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Resume text extraction failed or extracted text is too short"
                )
            
            candidate_name = extract_candidate_name_from_text(resume_data.get("raw_text", resume_text_final))
            # Fallback: infer candidate name from uploaded filename if not found
            if not candidate_name and resume_file_obj and resume_file_obj.filename:
                base = resume_file_obj.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                name_part = base.rsplit(".", 1)[0]
                # Remove common tokens
                for token in ["resume", "cv", "profile", "updated", "final"]:
                    name_part = re.sub(fr"\b{token}\b", "", name_part, flags=re.IGNORECASE)
                name_part = re.sub(r"[_-]+", " ", name_part).strip()
                # Simple two/three word cap
                words = [w.capitalize() for w in name_part.split() if w]
                if 2 <= len(words) <= 3 and all(w.isalpha() for w in words):
                    candidate_name = " ".join(words)
            resume_source = "file"
        else:
            # Unreachable due to source count validation, kept for safety
            raise HTTPException(
                status_code=422,
                detail="At least one Resume source is required: resume_id (database) or resume_file (file upload)"
            )
        
        # ========================================================================
        # Parse JD
        # ========================================================================
        jd_parsed = parse_job_description(
            jd_text=jd_text_final,
            title=jd_title,
            compute_embedding_vector=True
        )
        
        # Normalize skills and detect required ones
        jd_skills = normalize_skills(jd_parsed.get("skills", []))
        jd_embedding_str = jd_parsed.get("embedding")
        jd_embedding = None
        if jd_embedding_str:
            jd_embedding = deserialize_embedding(jd_embedding_str)
        
        # ========================================================================
        # Compute Similarity and Final Dynamic Score
        # ========================================================================
        resume_embedding, embedding_method = compute_embedding(resume_text_final)
        
        if jd_embedding is not None and resume_embedding is not None:
            similarity_score = cosine_similarity(jd_embedding, resume_embedding)
            method = "sbert" if embedding_method == "sbert" else "tfidf"
        else:
            similarity_score, method = score_texts(jd_text_final, resume_text_final)
        
        semantic_percentage = round(float(similarity_score) * 100.0, 2)
        resume_skills = normalize_skills(resume_skills)
        matched_skills = sorted(list(set(jd_skills) & set(resume_skills)))

        # Weighted skill overlap with required skills boosted
        required_jd_skills = detect_required_skills(jd_text_final, jd_skills)
        def _skill_weight(sk: str) -> float:
            return 2.0 if sk in required_jd_skills else 1.0
        total_weight = sum(_skill_weight(sk) for sk in jd_skills) if jd_skills else 1.0
        matched_weight = sum(_skill_weight(sk) for sk in matched_skills)
        skill_overlap = matched_weight / total_weight
        
        # First pass reasoning to detect experience match
        preliminary_reasoning = _generate_match_reasoning(
            match_percentage=semantic_percentage,
            jd_skills=jd_skills,
            resume_skills=resume_skills,
            matched_skills=matched_skills,
            jd_experience=jd_parsed.get("experience"),
            resume_text=resume_text_final
        )
        exp_bonus = 1.0 if preliminary_reasoning.get("experience_match") else 0.0
        
        # Defer final scoring until education alignment is computed
        
        # (scoring summary and next steps are computed later after final score)

        # ========================================================================
        # Return Response
        # ========================================================================
        # Enriched analytics
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        processing_time_sec = round(time.perf_counter() - start_time, 3)
        model_version = "sbert_all-MiniLM-L6-v2" if method == "sbert" else "tfidf"

        # Textual overlap (unique token coverage of JD by resume)
        def _tokens(s: str):
            return set(t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]+", s.lower()) if len(t) > 2)
        jd_tokens = _tokens(jd_text_final)
        resume_tokens = _tokens(resume_text_final)
        textual_overlap_ratio = round((len(jd_tokens & resume_tokens) / max(1, len(jd_tokens))), 3)

        # Relevant JD sections (sentences containing matched skills)
        jd_sentences = [s.strip() for s in jd_text_final.split(".") if s.strip()]
        key_terms = set(matched_skills[:5]) | set(["python", "sql", "database", "ml", "machine", "analytics"]) & resume_tokens
        relevant_sections = [s for s in jd_sentences if any(k.lower() in s.lower() for k in key_terms)][:3]

        # Experience details (extract numeric years from resume)
        resume_years = parse_years_from_resume(resume_text_final)
        jd_experience_req = jd_parsed.get("experience")

        # Education alignment (robust parsing + ranking)
        req_edu_text = jd_parsed.get("education")
        req_level = extract_education_level(str(req_edu_text)) if req_edu_text else None
        req_discipline = extract_discipline((str(req_edu_text) + " " + jd_text_final) if req_edu_text else jd_text_final)
        cand_level = extract_education_level(resume_text_final)
        cand_discipline = extract_discipline(resume_text_final)
        candidate_education = None
        if cand_level == "phd":
            candidate_education = "PhD"
        elif cand_level == "master":
            candidate_education = "Master"
        elif cand_level == "bachelor":
            candidate_education = "Bachelor"
        elif cand_level == "diploma":
            candidate_education = "Diploma"
        # Education matching rules
        if req_level is None and req_discipline is None:
            education_match = True
        else:
            # Level must satisfy when specified
            level_ok = True
            if req_level is not None:
                level_ok = cand_level is not None and EDU_RANK[cand_level] >= EDU_RANK[req_level]
            # Discipline must satisfy when specified
            disc_ok = True
            if req_discipline is not None:
                disc_ok = (cand_discipline is not None and cand_discipline == req_discipline)
            education_match = bool(level_ok and disc_ok)

        # Compute 4-factor weights and final score now that experience and education are available
        w_sem, w_skill, w_exp, w_edu = compute_auto_weights_4f(
            jd_text_final, jd_parsed, required_jd_skills, req_level, req_discipline, jd_title
        )
        # Factor scores
        experience_score = 1.0 if (preliminary_reasoning.get("experience_match") is True) else 1.0 if (parse_years_requirement(jd_parsed.get("experience")) is None) else 0.0
        education_score = 1.0 if (education_match is True or education_match is True) else (1.0 if (req_level is None and req_discipline is None) else 0.0)
        # Final weighted score
        final_similarity = max(0.0, min(1.0,
            (w_sem * float(similarity_score)) +
            (w_skill * float(skill_overlap)) +
            (w_exp * float(experience_score)) +
            (w_edu * float(education_score))
        ))

        # Penalize missing required JD skills (up to 40% reduction)
        missing_required = list(set(required_jd_skills) - set(matched_skills))
        if missing_required:
            penalty = max(0.6, 1.0 - 0.15 * len(missing_required))
            final_similarity = final_similarity * penalty
        match_percentage = round(final_similarity * 100.0, 2)

        # Final reasoning with dynamic percentage
        reasoning = _generate_match_reasoning(
            match_percentage=match_percentage,
            jd_skills=jd_skills,
            resume_skills=resume_skills,
            matched_skills=matched_skills,
            jd_experience=jd_parsed.get("experience"),
            resume_text=resume_text_final
        )

        # Enrich reasoning with education/discipline and explicit differences
        if education_match is True and (req_level is not None or req_discipline is not None):
            if req_level is not None and candidate_education:
                reasoning.setdefault("strengths", []).append(
                    f"Education meets requirement: {candidate_education} (required: {req_level.title()})"
                )
            if req_discipline is not None and cand_discipline == req_discipline:
                reasoning.setdefault("strengths", []).append(
                    f"Discipline match: {cand_discipline}"
                )
        elif education_match is False:
            # Explicitly note level or discipline mismatches
            if req_level is not None and candidate_education and EDU_RANK.get(extract_education_level(candidate_education.lower()), 0) < EDU_RANK.get(req_level, 0):
                reasoning.setdefault("weaknesses", []).append(
                    f"Education level below requirement: candidate {candidate_education}, required {req_level.title()}"
                )
            if req_discipline is not None and cand_discipline != req_discipline:
                reasoning.setdefault("weaknesses", []).append(
                    f"Discipline mismatch: required {req_discipline}, candidate {cand_discipline or 'unspecified'}"
                )
                reasoning.setdefault("opportunities", []).append(
                    f"Gain exposure/certification in {req_discipline} domain"
                )
        else:
            # JD has no explicit requirement; acknowledge candidate degree when present
            if candidate_education:
                reasoning.setdefault("strengths", []).append(
                    f"Education present: {candidate_education}"
                )

        # Match level mapping: 0–49 LOW, 50–74 MEDIUM, 75–100 HIGH
        if match_percentage >= 75:
            match_level_label = "HIGH"
        elif match_percentage >= 50:
            match_level_label = "MEDIUM"
        else:
            match_level_label = "LOW"
        # Recommendation labels aligned to levels
        if match_percentage >= 75:
            recommendation = "Shortlisted"
        elif match_percentage >= 50:
            recommendation = "Needs Review"
        else:
            recommendation = "Rejection with upskilling advice"

        # Simple confidence estimator
        agreement = 1.0 - abs(float(similarity_score) - (skill_overlap))  # agreement between semantics and skills
        base_conf = 0.5 + 0.5 * max(0.0, min(1.0, agreement))
        if method == "sbert":
            base_conf += 0.1
        base_conf = max(0.0, min(1.0, base_conf))
        confidence_level = "HIGH" if base_conf >= 0.75 else ("MEDIUM" if base_conf >= 0.5 else "LOW")
        automated_decision_confidence = round(base_conf, 2)

        # Map sources for metadata
        data_sources = {
            "jd_source": "dbo.JobPostings" if jd_source == "database" else jd_source,
            "resume_source": "dbo.Resume" if resume_source == "database" else resume_source,
            "storage_table": "dbo.ParsedData"
        }

        # Semantic gap reason (brief HR-readable explanation)
        if similarity_score < 0.5 and skill_overlap >= 0.5:
            semantic_gap_reason = "Resume lists relevant tools, but JD emphasizes role responsibilities and outcomes."
        elif similarity_score >= 0.5 and skill_overlap < 0.4:
            semantic_gap_reason = "JD stresses specific technical skills not sufficiently evidenced in the resume."
        elif similarity_score < 0.5 and skill_overlap < 0.4:
            semantic_gap_reason = "Limited overlap in both responsibilities and key skills between JD and resume."
        else:
            semantic_gap_reason = "Strong alignment between responsibilities and skills."

        # Verdict and next steps (build after we have final score)
        verdict_map = {"HIGH": "Strong match", "MEDIUM": "Moderate match", "LOW": "Low match"}
        verdict = verdict_map.get(match_level_label, match_level_label) + f" ({match_percentage:.1f}%)"

        top_gaps = reasoning.get("missing_skills", [])[:5]
        next_steps = []
        if top_gaps:
            next_steps.append(f"Upskill on: {', '.join(top_gaps)}")
        if reasoning.get("experience_match") is False and jd_parsed.get("experience"):
            next_steps.append("Consider roles matching current experience or highlight relevant projects")
        if not matched_skills and jd_skills:
            next_steps.append("Align resume keywords with JD core skills")

        # Score breakdown (with 4 factors and weights)
        score_breakdown = {
            "final": round(float(final_similarity), 3),
            "factors": {
                "semantic_similarity": {"score": round(float(similarity_score), 3), "weight": round(w_sem, 3)},
                "skill_overlap": {"score": round(float(skill_overlap), 3), "weight": round(w_skill, 3)},
                "experience": {"score": round(float(experience_score), 3), "weight": round(w_exp, 3)},
                "education": {"score": round(float(education_score), 3), "weight": round(w_edu, 3)}
            },
            "required_skill_penalty": (len(missing_required) if 'missing_required' in locals() else 0)
        }

        # Generate match_id now that scoring is complete
        match_id = str(uuid.uuid4())

        # Optional DB writes (after we have final score)
        if settings.db_conn:
            try:
                if jd_source in ["text", "file"]:
                    embedding_bytes = None
                    if jd_embedding is not None:
                        embedding_bytes = jd_embedding.astype(np.float32).tobytes()
                    jd_record = {
                        "Type": "JD",
                        "JobID": None,
                        "ResumeID": None,
                        "ParsedJSON": json.dumps({
                            "type": "JD",
                            "title": jd_title,
                            "experience": jd_parsed.get("experience"),
                            "education": jd_parsed.get("education"),
                            "skills": jd_skills,
                            "summary": jd_parsed.get("summary"),
                            "source": jd_source,
                        }, ensure_ascii=False),
                        "MatchScore": None,
                        "Embedding": embedding_bytes,
                    }
                    insert_parsed_data(jd_record)

                resume_embedding_bytes = None
                if resume_embedding is not None:
                    resume_embedding_bytes = resume_embedding.astype(np.float32).tobytes()

                match_record = {
                    "Type": "Resume",
                    "JobID": job_id_final,
                    "ResumeID": resume_id_final,
                    "ParsedJSON": json.dumps({
                        "type": "Match",
                        "match_id": match_id,
                        "job_id": job_id_final,
                        "resume_id": resume_id_final,
                        "job_title": jd_title,
                        "candidate_name": candidate_name,
                        "contact": {
                            "email": resume_personal.get("email"),
                            "phone": resume_personal.get("phone"),
                            "linkedin": resume_personal.get("linkedin"),
                            "github": resume_personal.get("github"),
                            "website": resume_personal.get("website"),
                            "address": resume_personal.get("address"),
                            "urls": resume_personal.get("urls", []),
                        },
                        "similarity_score": float(final_similarity),
                        "match_percentage": match_percentage,
                        "matched_skills": matched_skills,
                        "jd_skills": jd_skills,
                        "resume_skills": resume_skills,
                        "match_level": match_level_label,
                        "analysis": reasoning,
                        "source": {"jd": jd_source, "resume": resume_source},
                        "method": method,
                        "components": {
                            "semantic": float(similarity_score),
                            "skill_overlap": float(skill_overlap),
                            "experience_score": float(experience_score),
                            "education_score": float(education_score)
                        }
                    }, ensure_ascii=False),
                    "MatchScore": match_percentage,
                    "Embedding": resume_embedding_bytes,
                }
                insert_parsed_data(match_record)
            except Exception as exc:
                logger.error("Failed to store records: {}", exc)
        else:
            logger.info("Skipping DB writes: DB_CONN not configured")

        enriched = {
            "match_id": match_id,
            "timestamp": timestamp,
            "metadata": {
                "job_id": job_id_final,
                "resume_id": resume_id_final,
                "job_title": jd_title,
                "candidate_name": candidate_name,
                "model_version": model_version,
                "processing_time_sec": processing_time_sec,
                "data_sources": data_sources
            },
            "candidate_profile": {
                "contact_details": {
                    "email": resume_personal.get("email"),
                    "phone": resume_personal.get("phone"),
                    "linkedin": resume_personal.get("linkedin"),
                    "github": resume_personal.get("github"),
                    "website": resume_personal.get("website"),
                    "address": resume_personal.get("address"),
                    "urls": resume_personal.get("urls", []),
                }
            },
            "match_summary": {
                "overall_match_score": round(float(final_similarity), 3),
                "match_percentage": match_percentage,
                "match_level": match_level_label,
                "recommendation": recommendation,
                "confidence_level": confidence_level,
                "verdict": verdict
            },
            "scoring_factors": score_breakdown,
            "skill_analysis": {
                "required_skills": jd_skills,
                "candidate_skills": resume_skills,
                "matched_skills": matched_skills,
                "missing_skills": reasoning.get("missing_skills", []),
                "extra_skills": reasoning.get("extra_skills", []),
                "skill_match_ratio": reasoning.get("skill_match_ratio"),
                "skill_coverage_score": round(float(skill_overlap), 3),
                "skill_gap_impact": ("Low" if not reasoning.get("missing_skills") else ("Moderate" if len(reasoning.get("missing_skills", [])) / max(1, len(jd_skills)) <= 0.4 else "High"))
            },
            "experience_alignment": {
                "experience_required": jd_experience_req,
                "experience_detected": (f"{resume_years} years" if resume_years is not None else None),
                "experience_match": (True if parse_years_requirement(jd_experience_req) is None else reasoning.get("experience_match")),
                "gap_description": ("Candidate below required years of experience" if (parse_years_requirement(jd_experience_req) is not None and reasoning.get("experience_match") is False) else None)
            },
            "education_alignment": {
                "required_education": req_edu_text,
                "candidate_education": candidate_education,
                "required_discipline": req_discipline,
                "candidate_discipline": cand_discipline,
                "education_match": education_match
            },
            "reasoning_summary": {
                "strengths": reasoning.get("strengths", []),
                "weaknesses": reasoning.get("weaknesses", []),
                "opportunities": next_steps,
                "semantic_gap_reason": semantic_gap_reason,
                "overall_comment": reasoning.get("summary")
            },
            "final_recommendation": {
                "status": recommendation,
                "next_step": ("Move to interview" if recommendation == "Shortlisted" else ("Manual review by HR/Manager" if recommendation == "Needs Review" else "Send polite rejection with targeted upskilling advice")),
                "action_by": ("Interview Bot / HR Reviewer"),
                "automated_decision_confidence": automated_decision_confidence
            }
        }

        return JSONResponse(content=enriched)
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Match processing failed")
        raise HTTPException(status_code=500, detail=f"Match processing failed: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
