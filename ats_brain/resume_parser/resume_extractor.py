"""Resume text extraction from PDF and DOCX with OCR fallback support."""
from typing import Dict, Any, Optional
import io
from loguru import logger
import re

from utils.text_cleaner import normalize_text
from utils.skill_extractor import extract_skills
from utils.location_parser import extract_candidate_location


def _is_pdf(data: bytes) -> bool:
    """Check if bytes represent a PDF file."""
    return data.startswith(b"%PDF")


def _is_docx(data: bytes) -> bool:
    """Check if bytes represent a DOCX file (ZIP-based format)."""
    return len(data) >= 2 and data[:2] == b"PK"


def _extract_pdf_text(data: bytes) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    
    Args:
        data: PDF file bytes
        
    Returns:
        Extracted text string
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF extraction. Install with: pip install PyMuPDF")
    
    text_parts = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
        doc.close()
    except Exception as e:
        logger.error("PDF extraction failed: {}", e)
        raise
    
    combined_text = "\n".join(text_parts)
    
    # Check if extracted text is too short (possible scanned PDF)
    if len(combined_text.strip()) < 100:
        logger.warning("PDF extracted text is very short ({} chars), may be scanned. Consider OCR.", len(combined_text.strip()))
    
    return combined_text


def _extract_pdf_with_ocr(data: bytes) -> str:
    """
    Extract text from PDF using OCR (pytesseract) as fallback for scanned PDFs.
    This is optional and requires pytesseract and Tesseract OCR installed.
    
    Args:
        data: PDF file bytes
        
    Returns:
        Extracted text string from OCR
    """
    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "OCR dependencies not installed. Install with: "
            "pip install pytesseract pillow PyMuPDF"
            "And install Tesseract OCR binary: https://github.com/tesseract-ocr/tesseract"
        )
    
    text_parts = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(img)
            text_parts.append(ocr_text)
        doc.close()
    except Exception as e:
        logger.error("PDF OCR extraction failed: {}", e)
        raise
    
    return "\n".join(text_parts)


def _extract_docx_text(data: bytes) -> str:
    """
    Extract text from DOCX file using docx2txt.
    
    Args:
        data: DOCX file bytes
        
    Returns:
        Extracted text string
    """
    try:
        import docx2txt
    except ImportError:
        raise ImportError("docx2txt is required for DOCX extraction. Install with: pip install docx2txt")
    
    try:
        bio = io.BytesIO(data)
        text = docx2txt.process(bio)
        return text or ""
    except Exception as e:
        logger.error("DOCX extraction failed: {}", e)
        raise


def extract_resume_text(resume_bytes: bytes, use_ocr_fallback: bool = False) -> Dict[str, Any]:
    """
    Extract text from resume (PDF or DOCX) and extract skills.
    
    Args:
        resume_bytes: Raw bytes of the resume file (PDF or DOCX)
        use_ocr_fallback: If True and PDF extraction yields little text, try OCR (requires pytesseract)
        
    Returns:
        Dictionary with keys:
            - text: Extracted and normalized text
            - skills: List of extracted skills
            - file_type: "pdf" or "docx"
            - extraction_method: "direct" or "ocr"
    
    Raises:
        ValueError: If resume_bytes is empty or file type is unsupported
        ImportError: If required libraries are missing
    """
    if not resume_bytes:
        return {
            "text": "",
            "skills": [],
            "file_type": None,
            "extraction_method": None,
        }
    
    text = ""
    file_type = None
    extraction_method = "direct"
    
    try:
        if _is_pdf(resume_bytes):
            file_type = "pdf"
            text = _extract_pdf_text(resume_bytes)
            
            # Check if text extraction yielded very little text (likely scanned PDF)
            if use_ocr_fallback and len(text.strip()) < 100:
                logger.info("PDF text extraction yielded little text, attempting OCR fallback")
                try:
                    text = _extract_pdf_with_ocr(resume_bytes)
                    extraction_method = "ocr"
                except ImportError:
                    logger.warning("OCR not available, using extracted text as-is")
                except Exception as e:
                    logger.warning("OCR failed: {}, using extracted text as-is", e)
            
        elif _is_docx(resume_bytes):
            file_type = "docx"
            text = _extract_docx_text(resume_bytes)
        else:
            # Try PDF first, then DOCX
            try:
                text = _extract_pdf_text(resume_bytes)
                file_type = "pdf"
            except Exception:
                try:
                    text = _extract_docx_text(resume_bytes)
                    file_type = "docx"
                except Exception as e:
                    raise ValueError(f"Unsupported file format. Failed to extract as PDF or DOCX: {e}")
    
    except Exception as e:
        logger.exception("Resume extraction failed")
        raise
    
    # Preserve original text for name extraction
    original_text = text
    # Normalize and clean text
    normalized_text = normalize_text(text)

    # Extract skills
    skills = extract_skills(normalized_text)

    # Extract personal/contact details on original text to better preserve header structure
    personal_details = _extract_personal_details(original_text)

    # Derive a normalized candidate location structure in an additive way,
    # based on the resume text and detected personal details (e.g. address
    # or explicit location fields). This does not change existing parsing
    # behavior and is safe to ignore by callers that do not need it.
    candidate_location = extract_candidate_location(original_text, personal_details)

    return {
        "text": normalized_text,
        "raw_text": original_text,
        "skills": skills,
        "file_type": file_type,
        "extraction_method": extraction_method,
        "personal_details": personal_details,
        "candidate_location": candidate_location,
    }


def _extract_personal_details(text: str) -> Dict[str, Any]:
    """Extract email, phone, LinkedIn, GitHub, website/portfolio, and address heuristically.

    Returns keys:
      - email: str | None
      - phone: str | None
      - linkedin: str | None
      - github: str | None
      - website: str | None
      - address: str | None
      - urls: list[str]
    """
    if not text:
        return {
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "website": None,
            "address": None,
            "urls": [],
        }

    # Email
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    emails = email_pattern.findall(text)
    email = emails[0] if emails else None

    # Phone numbers (normalize and pick one with 10-13 digits)
    phone_candidates = re.findall(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}", text)
    def _norm_phone(p: str) -> str:
        return re.sub(r"[^\d+]", "", p).lstrip("+")
    phones_norm = []
    for p in phone_candidates:
        d = re.sub(r"\D", "", p)
        if 10 <= len(d) <= 13:
            phones_norm.append(p.strip())
    phone = phones_norm[0] if phones_norm else None

    # URLs
    url_pattern = re.compile(r"https?://[A-Za-z0-9./_#%?=&+-]+", re.IGNORECASE)
    urls = list(dict.fromkeys(url_pattern.findall(text)))  # unique while preserving order
    linkedin = next((u for u in urls if "linkedin.com" in u.lower()), None)
    github = next((u for u in urls if "github.com" in u.lower()), None)
    website = next((u for u in urls if ("linkedin.com" not in u.lower() and "github.com" not in u.lower())), None)

    # Address heuristics
    countries = {
        "india","usa","united states","uk","united kingdom","canada","australia",
        "germany","singapore","uae","dubai"
    }
    states = {
        "maharashtra","karnataka","tamil nadu","telangana","delhi","ncr","gujarat",
        "west bengal","uttar pradesh","rajasthan","haryana","punjab","kerala"
    }
    cities = {
        "mumbai","pune","bengaluru","bangalore","hyderabad","chennai","delhi",
        "new delhi","noida","gurgaon","gurugram","kolkata","ahmedabad","jaipur"
    }

    def _has_geo(s: str) -> bool:
        low = s.lower()
        return any(w in low for w in countries | states | cities)

    def _looks_like_phone_email(s: str) -> bool:
        return re.search(r"(phone|email|github|linkedin|https?://|@)", s, flags=re.IGNORECASE) is not None

    def _has_pincode(s: str) -> bool:
        return re.search(r"\b\d{5,6}\b", s) is not None
    
    def _looks_like_sentence(s: str) -> bool:
        # crude heuristic: presence of common verbs or ends with period and multiple words
        verbs = r"taught|developed|built|designed|implemented|maintained|led|managed|conducted|mentored|worked|participated|created|collaborated|performed"
        if re.search(rf"\b({verbs})\b", s, flags=re.IGNORECASE):
            return True
        if s.endswith('.') and len(s.split()) > 6:
            return True
        return False
    def _sanitize_address(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = re.sub(r"\s+", " ", s).strip()
        cut_terms = [
            "profile", "professional", "summary", "experience", "education",
            "projects", "technical", "skills", "contact", "languages",
            "hobbies", "interests", "phone", "email", "github", "linkedin",
            "website", "portfolio"
        ]
        seps = ["|", "•", "●", "□", "■", "▪", "·"]
        cut_idx = len(s)
        for sep in seps:
            if sep in s:
                cut_idx = min(cut_idx, s.find(sep))
        for term in cut_terms:
            m = re.search(rf"\b{term}\b", s, flags=re.IGNORECASE)
            if m:
                cut_idx = min(cut_idx, m.start())
        s = s[:cut_idx].strip()
        s = re.sub(r"^(?:address|location|current location)[:\-\s]*", "", s, flags=re.IGNORECASE).strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 3:
            parts = parts[:3]
        s = ", ".join(parts)
        s = re.sub(r"[^A-Za-z0-9,\-\s]", "", s).strip()
        if len(s) > 80:
            s = s[:80].rstrip(",- ")
        # Reject if line is likely noise or lacks geo/pincode
        if _looks_like_phone_email(s) or _looks_like_sentence(s):
            return None
        if not (_has_geo(s) or _has_pincode(s)):
            return None
        return s if 2 <= len(s) <= 80 else None
    
    def _is_label_line(ln: str, targets: list[str]) -> Optional[str]:
        raw = ln.strip()
        compact = re.sub(r"[^A-Za-z]", "", raw).lower()
        for t in targets:
            if compact.startswith(t.replace(" ", "").lower()):
                return raw
        return None
    
    def _after_label(raw: str) -> str:
        if ":" in raw:
            return raw.split(":", 1)[1]
        if "-" in raw:
            return raw.split("-", 1)[1]
        return raw
    address = None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Try explicit label
    for i, ln in enumerate(lines):
        if ln.lower().startswith("address") or _is_label_line(ln, ["address"]):
            parts = [ln]
            if i + 1 < len(lines):
                parts.append(lines[i + 1])
            if i + 2 < len(lines):
                parts.append(lines[i + 2])
            cleaned = []
            for p in parts:
                p2 = re.sub(r"^(address[:\-\s]*)", "", p, flags=re.IGNORECASE).strip()
                cleaned.append(p2)
            address = ", ".join([c for c in cleaned if c])
            address = _sanitize_address(address)
            break
    if not address:
        for i, ln in enumerate(lines):
            if (ln.lower().startswith("location") or ln.lower().startswith("current location") or
                _is_label_line(ln, ["location", "current location"])):
                parts = [ln]
                if i + 1 < len(lines):
                    parts.append(lines[i + 1])
                cleaned = []
                for p in parts:
                    p2 = re.sub(r"^(?:location|current location)[:\-\s]*", "", p, flags=re.IGNORECASE).strip()
                    if p2 == p:
                        p2 = _after_label(p2)
                    cleaned.append(p2)
                address = ", ".join([c for c in cleaned if c])
                address = _sanitize_address(address)
                if address:
                    break
    # Inline 'Location:' anywhere in the header text (not just line-start), including spaced letters
    if not address:
        m = re.search(r"(?:\b[Ll]\s*[Oo]\s*[Cc]\s*[Aa]\s*[Tt]\s*[Ii]\s*[Oo]\s*[Nn]\b|\bcurrent\s*location\b)\s*[:\-]\s*([A-Za-z ,\-]{2,120})", text, flags=re.IGNORECASE)
        if m:
            address = _sanitize_address(m.group(1).strip())
    # Try postal code typical patterns if not found
    if not address:
        for ln in lines[:15]:  # header region
            # avoid phone/email lines, and ensure line has comma or geo hint with the pincode
            if re.search(r"\b\d{5,6}\b", ln):
                if re.search(r"(phone|email|github|linkedin|https?://)", ln, flags=re.IGNORECASE):
                    continue
                cand = _sanitize_address(ln)
                if cand:
                    address = cand
                    break
    if not address:
        country_hint = re.compile(r"\b(india|usa|united states|uk|canada|australia|singapore|germany|uae|dubai)\b", re.IGNORECASE)
        for ln in lines[:20]:
            if country_hint.search(ln):
                if not re.search(r"(phone|email|github|linkedin|https?://)", ln, flags=re.IGNORECASE):
                    cand = _sanitize_address(ln.strip())
                    if cand:
                        address = cand
                        break
    if not address:
        for ln in lines[:20]:
            if _looks_like_phone_email(ln):
                continue
            if _has_geo(ln):
                cand = _sanitize_address(ln)
                if cand:
                    address = cand
                    break

    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "website": website,
        "address": address,
        "urls": urls,
    }
