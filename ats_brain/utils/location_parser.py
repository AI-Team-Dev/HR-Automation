from typing import Dict, Any, Optional, List
import re
from loguru import logger

from .location_data import (
    CANON_COUNTRIES,
    INDIAN_STATES,
    INDIAN_CITIES,
    normalize_city,
    normalize_country,
    normalize_state,
)


def _extract_city_state_country(text: str) -> Dict[str, Optional[str]]:
    """Heuristic extraction of city/state/country from a free-form address line.

    This is intentionally conservative to avoid interfering with existing logic.
    """
    if not text:
        return {"city": None, "state": None, "country": None}

    t = " ".join(str(text).strip().lower().split())
    parts = [p.strip() for p in re.split(r"[,;/]", t) if p.strip()]

    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    for part in parts:
        norm = normalize_city(part)
        if norm in INDIAN_CITIES:
            city = norm
        s_norm = normalize_state(part)
        if s_norm in INDIAN_STATES:
            state = s_norm
        c_norm = normalize_country(part)
        if c_norm in CANON_COUNTRIES:
            country = c_norm

    # Reasonable defaults: if city is Indian and no country specified -> India
    if city in INDIAN_CITIES and not country:
        country = "india"

    return {"city": city, "state": state, "country": country}


def extract_candidate_location(resume_text: str, personal_details: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extract normalized candidate location from resume text and personal details.

    This does **not** change existing resume parsing behavior; it is purely
    additive. The output structure is:

    {
      "country": "India" | None,
      "state": "karnataka" | None,
      "city": "bangalore" | None,
      "current_location": str | None,
      "preferred_location": str | None,
      "relocation_preference": str | None,
    }
    """
    addr = (personal_details or {}).get("address")
    base_geo = _extract_city_state_country(addr) if addr else {"city": None, "state": None, "country": None}

    current_location = None
    preferred_location = None
    relocation_pref = None

    text = resume_text or ""

    # Current location label
    m_curr = re.search(r"current\s+location\s*[:\-]\s*([A-Za-z ,\-]{2,120})", text, flags=re.IGNORECASE)
    if m_curr:
        current_location = m_curr.group(1).strip()

    # Preferred location label
    m_pref = re.search(r"preferred\s+location[s]?\s*[:\-]\s*([A-Za-z ,/\-]{2,160})", text, flags=re.IGNORECASE)
    if m_pref:
        preferred_location = m_pref.group(1).strip()

    # Relocation preference
    m_reloc = re.search(r"(willing to relocate[^.\n]*|not willing to relocate[^.\n]*|open to relocation[^.\n]*)", text, flags=re.IGNORECASE)
    if m_reloc:
        relocation_pref = m_reloc.group(1).strip()

    # If current location string exists, refine geo from it
    if current_location:
        geo = _extract_city_state_country(current_location)
        for k, v in geo.items():
            if v and not base_geo.get(k):
                base_geo[k] = v

    # Normalize country/state/city values to title case for output while
    # preserving canonical comparisons internally.
    def _title_or_none(v: Optional[str]) -> Optional[str]:
        return v.title() if isinstance(v, str) and v else None

    return {
        "country": _title_or_none(base_geo.get("country")),
        "state": _title_or_none(base_geo.get("state")),
        "city": _title_or_none(base_geo.get("city")),
        "current_location": current_location,
        "preferred_location": preferred_location,
        "relocation_preference": relocation_pref,
    }


def extract_jd_locations(jd_text: str) -> Dict[str, Any]:
    """Extract normalized location requirements from JD text.

    Output structure:
    {
      "country": "India" | None,
      "states": ["Karnataka", ...],
      "cities": ["Mumbai", ...],
      "strict": bool,           # True if "only" style hard requirement
      "multi_city": bool        # True if explicit list of cities accepted
    }
    """
    if not jd_text:
        return {"country": None, "states": [], "cities": [], "strict": False, "multi_city": False}

    text = " ".join(str(jd_text).split())
    t_lc = text.lower()

    strict = False
    multi_city = False
    cities: List[str] = []
    states: List[str] = []
    country: Optional[str] = None

    # Country-level hints
    if re.search(r"\bpan\s*india\b", t_lc) or re.search(r"\bindia\s+only\b", t_lc):
        country = "india"
    elif re.search(r"\b(india)\b", t_lc):
        country = "india"

    # Patterns like "Looking for candidates from Mumbai only" / "Mumbai location only"
    city_only_patterns = [
        r"candidates?\s+from\s+([A-Za-z ,/]+?)\s+only",
        r"\b([A-Za-z ]+?)\s+location\s+only\b",
        r"only\s+([A-Za-z ,/]+?)\s+candidates",
    ]
    for pat in city_only_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            strict = True
            seg = m.group(1)
            for piece in re.split(r"[,/]|\band\b", seg):
                nm = normalize_city(piece)
                if nm in INDIAN_CITIES:
                    if nm not in cities:
                        cities.append(nm)

    # Patterns like "India (Mumbai, Pune, Chennai, Bangalore)"
    m_list = re.search(r"\b(india)\b\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
    if m_list:
        country = "india"
        multi_city = True
        seg = m_list.group(2)
        for piece in re.split(r"[,/]|\band\b", seg):
            nm = normalize_city(piece)
            if nm in INDIAN_CITIES and nm not in cities:
                cities.append(nm)

    # Generic lists of major cities
    for m in re.finditer(r"\b(mumbai|pune|bangalore|bengaluru|hyderabad|chennai|delhi|noida|gurgaon|gurugram|kolkata)\b", t_lc):
        nm = normalize_city(m.group(1))
        if nm in INDIAN_CITIES and nm not in cities:
            cities.append(nm)

    # Simple state extraction
    for st in INDIAN_STATES:
        if re.search(rf"\b{re.escape(st)}\b", t_lc):
            if st not in states:
                states.append(st)

    def _title_list(xs: List[str]) -> List[str]:
        return sorted({x.title() for x in xs if x})

    return {
        "country": country.title() if country else None,
        "states": _title_list(states),
        "cities": _title_list(cities),
        "strict": bool(strict and cities),
        "multi_city": bool(multi_city and len(cities) > 1),
    }


def matchLocation(jd_locations: Dict[str, Any], candidate_location: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Match JD location requirements against candidate location.

    This function is **pure** and side-effect free. It does not alter any
    existing scores; callers can use the returned score as a separate
    component.
    """
    jd_locations = jd_locations or {}
    candidate_location = candidate_location or {}

    jd_country = (jd_locations.get("country") or "").strip().lower()
    jd_states = [s.strip().lower() for s in jd_locations.get("states", []) if s]
    jd_cities = [c.strip().lower() for c in jd_locations.get("cities", []) if c]
    strict = bool(jd_locations.get("strict"))

    cand_country = (candidate_location.get("country") or "").strip().lower()
    cand_state = (candidate_location.get("state") or "").strip().lower()
    cand_city = (candidate_location.get("city") or "").strip().lower()

    # If JD has no explicit location requirement -> skip scoring
    if not jd_country and not jd_states and not jd_cities:
        return {
            "location_match": True,
            "location_score": None,
            "jd_locations": jd_locations,
            "candidate_location": candidate_location,
            "reason": "No explicit location requirement in JD; location not considered in scoring.",
        }

    # Normalize candidate city for comparison
    cand_city_norm = normalize_city(cand_city) if cand_city else ""

    # A. Strict requirement: city must match exactly
    if strict and jd_cities:
        allowed = {normalize_city(c) for c in jd_cities}
        if cand_city_norm and cand_city_norm in allowed:
            return {
                "location_match": True,
                "location_score": 100,
                "jd_locations": jd_locations,
                "candidate_location": candidate_location,
                "reason": "JD requires specific city only and candidate matches.",
            }
        return {
            "location_match": False,
            "location_score": 0,
            "jd_locations": jd_locations,
            "candidate_location": candidate_location,
            "reason": "JD requires specific city only; candidate city does not match.",
        }

    # B. Multi-city requirement: candidate must match any in the list
    if jd_cities:
        allowed = {normalize_city(c) for c in jd_cities}
        if cand_city_norm and cand_city_norm in allowed:
            return {
                "location_match": True,
                "location_score": 100,
                "jd_locations": jd_locations,
                "candidate_location": candidate_location,
                "reason": "Candidate city matches one of the allowed JD cities.",
            }
        # No city match yet; continue to evaluate state/country levels as softer matches

    # C. State-level match
    if jd_states and cand_state:
        cand_state_norm = normalize_state(cand_state)
        allowed_states = {normalize_state(s) for s in jd_states}
        if cand_state_norm in allowed_states:
            return {
                "location_match": True,
                "location_score": 70,
                "jd_locations": jd_locations,
                "candidate_location": candidate_location,
                "reason": "Candidate state matches JD state requirement.",
            }

    # D. Country/region-level requirement (e.g., PAN India, India only)
    if jd_country:
        cand_country_norm = normalize_country(cand_country)
        jd_country_norm = normalize_country(jd_country)
        if cand_country_norm and cand_country_norm == jd_country_norm:
            return {
                "location_match": True,
                "location_score": 40,
                "jd_locations": jd_locations,
                "candidate_location": candidate_location,
                "reason": "Candidate country matches JD country requirement.",
            }
        # Country mismatch but requirement exists -> hard fail
        return {
            "location_match": False,
            "location_score": 0,
            "jd_locations": jd_locations,
            "candidate_location": candidate_location,
            "reason": f"JD requires candidates from {jd_locations.get('country')}; candidate country is different or unknown.",
        }

    # Fallback: JD had some locations but nothing matched clearly
    return {
        "location_match": False,
        "location_score": 0,
        "jd_locations": jd_locations,
        "candidate_location": candidate_location,
        "reason": "Candidate location does not satisfy the JD location hints.",
    }
