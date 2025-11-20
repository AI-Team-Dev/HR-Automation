from typing import Set, Dict

# Canonical countries (minimal for now; can be extended)
CANON_COUNTRIES: Set[str] = {
    "india",
    "united states",
    "usa",
    "united kingdom",
    "uk",
    "canada",
    "australia",
    "germany",
    "singapore",
    "uae",
    "dubai",
}

# Canonical Indian states (lowercase)
INDIAN_STATES: Set[str] = {
    "maharashtra",
    "karnataka",
    "tamil nadu",
    "telangana",
    "andhra pradesh",
    "delhi",
    "ncr",
    "gujarat",
    "west bengal",
    "uttar pradesh",
    "rajasthan",
    "haryana",
    "punjab",
    "kerala",
    "madhya pradesh",
    "bihar",
    "odisha",
    "chhattisgarh",
}

# Canonical Indian cities (lowercase)
INDIAN_CITIES: Set[str] = {
    "mumbai",
    "bombay",
    "pune",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "chennai",
    "delhi",
    "new delhi",
    "noida",
    "greater noida",
    "gurgaon",
    "gurugram",
    "ghaziabad",
    "faridabad",
    "kolkata",
    "ahmedabad",
    "surat",
    "jaipur",
    "indore",
    "nagpur",
    "coimbatore",
    "kochi",
    "trivandrum",
    "thiruvananthapuram",
    "vijayawada",
    "vizag",
    "visakhapatnam",
    "lucknow",
    "kanpur",
}

# Common abbreviations / variants to canonical city names (lowercase keys & values)
CITY_ABBREVIATIONS: Dict[str, str] = {
    "blr": "bangalore",
    "bengaluru": "bangalore",
    "banglore": "bangalore",
    "bengaluru, karnataka": "bangalore",
    "mum": "mumbai",
    "bby": "mumbai",
    "bom": "mumbai",
    "delhi ncr": "delhi",
    "ncr": "delhi",
}


def normalize_city(name: str) -> str:
    """Normalize a city name to a canonical lowercase form.

    This handles common abbreviations like "Blr" -> "bangalore" and
    variant spellings / combined forms.
    """
    if not name:
        return ""
    t = " ".join(str(name).strip().lower().split())
    if t in CITY_ABBREVIATIONS:
        return CITY_ABBREVIATIONS[t]
    # Strip country/state suffixes for matching (e.g., "mumbai, india")
    for sep in [", india", ", maharashtra", ", karnataka", ", tamil nadu", ", telangana"]:
        if t.endswith(sep):
            t = t[: -len(sep)].strip()
            break
    return t


def normalize_country(name: str) -> str:
    if not name:
        return ""
    t = " ".join(str(name).strip().lower().split())
    if t in {"india", "in"}:
        return "india"
    if t in {"united states", "usa", "us"}:
        return "united states"
    if t in {"united kingdom", "uk"}:
        return "united kingdom"
    return t


def normalize_state(name: str) -> str:
    if not name:
        return ""
    t = " ".join(str(name).strip().lower().split())
    # Simple aliases
    if t in {"tn"}:
        return "tamil nadu"
    if t in {"mp"}:
        return "madhya pradesh"
    if t in {"up"}:
        return "uttar pradesh"
    return t
