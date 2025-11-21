# Education field to sector normalization

from typing import Optional


EDUCATION_SECTOR_KEYWORDS = {
    "Computer Science / IT": [
        "computer science",
        "information technology",
        "it",
        "computer engineering",
        "software engineering",
        "information systems",
        "data science",
        "artificial intelligence",
        "ai",
        "machine learning",
        "ml",
        "aiml",
        "data analytics",
        "analytics",
        "cyber security",
        "cybersecurity",
        "cloud computing",
    ],
    "Finance": [
        "finance",
        "financial management",
        "accounting & finance",
        "accounting and finance",
        "financial analytics",
        "fintech",
        "investment banking",
        "capital markets",
        "corporate finance",
    ],
    "Banking": [
        "banking",
        "banking & insurance",
        "banking and insurance",
        "financial services",
        "bfsi",
        "risk management",
        "audit & compliance",
        "audit and compliance",
    ],
    "Commerce": [
        "commerce",
        "bcom",
        "b.com",
        "bachelor of commerce",
        "mcom",
        "m.com",
        "master of commerce",
        "accounting",
        "taxation",
        "business administration",
        "business management",
    ],
    "Management": [
        "mba",
        "business administration",
        "operations management",
        "hr management",
        "human resource management",
        "marketing management",
        "supply chain management",
        "project management",
    ],
    "HR": [
        "human resources",
        "human resource",
        "hrm",
        "talent management",
        "organizational psychology",
        "industrial relations",
    ],
    "Marketing": [
        "marketing",
        "digital marketing",
        "advertising",
        "brand management",
        "communications",
        "market research",
    ],
    "Engineering": [
        "mechanical engineering",
        "electrical engineering",
        "civil engineering",
        "electronics engineering",
        "instrumentation",
        "robotics",
    ],
    "Healthcare": [
        "mbbs",
        "nursing",
        "pharmacy",
        "physiotherapy",
        "biomedical science",
        "public health",
    ],
    "Arts & Humanities": [
        "arts",
        "humanities",
        "psychology",
        "sociology",
        "political science",
        "literature",
    ],
    "Science": [
        "physics",
        "chemistry",
        "biology",
        "mathematics",
        "biotechnology",
        "environmental science",
    ],
    "Hospitality / Travel": [
        "hospitality management",
        "hotel management",
        "travel & tourism",
        "travel and tourism",
    ],
    "Law": [
        "llb",
        "ll.m",
        "llm",
        "legal studies",
        "corporate law",
    ],
}


def mapEducationToSector(text: Optional[str]) -> Optional[str]:
    """Normalize raw education/degree text into a broad sector label.

    Returns a canonical sector name (e.g. "Computer Science / IT") or None
    if no reasonable mapping is found. Matching is case-insensitive and based
    on substring presence, so minor title variations don't break mapping.
    """
    if not text:
        return None

    t = " ".join(str(text).lower().split())

    for sector, keywords in EDUCATION_SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw and kw in t:
                return sector

    return None
