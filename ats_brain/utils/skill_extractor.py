"""Skill extraction using spaCy PhraseMatcher with multi-word skill support and synonyms."""
from typing import List, Set, Dict, Optional
import re
from loguru import logger

# Try to import spaCy, but handle gracefully if not available
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, using simple keyword matching")


# Comprehensive skill dictionary with synonyms and multi-word skills
SKILL_DICTIONARY = {
    # Programming Languages
    "python": ["python", "python3", "py"],
    "java": ["java", "java8", "java11", "j2ee"],
    "javascript": ["javascript", "js", "ecmascript", "node.js", "nodejs"],
    "typescript": ["typescript", "ts"],
    "c#": ["c#", "csharp", "dotnet", ".net"],
    "c++": ["c++", "cpp", "c plus plus"],
    "go": ["go", "golang"],
    "rust": ["rust"],
    "ruby": ["ruby", "rails"],
    "php": ["php"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
    
    # Frameworks & Libraries
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "spring": ["spring", "spring boot", "springboot"],
    "express": ["express", "expressjs"],
    
    # Data & ML
    "pandas": ["pandas"],
    "numpy": ["numpy", "np"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "spacy": ["spacy", "spaCy"],
    "nltk": ["nltk"],
    
    # Databases
    "sql": ["sql", "structured query language"],
    "postgresql": ["postgresql", "postgres", "pg"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "oracle": ["oracle", "oracle db"],
    "sql server": ["sql server", "mssql", "sqlserver"],
    
    # Cloud & DevOps
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "git": ["git", "gitlab", "github"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    
    # Other Technologies
    "rest": ["rest", "rest api", "restful"],
    "graphql": ["graphql", "gql"],
    "microservices": ["microservices", "micro services"],
    "api": ["api", "apis"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "sass": ["sass", "scss"],
    "linux": ["linux", "ubuntu", "centos"],
    "bash": ["bash", "shell scripting"],
    "powershell": ["powershell", "ps"],
}


# Global spaCy model cache
_spacy_model = None

def _load_spacy_model() -> Optional[object]:
    """Load spaCy English model, cached globally."""
    global _spacy_model
    if _spacy_model is None and SPACY_AVAILABLE:
        try:
            _spacy_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            _spacy_model = None
        except Exception as e:
            logger.warning("Failed to load spaCy model: {}", e)
            _spacy_model = None
    return _spacy_model


def extract_skills_spacy(text: str) -> List[str]:
    """
    Extract skills using spaCy PhraseMatcher for phrase-aware matching.
    
    Args:
        text: Input text to extract skills from
        
    Returns:
        List of canonical skill names found in text
    """
    if not SPACY_AVAILABLE:
        return []
    
    nlp = _load_spacy_model()
    if nlp is None:
        return []
    
    try:
        doc = nlp(text.lower())
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        
        # Add patterns for each skill and its synonyms
        patterns = {}
        for canonical_skill, synonyms in SKILL_DICTIONARY.items():
            for synonym in synonyms:
                pattern = nlp.make_doc(synonym)
                matcher.add(canonical_skill, [pattern])
                patterns[canonical_skill] = True
        
        matches = matcher(doc)
        found_skills = set()
        
        for match_id, start, end in matches:
            canonical_skill = nlp.vocab.strings[match_id]
            found_skills.add(canonical_skill)
        
        return sorted(list(found_skills))
    except Exception as e:
        logger.warning("spaCy skill extraction failed: {}, falling back to simple matching", e)
        return []


def extract_skills_simple(text: str) -> List[str]:
    """
    Simple keyword-based skill extraction (fallback when spaCy unavailable).
    
    Args:
        text: Input text
        
    Returns:
        List of canonical skill names
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_skills = set()
    
    # Normalize text for matching (remove punctuation, normalize whitespace)
    normalized_text = re.sub(r"[^\w\s]", " ", text_lower)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    
    # Check each skill and its synonyms
    for canonical_skill, synonyms in SKILL_DICTIONARY.items():
        for synonym in synonyms:
            # Use word boundaries for better matching
            pattern = r"\b" + re.escape(synonym.lower()) + r"\b"
            if re.search(pattern, normalized_text, re.IGNORECASE):
                found_skills.add(canonical_skill)
                break  # Found this skill, no need to check other synonyms
    
    return sorted(list(found_skills))


def extract_skills(text: str, use_spacy: bool = True) -> List[str]:
    """
    Extract skills from text using spaCy PhraseMatcher if available, otherwise simple matching.
    
    Args:
        text: Input text to extract skills from
        use_spacy: Whether to prefer spaCy (default: True)
        
    Returns:
        List of canonical skill names found in text
    """
    if not text:
        return []
    
    if use_spacy and SPACY_AVAILABLE:
        skills = extract_skills_spacy(text)
        if skills:
            return skills
    
    # Fallback to simple matching
    return extract_skills_simple(text)

