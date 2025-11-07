"""Text normalization and cleaning utilities."""
import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize and clean text:
    - Remove extra whitespace
    - Fix line endings (CRLF -> LF)
    - Remove excessive newlines
    - Fix encoding issues (basic cleanup)
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove excessive spaces (more than 2 consecutive)
    text = re.sub(r" {3,}", "  ", text)
    
    # Remove tabs and replace with spaces
    text = text.replace("\t", " ")
    
    # Remove leading/trailing whitespace from each line
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)
    
    # Final cleanup: remove any remaining excessive whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing whitespace
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    if not text:
        return []
    
    # Simple sentence splitting (can be enhanced with spaCy if needed)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences
