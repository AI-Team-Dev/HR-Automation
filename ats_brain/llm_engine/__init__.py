"""LLM engine package for ATS Brain.

Provides Grok-powered candidate scoring utilities.
"""

from .scoring import score_candidate

__all__ = ["score_candidate"]
