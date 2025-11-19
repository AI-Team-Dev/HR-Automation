"""Date utilities for consistent date handling across the HR Automation System."""

from datetime import datetime
from typing import Optional


def get_current_date_str() -> str:
    """Get the current date as a formatted string.
    
    Returns:
        The current date in format "Month DD, YYYY" (e.g., "November 19, 2025")
    """
    return datetime.now().strftime("%B %d, %Y")


def get_current_year() -> int:
    """Get the current year.
    
    Returns:
        The current year as an integer
    """
    return datetime.now().year


def get_current_month_year() -> str:
    """Get the current month and year.
    
    Returns:
        The current month and year in format "Month YYYY" (e.g., "November 2025")
    """
    return datetime.now().strftime("%B %Y")


def calculate_experience_duration(start_date: str, end_date: Optional[str] = None) -> str:
    """Calculate experience duration from start date to end date or current date.
    
    Args:
        start_date: Start date string (various formats supported)
        end_date: End date string or None for current date
    
    Returns:
        Duration string in format "X years Y months"
    """
    # This is a placeholder - actual implementation would parse various date formats
    # and calculate the exact duration
    if end_date is None or end_date.lower() in ['present', 'current', 'till date', 'now', 'ongoing']:
        end_date = datetime.now()
    
    # For now, return a simple string
    # In production, this would do actual date parsing and calculation
    return "Duration calculation requires date parsing implementation"


def is_current_employment(date_str: str) -> bool:
    """Check if a date string indicates current employment.
    
    Args:
        date_str: Date string to check
    
    Returns:
        True if the date indicates current/ongoing employment
    """
    if not date_str:
        return False
    
    current_indicators = [
        'present', 'current', 'till date', 'now', 'ongoing', 
        'till now', 'to date', 'to present', 'till present'
    ]
    
    return any(indicator in date_str.lower() for indicator in current_indicators)
