"""
Utility Functions
================

This module contains utility functions used throughout the resume evaluator
application. It includes text processing, data validation, formatting,
and other helper functions.

Key Functions:
- Text cleaning and normalization
- Data validation
- Score formatting and display
- File handling utilities
- Color coding for scores

Author: AI Resume Evaluator
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file is a valid PDF.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not file_path.lower().endswith('.pdf'):
        return False, "File is not a PDF"
    
    # Check file size (max 10MB)
    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024:  # 10MB
        return False, "File is too large (max 10MB)"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "File is valid"

def format_score(score: float) -> str:
    """
    Format a score for display with color coding.
    
    Args:
        score (float): Score value (0-100)
        
    Returns:
        str: Formatted score string
    """
    if score >= 90:
        return f"🟢 {score:.1f}/100 (Excellent)"
    elif score >= 80:
        return f"🟡 {score:.1f}/100 (Good)"
    elif score >= 70:
        return f"🟠 {score:.1f}/100 (Fair)"
    elif score >= 60:
        return f"🔴 {score:.1f}/100 (Poor)"
    else:
        return f"🔴 {score:.1f}/100 (Very Poor)"

def get_score_color(score: float) -> str:
    """
    Get color for score display.
    
    Args:
        score (float): Score value (0-100)
        
    Returns:
        str: Color name for display
    """
    if score >= 90:
        return "green"
    elif score >= 80:
        return "orange"
    elif score >= 70:
        return "yellow"
    elif score >= 60:
        return "red"
    else:
        return "darkred"

def format_percentage(value: float) -> str:
    """
    Format a percentage value for display.
    
    Args:
        value (float): Percentage value (0-1)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.1f}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_file_name(file_path: str) -> str:
    """
    Extract file name from file path.
    
    Args:
        file_path (str): Full file path
        
    Returns:
        str: File name
    """
    return os.path.basename(file_path)

def create_summary_report(ats_report: Any, jd_report: Any, scoring_result: Any) -> Dict[str, Any]:
    """
    Create a summary report combining all analysis results.
    
    Args:
        ats_report: ATS analysis report
        jd_report: Job description matching report
        scoring_result: Resume scoring result
        
    Returns:
        Dict[str, Any]: Summary report
    """
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'overall_assessment': {},
        'scores': {},
        'recommendations': [],
        'priority_actions': []
    }
    
    # Overall assessment
    ats_score = ats_report.overall_score if hasattr(ats_report, 'overall_score') else 0
    jd_score = jd_report.overall_similarity if hasattr(jd_report, 'overall_similarity') else 0
    quality_score = scoring_result.overall_score if hasattr(scoring_result, 'overall_score') else 0
    
    avg_score = (ats_score + jd_score * 100 + quality_score) / 3
    
    if avg_score >= 85:
        summary['overall_assessment']['status'] = "Excellent"
        summary['overall_assessment']['message'] = "Your resume is well-optimized and should perform well in ATS systems."
    elif avg_score >= 75:
        summary['overall_assessment']['status'] = "Good"
        summary['overall_assessment']['message'] = "Your resume is good but could benefit from some improvements."
    elif avg_score >= 65:
        summary['overall_assessment']['status'] = "Fair"
        summary['overall_assessment']['message'] = "Your resume needs several improvements to be competitive."
    else:
        summary['overall_assessment']['status'] = "Poor"
        summary['overall_assessment']['message'] = "Your resume requires significant improvements to pass ATS systems."
    
    # Scores
    summary['scores'] = {
        'ats_friendliness': format_score(ats_score),
        'job_match': format_score(jd_score * 100),
        'overall_quality': format_score(quality_score),
        'average_score': format_score(avg_score)
    }
    
    # Compile recommendations
    all_recommendations = []
    
    if hasattr(ats_report, 'recommendations'):
        all_recommendations.extend(ats_report.recommendations)
    
    if hasattr(jd_report, 'suggestions'):
        all_recommendations.extend(jd_report.suggestions)
    
    if hasattr(scoring_result, 'suggestions'):
        all_recommendations.extend(scoring_result.suggestions)
    
    # Remove duplicates while preserving order
    unique_recommendations = []
    for rec in all_recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
    
    summary['recommendations'] = unique_recommendations
    
    # Priority actions (top 5 most important)
    summary['priority_actions'] = unique_recommendations[:5]
    
    return summary

def format_check_status(status: str) -> str:
    """
    Format check status for display.
    
    Args:
        status (str): Status value
        
    Returns:
        str: Formatted status with emoji
    """
    status_map = {
        'PASS': '✅ PASS',
        'FAIL': '❌ FAIL',
        'WARNING': '⚠️ WARNING'
    }
    
    return status_map.get(status, status)

def create_detailed_breakdown(parsed_resume: Dict) -> Dict[str, Any]:
    """
    Create a detailed breakdown of resume content.
    
    Args:
        parsed_resume (Dict): Parsed resume data
        
    Returns:
        Dict[str, Any]: Detailed breakdown
    """
    breakdown = {
        'contact_info': {},
        'sections': [],
        'skills': [],
        'experience': [],
        'statistics': {}
    }
    
    # Contact information
    if 'contact_info' in parsed_resume:
        contact = parsed_resume['contact_info']
        breakdown['contact_info'] = {
            'name': contact.name or 'Not found',
            'email': contact.email or 'Not found',
            'phone': contact.phone or 'Not found',
            'location': contact.location or 'Not found'
        }
    
    # Sections
    if 'sections' in parsed_resume:
        breakdown['sections'] = [
            {
                'title': section.title,
                'content_preview': truncate_text(section.content, 150)
            }
            for section in parsed_resume['sections']
        ]
    
    # Skills
    if 'skills' in parsed_resume:
        breakdown['skills'] = parsed_resume['skills']
    
    # Experience
    if 'experience' in parsed_resume:
        breakdown['experience'] = [
            {
                'title': exp.get('title', 'Unknown'),
                'description_preview': truncate_text(exp.get('description', ''), 100)
            }
            for exp in parsed_resume['experience']
        ]
    
    # Statistics
    if 'raw_text' in parsed_resume:
        text = parsed_resume['raw_text']
        breakdown['statistics'] = {
            'total_characters': len(text),
            'total_words': len(text.split()),
            'total_sentences': len(text.split('.')),
            'total_paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'skills_count': len(parsed_resume.get('skills', [])),
            'experience_entries': len(parsed_resume.get('experience', [])),
            'sections_count': len(parsed_resume.get('sections', []))
        }
    
    return breakdown

def validate_job_description(text: str) -> Tuple[bool, str]:
    """
    Validate job description text.
    
    Args:
        text (str): Job description text
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Job description is empty"
    
    if len(text.strip()) < 50:
        return False, "Job description is too short (minimum 50 characters)"
    
    if len(text.strip()) > 10000:
        return False, "Job description is too long (maximum 10,000 characters)"
    
    return True, "Job description is valid"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def create_error_message(error: Exception) -> str:
    """
    Create a user-friendly error message.
    
    Args:
        error (Exception): The error that occurred
        
    Returns:
        str: User-friendly error message
    """
    error_type = type(error).__name__
    
    if "FileNotFoundError" in error_type:
        return "The file was not found. Please check the file path and try again."
    elif "PermissionError" in error_type:
        return "Permission denied. Please check file permissions and try again."
    elif "ValueError" in error_type:
        return f"Invalid input: {str(error)}"
    elif "RuntimeError" in error_type:
        return f"An error occurred while processing: {str(error)}"
    else:
        return f"An unexpected error occurred: {str(error)}"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone (str): Phone number to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check if it's a reasonable length (7-15 digits)
    return 7 <= len(digits_only) <= 15

def create_progress_message(current: int, total: int, message: str = "Processing") -> str:
    """
    Create a progress message.
    
    Args:
        current (int): Current step
        total (int): Total steps
        message (str): Base message
        
    Returns:
        str: Progress message
    """
    percentage = (current / total) * 100
    return f"{message}... {percentage:.0f}% ({current}/{total})"

# Example usage and testing
if __name__ == "__main__":
    # This section runs when the file is executed directly
    # It's useful for testing the utility functions
    
    print("Utility functions module loaded successfully!")
    print("Available functions:")
    print("- clean_text()")
    print("- validate_pdf_file()")
    print("- format_score()")
    print("- truncate_text()")
    print("- validate_email()")
    print("- validate_phone()")
    print("And many more...")
    
    # Test some functions
    test_text = "  This   is   a   test   text   "
    cleaned = clean_text(test_text)
    print(f"Clean text example: '{cleaned}'")
    
    test_score = 85.5
    formatted = format_score(test_score)
    print(f"Score formatting example: {formatted}")
    
    print("To use these utilities, import them in your main application.") 