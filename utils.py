"""
Utility Functions
================

This module contains utility functions used throughout the resume evaluator
application. It includes formatting and reporting helpers.

Key Functions:
- Score formatting and display
- Summary report generation

Author: AI Resume Evaluator
"""

import logging
from typing import Any, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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