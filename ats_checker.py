"""
ATS-Friendliness Checker
========================

This module analyzes resumes for ATS (Applicant Tracking System) compatibility.
It checks various factors that affect how well a resume will be processed by
automated systems used by employers.

Key Checks:
- Formatting compliance (no tables, images, fancy fonts)
- Required sections presence
- Keyword optimization
- Length and structure
- File format compatibility

Author: AI Resume Evaluator
"""

import re
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckStatus(Enum):
    """Enumeration for check status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

@dataclass
class ATSCheck:
    """Data class for individual ATS checks"""
    name: str
    status: CheckStatus
    score: float  # 0-100
    description: str
    suggestions: List[str]

@dataclass
class ATSReport:
    """Data class for complete ATS analysis report"""
    overall_score: float
    checks: List[ATSCheck]
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    summary: str
    recommendations: List[str]

class ATSChecker:
    """
    Main class for analyzing resume ATS-friendliness.
    
    This class performs comprehensive checks on resume content to ensure
    it will be properly processed by Applicant Tracking Systems.
    """
    
    def __init__(self):
        """Initialize the ATS checker with required data"""
        
        # Standard resume sections that ATS systems expect
        self.required_sections = {
            'contact': ['contact', 'personal', 'header'],
            'summary': ['summary', 'profile', 'objective', 'about'],
            'experience': ['experience', 'work experience', 'employment', 'career'],
            'education': ['education', 'academic', 'qualifications', 'degrees'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise']
        }
        
        # Common ATS-unfriendly elements
        self.ats_unfriendly_patterns = {
            'tables': [
                r'\|.*\|',  # Table separators
                r'┌.*┐',   # Box drawing characters
                r'╔.*╗',   # Double box characters
            ],
            'images': [
                r'\[image\]',  # Image placeholders
                r'<img',       # HTML image tags
                r'\.jpg|\.png|\.gif|\.bmp',  # Image file extensions
            ],
            'fancy_fonts': [
                r'[^\x00-\x7F]',  # Non-ASCII characters
                r'[𝒜-𝒵]',        # Mathematical script characters
                r'[𝔄-𝔜]',        # Fraktur characters
            ],
            'headers_footers': [
                r'page \d+ of \d+',  # Page numbers
                r'header|footer',     # Header/footer text
            ]
        }
        
        # Common resume keywords that ATS systems look for
        self.ats_keywords = {
            'technical': [
                'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'angular',
                'node.js', 'django', 'flask', 'aws', 'azure', 'docker', 'kubernetes',
                'machine learning', 'data analysis', 'artificial intelligence',
                'git', 'github', 'jenkins', 'jira', 'agile', 'scrum'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'collaboration', 'time management',
                'critical thinking', 'adaptability', 'creativity'
            ],
            'action_verbs': [
                'developed', 'implemented', 'managed', 'led', 'created', 'designed',
                'analyzed', 'improved', 'coordinated', 'trained', 'mentored',
                'optimized', 'delivered', 'achieved', 'increased', 'decreased'
            ]
        }
        
        # Optimal resume characteristics
        self.optimal_length = {
            'min_words': 200,
            'max_words': 800,
            'min_pages': 1,
            'max_pages': 2
        }
    
    def check_formatting(self, text: str) -> ATSCheck:
        """
        Check for ATS-unfriendly formatting elements.
        
        Args:
            text (str): Resume text content
            
        Returns:
            ATSCheck: Formatting analysis results
        """
        issues = []
        suggestions = []
        score = 100
        
        # Check for tables
        for pattern in self.ats_unfriendly_patterns['tables']:
            if re.search(pattern, text):
                issues.append("Contains table formatting")
                suggestions.append("Convert tables to simple text format")
                score -= 20
        
        # Check for images
        for pattern in self.ats_unfriendly_patterns['images']:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Contains images or graphics")
                suggestions.append("Remove all images and graphics")
                score -= 25
        
        # Check for fancy fonts
        for pattern in self.ats_unfriendly_patterns['fancy_fonts']:
            if re.search(pattern, text):
                issues.append("Contains non-standard fonts")
                suggestions.append("Use standard fonts like Arial, Calibri, or Times New Roman")
                score -= 15
        
        # Check for headers/footers
        for pattern in self.ats_unfriendly_patterns['headers_footers']:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Contains headers or footers")
                suggestions.append("Remove headers and footers")
                score -= 10
        
        # Determine status
        if score >= 90:
            status = CheckStatus.PASS
        elif score >= 70:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL
        
        description = "Formatting is ATS-friendly" if not issues else f"Found {len(issues)} formatting issues"
        
        return ATSCheck(
            name="Formatting Compliance",
            status=status,
            score=max(0, score),
            description=description,
            suggestions=suggestions
        )
    
    def check_sections(self, sections: List[str]) -> ATSCheck:
        """
        Check if required sections are present in the resume.
        
        Args:
            sections (List[str]): List of identified section names
            
        Returns:
            ATSCheck: Section analysis results
        """
        missing_sections = []
        suggestions = []
        score = 100
        
        # Check for required sections
        for section_type, keywords in self.required_sections.items():
            found = False
            for section in sections:
                if any(keyword in section.lower() for keyword in keywords):
                    found = True
                    break
            
            if not found:
                missing_sections.append(section_type)
                suggestions.append(f"Add a {section_type} section")
                score -= 15
        
        # Determine status
        if score >= 90:
            status = CheckStatus.PASS
        elif score >= 70:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL
        
        description = f"All required sections present" if not missing_sections else f"Missing {len(missing_sections)} sections"
        
        return ATSCheck(
            name="Required Sections",
            status=status,
            score=max(0, score),
            description=description,
            suggestions=suggestions
        )
    
    def check_length(self, text: str) -> ATSCheck:
        """
        Check if resume length is appropriate for ATS systems.
        
        Args:
            text (str): Resume text content
            
        Returns:
            ATSCheck: Length analysis results
        """
        word_count = len(text.split())
        suggestions = []
        score = 100
        
        # Check word count
        if word_count < self.optimal_length['min_words']:
            suggestions.append(f"Resume is too short ({word_count} words). Aim for at least {self.optimal_length['min_words']} words")
            score -= 30
        elif word_count > self.optimal_length['max_words']:
            suggestions.append(f"Resume is too long ({word_count} words). Keep it under {self.optimal_length['max_words']} words")
            score -= 20
        
        # Check for optimal length
        if self.optimal_length['min_words'] <= word_count <= self.optimal_length['max_words']:
            description = f"Optimal length ({word_count} words)"
        else:
            description = f"Length needs adjustment ({word_count} words)"
        
        # Determine status
        if score >= 90:
            status = CheckStatus.PASS
        elif score >= 70:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL
        
        return ATSCheck(
            name="Resume Length",
            status=status,
            score=max(0, score),
            description=description,
            suggestions=suggestions
        )
    
    def check_keywords(self, text: str, job_description: str = "") -> ATSCheck:
        """
        Check for keyword optimization in the resume.
        
        Args:
            text (str): Resume text content
            job_description (str): Job description for keyword matching
            
        Returns:
            ATSCheck: Keyword analysis results
        """
        text_lower = text.lower()
        suggestions = []
        score = 100
        
        # Count keyword categories found
        keyword_counts = {}
        for category, keywords in self.ats_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            keyword_counts[category] = count
        
        # Analyze keyword usage
        total_keywords = sum(keyword_counts.values())
        
        if total_keywords < 5:
            suggestions.append("Add more industry-specific keywords")
            score -= 25
        elif total_keywords < 10:
            suggestions.append("Consider adding more relevant keywords")
            score -= 10
        
        # Check for action verbs
        if keyword_counts.get('action_verbs', 0) < 3:
            suggestions.append("Use more action verbs to describe your achievements")
            score -= 15
        
        # If job description provided, check for matching keywords
        if job_description:
            jd_lower = job_description.lower()
            jd_keywords = set(re.findall(r'\b\w{4,}\b', jd_lower))  # Words with 4+ characters
            
            # Find keywords in resume that match job description
            resume_words = set(re.findall(r'\b\w{4,}\b', text_lower))
            matching_keywords = jd_keywords.intersection(resume_words)
            
            match_percentage = len(matching_keywords) / len(jd_keywords) * 100 if jd_keywords else 0
            
            if match_percentage < 30:
                suggestions.append(f"Low keyword match with job description ({match_percentage:.1f}%). Add more relevant keywords")
                score -= 20
            elif match_percentage < 50:
                suggestions.append(f"Moderate keyword match ({match_percentage:.1f}%). Consider adding more job-specific terms")
                score -= 10
        
        # Determine status
        if score >= 90:
            status = CheckStatus.PASS
        elif score >= 70:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL
        
        description = f"Good keyword optimization ({total_keywords} keywords found)"
        
        return ATSCheck(
            name="Keyword Optimization",
            status=status,
            score=max(0, score),
            description=description,
            suggestions=suggestions
        )
    
    def check_structure(self, text: str) -> ATSCheck:
        """
        Check resume structure and readability for ATS systems.
        
        Args:
            text (str): Resume text content
            
        Returns:
            ATSCheck: Structure analysis results
        """
        suggestions = []
        score = 100
        
        # Check for proper line breaks and formatting
        lines = text.split('\n')
        
        # Check for very long lines (might indicate formatting issues)
        long_lines = sum(1 for line in lines if len(line.strip()) > 100)
        if long_lines > len(lines) * 0.3:  # More than 30% long lines
            suggestions.append("Too many long lines. Use proper line breaks for readability")
            score -= 15
        
        # Check for bullet points
        bullet_points = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '○', '▪')))
        if bullet_points < 5:
            suggestions.append("Use more bullet points to highlight achievements and skills")
            score -= 10
        
        # Check for consistent formatting
        if len(set(len(line.strip()) for line in lines if line.strip())) > len(lines) * 0.8:
            suggestions.append("Inconsistent formatting. Use consistent spacing and alignment")
            score -= 10
        
        # Determine status
        if score >= 90:
            status = CheckStatus.PASS
        elif score >= 70:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL
        
        description = "Good structure and formatting" if score >= 80 else "Structure needs improvement"
        
        return ATSCheck(
            name="Structure & Readability",
            status=status,
            score=max(0, score),
            description=description,
            suggestions=suggestions
        )
    
    def analyze_ats_friendliness(self, parsed_resume: Dict, job_description: str = "") -> ATSReport:
        """
        Perform comprehensive ATS-friendliness analysis.
        
        Args:
            parsed_resume (Dict): Parsed resume data from ResumeParser
            job_description (str): Optional job description for keyword matching
            
        Returns:
            ATSReport: Complete ATS analysis report
        """
        logger.info("Starting ATS-friendliness analysis")
        
        text = parsed_resume.get('raw_text', '')
        sections = [section.title for section in parsed_resume.get('sections', [])]
        
        # Perform all checks
        checks = []
        
        # Formatting check
        formatting_check = self.check_formatting(text)
        checks.append(formatting_check)
        
        # Sections check
        sections_check = self.check_sections(sections)
        checks.append(sections_check)
        
        # Length check
        length_check = self.check_length(text)
        checks.append(length_check)
        
        # Keywords check
        keywords_check = self.check_keywords(text, job_description)
        checks.append(keywords_check)
        
        # Structure check
        structure_check = self.check_structure(text)
        checks.append(structure_check)
        
        # Calculate overall score
        overall_score = sum(check.score for check in checks) / len(checks)
        
        # Count check results
        passed_checks = sum(1 for check in checks if check.status == CheckStatus.PASS)
        failed_checks = sum(1 for check in checks if check.status == CheckStatus.FAIL)
        warnings = sum(1 for check in checks if check.status == CheckStatus.WARNING)
        
        # Generate summary
        if overall_score >= 90:
            summary = "Excellent ATS-friendliness! Your resume should pass through most ATS systems."
        elif overall_score >= 80:
            summary = "Good ATS-friendliness with minor improvements needed."
        elif overall_score >= 70:
            summary = "Moderate ATS-friendliness. Several improvements recommended."
        else:
            summary = "Poor ATS-friendliness. Significant improvements needed to pass ATS systems."
        
        # Compile recommendations
        all_recommendations = []
        for check in checks:
            all_recommendations.extend(check.suggestions)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        logger.info(f"ATS analysis completed. Overall score: {overall_score:.1f}")
        
        return ATSReport(
            overall_score=overall_score,
            checks=checks,
            total_checks=len(checks),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            summary=summary,
            recommendations=unique_recommendations
        )

# Example usage and testing
if __name__ == "__main__":
    # This section runs when the file is executed directly
    # It's useful for testing the ATS checker
    
    checker = ATSChecker()
    
    # Example usage
    # parsed_resume = {...}  # From ResumeParser
    # job_description = "Software Engineer position..."
    # report = checker.analyze_ats_friendliness(parsed_resume, job_description)
    # print(f"ATS Score: {report.overall_score:.1f}")
    
    print("ATS Checker module loaded successfully!")
    print("To use this checker, import it in your main application.") 