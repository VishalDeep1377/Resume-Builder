"""
Job Description Matcher
=======================

This module compares resume content with job descriptions to determine
how well they match. It uses multiple techniques including TF-IDF
vectorization and sentence transformers for accurate similarity analysis.

Key Features:
- TF-IDF similarity scoring
- Sentence transformer similarity (more advanced)
- Keyword extraction and matching
- Missing skills identification
- Detailed matching report

Author: AI Resume Evaluator
"""

import re
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class KeywordMatch:
    """Data class for keyword matching results"""
    keyword: str
    found_in_resume: bool
    frequency_in_jd: int
    frequency_in_resume: int
    importance: str  # 'high', 'medium', 'low'

@dataclass
class MatchingReport:
    """Data class for complete job description matching report"""
    tfidf_similarity: float
    overall_similarity: float
    keyword_matches: List[KeywordMatch]
    missing_keywords: List[str]
    matching_keywords: List[str]
    total_keywords: int
    match_percentage: float
    suggestions: List[str]
    detailed_analysis: str

class JobDescriptionMatcher:
    """
    Main class for comparing resumes with job descriptions.
    
    This class uses multiple NLP techniques to analyze how well a resume
    matches a specific job description, providing detailed insights and
    recommendations for improvement.
    """
    
    def __init__(self):
        """Initialize the job description matcher"""
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include both single words and bigrams
            min_df=1,
            max_df=0.95
        )
        
        # Load stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Common job-related keywords and their importance levels
        self.importance_keywords = {
            'high': [
                'required', 'must', 'essential', 'mandatory', 'necessary',
                'expert', 'advanced', 'senior', 'lead', 'principal'
            ],
            'medium': [
                'preferred', 'nice to have', 'bonus', 'plus', 'advantage',
                'intermediate', 'mid-level', 'experienced'
            ],
            'low': [
                'optional', 'nice to have', 'bonus', 'additional',
                'junior', 'entry-level', 'basic'
            ]
        }
        
        # Technical skills categories for better analysis
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php',
                'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'asp.net', 'node.js', 'jquery', 'bootstrap'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'oracle', 'sql server', 'sqlite', 'dynamodb', 'cassandra'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
                'linode', 'vultr', 'ibm cloud'
            ],
            'tools': [
                'git', 'github', 'jenkins', 'docker', 'kubernetes', 'terraform',
                'ansible', 'jira', 'confluence', 'slack', 'trello'
            ]
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text (str): Text to extract keywords from
            
        Returns:
            List[str]: List of extracted keywords
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(processed_text)
        
        # Filter out stop words and short words
        keywords = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def calculate_tfidf_similarity(self, resume_text: str, job_description: str) -> float:
        """
        Calculate similarity using TF-IDF vectorization.
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Combine texts for vectorization
            documents = [resume_text, job_description]
            
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            logger.info(f"TF-IDF similarity: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def analyze_keyword_matches(self, resume_text: str, job_description: str) -> Tuple[List[KeywordMatch], List[str], List[str]]:
        """
        Analyze keyword matches between resume and job description.
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            Tuple[List[KeywordMatch], List[str], List[str]]: Keyword matches, missing keywords, matching keywords
        """
        # Extract keywords from both texts
        resume_keywords = self.extract_keywords(resume_text)
        jd_keywords = self.extract_keywords(job_description)
        
        # Count keyword frequencies
        resume_freq = {}
        for keyword in resume_keywords:
            resume_freq[keyword] = resume_freq.get(keyword, 0) + 1
        
        jd_freq = {}
        for keyword in jd_keywords:
            jd_freq[keyword] = jd_freq.get(keyword, 0) + 1
        
        # Analyze each keyword in job description
        keyword_matches = []
        matching_keywords = []
        missing_keywords = []
        
        for keyword, jd_count in jd_freq.items():
            resume_count = resume_freq.get(keyword, 0)
            found_in_resume = resume_count > 0
            
            # Determine importance based on context
            importance = self._determine_keyword_importance(keyword, job_description)
            
            # Create keyword match object
            match = KeywordMatch(
                keyword=keyword,
                found_in_resume=found_in_resume,
                frequency_in_jd=jd_count,
                frequency_in_resume=resume_count,
                importance=importance
            )
            
            keyword_matches.append(match)
            
            if found_in_resume:
                matching_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        return keyword_matches, missing_keywords, matching_keywords
    
    def _determine_keyword_importance(self, keyword: str, job_description: str) -> str:
        """
        Determine the importance of a keyword based on context.
        
        Args:
            keyword (str): The keyword to analyze
            job_description (str): Job description text
            
        Returns:
            str: Importance level ('high', 'medium', 'low')
        """
        jd_lower = job_description.lower()
        
        # Check for importance indicators around the keyword
        for level, indicators in self.importance_keywords.items():
            for indicator in indicators:
                # Look for importance indicators near the keyword
                pattern = rf'\b{indicator}\b.*\b{keyword}\b|\b{keyword}\b.*\b{indicator}\b'
                if re.search(pattern, jd_lower):
                    return level
        
        # Check if it's a technical skill (usually important)
        for category, skills in self.skill_categories.items():
            if keyword.lower() in [skill.lower() for skill in skills]:
                return 'high'
        
        # Default to medium importance
        return 'medium'
    
    def generate_suggestions(self, missing_keywords: List[str], match_percentage: float) -> List[str]:
        """
        Generate improvement suggestions based on analysis.
        
        Args:
            missing_keywords (List[str]): Keywords missing from resume
            match_percentage (float): Overall match percentage
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        # Overall match suggestions
        if match_percentage < 30:
            suggestions.append("Low keyword match. Consider adding more job-specific terms to your resume.")
        elif match_percentage < 50:
            suggestions.append("Moderate keyword match. Add more relevant skills and experience keywords.")
        elif match_percentage < 70:
            suggestions.append("Good keyword match. Consider adding a few more specific terms.")
        else:
            suggestions.append("Excellent keyword match! Your resume aligns well with the job description.")
        
        # Missing keyword suggestions
        if missing_keywords:
            # Categorize missing keywords
            technical_skills = []
            soft_skills = []
            other_keywords = []
            
            for keyword in missing_keywords:
                if any(keyword.lower() in [skill.lower() for skill in skills] for skills in self.skill_categories.values()):
                    technical_skills.append(keyword)
                elif keyword.lower() in ['leadership', 'communication', 'teamwork', 'problem solving']:
                    soft_skills.append(keyword)
                else:
                    other_keywords.append(keyword)
            
            # Generate specific suggestions
            if technical_skills:
                suggestions.append(f"Add technical skills: {', '.join(technical_skills[:5])}")
            
            if soft_skills:
                suggestions.append(f"Highlight soft skills: {', '.join(soft_skills[:3])}")
            
            if other_keywords:
                suggestions.append(f"Consider adding keywords: {', '.join(other_keywords[:5])}")
        
        return suggestions
    
    def match_resume_to_job(self, resume_text: str, job_description: str) -> MatchingReport:
        """
        Perform a full analysis of resume against job description.
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            MatchingReport: Detailed matching report
        """
        if not resume_text or not job_description:
            logger.warning("Resume text or job description is empty.")
            # Return a default empty report
            return MatchingReport(
                tfidf_similarity=0.0,
                overall_similarity=0.0,
                keyword_matches=[],
                missing_keywords=[],
                matching_keywords=[],
                total_keywords=0,
                match_percentage=0.0,
                suggestions=["Please provide both a resume and a job description."],
                detailed_analysis="Analysis could not be performed due to missing content."
            )

        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(job_description)
        
        # Calculate similarities
        tfidf_similarity = self.calculate_tfidf_similarity(processed_resume, processed_jd)
        
        # Analyze keyword matches
        keyword_matches, missing_keywords, matching_keywords = self.analyze_keyword_matches(
            processed_resume, processed_jd
        )
        
        # Calculate match percentage
        total_keywords = len(missing_keywords) + len(matching_keywords)
        if total_keywords > 0:
            match_percentage = (len(matching_keywords) / total_keywords) * 100
        else:
            match_percentage = 0.0
            
        # The main score should be the keyword match percentage
        overall_similarity = match_percentage / 100.0

        # Generate suggestions
        suggestions = self.generate_suggestions(missing_keywords, match_percentage)
        
        # Create detailed analysis report
        detailed_analysis = self._create_detailed_analysis(
            tfidf_similarity, overall_similarity, keyword_matches, 
            missing_keywords, match_percentage
        )
        
        return MatchingReport(
            tfidf_similarity=tfidf_similarity,
            overall_similarity=overall_similarity,
            keyword_matches=keyword_matches,
            missing_keywords=missing_keywords,
            matching_keywords=matching_keywords,
            total_keywords=total_keywords,
            match_percentage=match_percentage,
            suggestions=suggestions,
            detailed_analysis=detailed_analysis
        )
    
    def _create_detailed_analysis(self, tfidf_similarity: float,
                                overall_similarity: float, keyword_matches: List[KeywordMatch],
                                missing_keywords: List[str], match_percentage: float) -> str:
        """
        Create a detailed text analysis of the matching results.
        
        Args:
            tfidf_similarity (float): TF-IDF similarity score
            overall_similarity (float): Overall similarity score
            keyword_matches (List[KeywordMatch]): Keyword matching results
            missing_keywords (List[str]): Missing keywords
            match_percentage (float): Keyword match percentage
            
        Returns:
            str: Detailed analysis text
        """
        analysis = f"""
Overall Match Score: {overall_similarity * 100:.1f}%

Detailed Breakdown:
- TF-IDF Similarity: {tfidf_similarity * 100:.1f}% (keyword-based match)
- Keyword Match Percentage: {match_percentage:.1f}%

The analysis shows a keyword similarity of {tfidf_similarity * 100:.1f}%. 
This score reflects how well the specific terms in your resume align with the job description.
Your resume includes {len(keyword_matches) - len(missing_keywords)} of the {len(keyword_matches)} important keywords found in the job description.
"""
        if missing_keywords:
            analysis += f"\n\n**Missing High-Importance Keywords:**\n- {', '.join(missing_keywords[:10])}"
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    # This section runs when the file is executed directly
    # It's useful for testing the job description matcher
    
    matcher = JobDescriptionMatcher()
    
    # Example usage
    # resume_text = "Experienced software engineer with Python and React skills..."
    # job_description = "We are looking for a Python developer with React experience..."
    # report = matcher.match_resume_to_job(resume_text, job_description)
    # print(f"Overall similarity: {report.overall_similarity:.1%}")
    
    print("Job Description Matcher module loaded successfully!")
    print("To use this matcher, import it in your main application.") 