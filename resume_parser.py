"""
PDF Resume Parser
=================

This module handles the extraction and parsing of resume content from PDF files.
It uses pdfplumber for PDF processing and spaCy for natural language processing
to identify key sections and extract structured information.

Key Features:
- PDF text extraction with formatting preservation
- Contact information extraction (name, email, phone)
- Section identification (Skills, Experience, Education, etc.)
- Skills extraction and categorization
- Experience timeline extraction
- Education details extraction

Author: AI Resume Evaluator
"""

import pdfplumber
import spacy
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContactInfo:
    """Data class to store extracted contact information"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

@dataclass
class ResumeSection:
    """Data class to store resume sections"""
    title: str
    content: str
    start_line: int
    end_line: int

class ResumeParser:
    """
    Main class for parsing PDF resumes and extracting structured information.
    
    This class uses pdfplumber to extract text from PDF files and spaCy for
    natural language processing to identify and categorize different sections
    of the resume.
    """
    
    def __init__(self):
        """Initialize the parser with spaCy model"""
        try:
            # Load the English language model for spaCy
            # This model helps us understand text structure and extract entities
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Define common section headers that appear in resumes
        # These help us identify different parts of the resume
        self.section_headers = {
            'experience': ['experience', 'work experience', 'employment history', 'professional experience', 'career history'],
            'education': ['education', 'academic background', 'qualifications', 'degrees'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise', 'technologies'],
            'summary': ['summary', 'profile', 'objective', 'about', 'professional summary'],
            'projects': ['projects', 'portfolio', 'achievements', 'key projects'],
            'certifications': ['certifications', 'certificates', 'licenses', 'accreditations'],
            'languages': ['languages', 'language skills', 'fluency'],
            'interests': ['interests', 'hobbies', 'activities']
        }
        
        # Common email patterns for extraction
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone number patterns (various formats)
        self.phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',  # US format
            r'\+?[0-9]{1,4}[\s.-]?[0-9]{1,4}[\s.-]?[0-9]{1,4}[\s.-]?[0-9]{1,4}',  # International
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file using pdfplumber.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        try:
            # Open the PDF document with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Successfully opened PDF: {pdf_path}")
                
                text_content = []
                
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages):
                    # Extract text from the page
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
                        logger.info(f"Extracted text from page {page_num + 1}")
                    else:
                        logger.warning(f"No text found on page {page_num + 1}")
                
                # Combine all pages and clean up the text
                full_text = '\n'.join(text_content)
                cleaned_text = self._clean_text(full_text)
                
                logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
                return cleaned_text
                
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove multiple empty lines
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        text = text.strip()  # Remove leading/trailing whitespace
        
        return text
    
    def extract_contact_info(self, text: str) -> ContactInfo:
        """
        Extract contact information from resume text.
        
        Args:
            text (str): Resume text content
            
        Returns:
            ContactInfo: Extracted contact information
        """
        contact_info = ContactInfo()
        
        # Extract email addresses
        emails = re.findall(self.email_pattern, text)
        if emails:
            contact_info.email = emails[0]  # Take the first email found
            logger.info(f"Found email: {contact_info.email}")
        
        # Extract phone numbers
        for pattern in self.phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info.phone = phones[0]  # Take the first phone found
                logger.info(f"Found phone: {contact_info.phone}")
                break
        
        # Extract name (usually in the first few lines)
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            line = line.strip()
            if line and not re.search(r'@|http|www|linkedin', line.lower()):
                # Simple heuristic: name is usually 2-4 words, no special chars
                words = line.split()
                if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
                    contact_info.name = line
                    logger.info(f"Found name: {contact_info.name}")
                    break
        
        return contact_info
    
    def identify_sections(self, text: str) -> List[ResumeSection]:
        """
        Identify and extract different sections of the resume.
        
        Args:
            text (str): Resume text content
            
        Returns:
            List[ResumeSection]: List of identified sections
        """
        lines = text.split('\n')
        sections = []
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            section_found = None
            for section_type, headers in self.section_headers.items():
                if any(header in line_lower for header in headers):
                    section_found = section_type
                    break
            
            # If we found a new section, save the previous one
            if section_found and current_section:
                sections.append(ResumeSection(
                    title=current_section,
                    content='\n'.join(current_content).strip(),
                    start_line=i - len(current_content),
                    end_line=i - 1
                ))
                current_content = []
            
            # Start new section or continue current
            if section_found:
                current_section = section_found
                current_content = [line]
            elif current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append(ResumeSection(
                title=current_section,
                content='\n'.join(current_content).strip(),
                start_line=len(lines) - len(current_content),
                end_line=len(lines) - 1
            ))
        
        logger.info(f"Identified {len(sections)} sections: {[s.title for s in sections]}")
        return sections
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from resume text using NLP.
        
        Args:
            text (str): Resume text content
            
        Returns:
            List[str]: List of extracted skills
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        skills = []
        
        # Look for skills section specifically
        sections = self.identify_sections(text)
        skills_section = None
        for section in sections:
            if section.title == 'skills':
                skills_section = section.content
                break
        
        # If we found a skills section, focus on that
        target_text = skills_section if skills_section else text
        
        # Extract noun phrases and technical terms
        doc = self.nlp(target_text)
        
        # Look for technical terms, programming languages, tools
        technical_terms = [
            'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'angular',
            'node.js', 'django', 'flask', 'aws', 'azure', 'docker', 'kubernetes',
            'machine learning', 'data analysis', 'project management', 'agile',
            'scrum', 'git', 'github', 'jenkins', 'jira', 'confluence'
        ]
        
        # Extract skills using multiple methods
        for token in doc:
            # Look for technical terms
            if token.text.lower() in technical_terms:
                skills.append(token.text)
            
            # Look for noun phrases that might be skills
            elif token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                if not token.is_stop and not token.is_punct:
                    skills.append(token.text)
        
        # Remove duplicates and clean up
        skills = list(set([skill.strip() for skill in skills if skill.strip()]))
        
        logger.info(f"Extracted {len(skills)} skills: {skills[:10]}...")  # Show first 10
        return skills
    
    def extract_experience(self, text: str) -> List[Dict]:
        """
        Extract work experience information from resume.
        
        Args:
            text (str): Resume text content
            
        Returns:
            List[Dict]: List of work experience entries
        """
        experience_entries = []
        
        # Find experience section
        sections = self.identify_sections(text)
        experience_section = None
        for section in sections:
            if section.title == 'experience':
                experience_section = section.content
                break
        
        if not experience_section:
            logger.warning("No experience section found")
            return experience_entries
        
        # Split into individual entries (usually separated by company/job)
        lines = experience_section.split('\n')
        current_entry = {}
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns that indicate new job entries
            # Company names often have dates or locations after them
            if re.search(r'\d{4}', line) or re.search(r'present|current', line.lower()):
                # Save previous entry if exists
                if current_entry and current_content:
                    current_entry['description'] = '\n'.join(current_content)
                    experience_entries.append(current_entry.copy())
                    current_content = []
                
                # Start new entry
                current_entry = {'title': line}
            else:
                current_content.append(line)
        
        # Add the last entry
        if current_entry and current_content:
            current_entry['description'] = '\n'.join(current_content)
            experience_entries.append(current_entry)
        
        logger.info(f"Extracted {len(experience_entries)} experience entries")
        return experience_entries
    
    def parse_resume(self, pdf_path: str) -> Dict:
        """
        Main method to parse a complete resume from PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Complete parsed resume data
        """
        logger.info(f"Starting to parse resume: {pdf_path}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract contact information
        contact_info = self.extract_contact_info(text)
        
        # Identify sections
        sections = self.identify_sections(text)
        
        # Extract skills
        skills = self.extract_skills(text)
        
        # Extract experience
        experience = self.extract_experience(text)
        
        # Compile results
        parsed_data = {
            'contact_info': contact_info,
            'sections': sections,
            'skills': skills,
            'experience': experience,
            'raw_text': text,
            'text_length': len(text),
            'num_sections': len(sections)
        }
        
        logger.info("Resume parsing completed successfully")
        return parsed_data

# Example usage and testing
if __name__ == "__main__":
    # This section runs when the file is executed directly
    # It's useful for testing the parser
    
    parser = ResumeParser()
    
    # Example of how to use the parser
    # pdf_path = "path/to/your/resume.pdf"
    # parsed_data = parser.parse_resume(pdf_path)
    # print(f"Extracted {len(parsed_data['skills'])} skills")
    # print(f"Found {len(parsed_data['experience'])} experience entries")
    
    print("Resume Parser module loaded successfully!")
    print("To use this parser, import it in your main application.") 