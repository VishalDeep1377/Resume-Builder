"""
Resume Scoring Model
====================

This module implements a machine learning model to score resume quality.
It trains on simulated resume data and provides predictions based on
various features like experience, skills, formatting, and content quality.

Key Features:
- Feature extraction from resume data
- ML model training (XGBoost and Logistic Regression)
- Quality scoring (0-100)
- Feature importance analysis
- Model persistence and loading

Author: AI Resume Evaluator
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import textstat
import re

# Use lightgbm as the primary model
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.warning("LightGBM not available. Using alternative models.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResumeFeatures:
    """Data class for extracted resume features"""
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    readability_score: float
    skills_count: int
    experience_years: float
    education_level: int
    has_contact_info: bool
    has_summary: bool
    has_experience: bool
    has_education: bool
    has_skills: bool
    bullet_points_count: int
    action_verbs_count: int
    quantifiable_achievements: int
    keyword_density: float
    formatting_score: float

@dataclass
class ScoringResult:
    """Data class for resume scoring results"""
    overall_score: float
    feature_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    confidence: float
    breakdown: Dict[str, str]
    suggestions: List[str]

class ResumeScoringModel:
    """
    Machine learning model for scoring resume quality.
    
    This class trains a model on simulated resume data and provides
    quality predictions based on various resume features.
    """
    
    def __init__(self):
        """Initialize the scoring model"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'text_length', 'word_count', 'sentence_count', 'paragraph_count',
            'avg_sentence_length', 'readability_score', 'skills_count',
            'experience_years', 'education_level', 'has_contact_info',
            'has_summary', 'has_experience', 'has_education', 'has_skills',
            'bullet_points_count', 'action_verbs_count', 'quantifiable_achievements',
            'keyword_density', 'formatting_score'
        ]
        
        # Action verbs that indicate strong achievements
        self.action_verbs = [
            'achieved', 'increased', 'decreased', 'improved', 'developed',
            'implemented', 'managed', 'led', 'created', 'designed',
            'optimized', 'delivered', 'reduced', 'enhanced', 'streamlined',
            'coordinated', 'trained', 'mentored', 'established', 'launched'
        ]
        
        # Quantifiable terms that indicate measurable achievements
        self.quantifiable_terms = [
            '%', 'percent', 'increase', 'decrease', 'growth', 'reduction',
            'million', 'thousand', 'dollars', '$', 'users', 'customers',
            'team', 'employees', 'projects', 'revenue', 'sales', 'efficiency'
        ]
    
    def generate_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate simulated training data for the resume scoring model.
        
        Args:
            num_samples (int): Number of training samples to generate
            
        Returns:
            pd.DataFrame: Training data with features and target scores
        """
        logger.info(f"Generating {num_samples} training samples")
        
        data = []
        
        for i in range(num_samples):
            # Generate realistic feature values
            features = self._generate_sample_features()
            
            # Calculate target score based on features
            target_score = self._calculate_target_score(features)
            
            # Add target to features
            features['quality_score'] = target_score
            data.append(features)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated training data with shape: {df.shape}")
        return df
    
    def _generate_sample_features(self) -> Dict:
        """Generate a single sample of resume features"""
        # Text characteristics
        text_length = np.random.randint(500, 3000)
        word_count = text_length // 5  # Rough estimate
        sentence_count = word_count // 15
        paragraph_count = sentence_count // 3
        
        # Readability and structure
        avg_sentence_length = np.random.uniform(10, 25)
        readability_score = np.random.uniform(30, 80)
        
        # Content features
        skills_count = np.random.randint(5, 25)
        experience_years = np.random.uniform(0, 15)
        education_level = np.random.randint(1, 5)  # 1=HS, 2=Associate, 3=Bachelor, 4=Master, 5=PhD
        
        # Section presence
        has_contact_info = np.random.choice([True, False], p=[0.9, 0.1])
        has_summary = np.random.choice([True, False], p=[0.7, 0.3])
        has_experience = np.random.choice([True, False], p=[0.95, 0.05])
        has_education = np.random.choice([True, False], p=[0.9, 0.1])
        has_skills = np.random.choice([True, False], p=[0.8, 0.2])
        
        # Formatting features
        bullet_points_count = np.random.randint(5, 50)
        action_verbs_count = np.random.randint(3, 20)
        quantifiable_achievements = np.random.randint(0, 10)
        keyword_density = np.random.uniform(0.01, 0.05)
        formatting_score = np.random.uniform(60, 95)
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability_score,
            'skills_count': skills_count,
            'experience_years': experience_years,
            'education_level': education_level,
            'has_contact_info': has_contact_info,
            'has_summary': has_summary,
            'has_experience': has_experience,
            'has_education': has_education,
            'has_skills': has_skills,
            'bullet_points_count': bullet_points_count,
            'action_verbs_count': action_verbs_count,
            'quantifiable_achievements': quantifiable_achievements,
            'keyword_density': keyword_density,
            'formatting_score': formatting_score
        }
    
    def _calculate_target_score(self, features: Dict) -> float:
        """
        Calculate target quality score based on features.
        
        Args:
            features (Dict): Resume features
            
        Returns:
            float: Quality score (0-100)
        """
        score = 50  # Base score
        
        # Text quality (20 points)
        if 1000 <= features['text_length'] <= 2000:
            score += 10
        elif 800 <= features['text_length'] <= 2500:
            score += 5
        
        if 30 <= features['readability_score'] <= 70:
            score += 10
        elif 20 <= features['readability_score'] <= 80:
            score += 5
        
        # Content completeness (25 points)
        if features['has_contact_info']:
            score += 5
        if features['has_summary']:
            score += 5
        if features['has_experience']:
            score += 5
        if features['has_education']:
            score += 5
        if features['has_skills']:
            score += 5
        
        # Experience and skills (20 points)
        if features['experience_years'] >= 2:
            score += 10
        elif features['experience_years'] >= 1:
            score += 5
        
        if features['skills_count'] >= 10:
            score += 10
        elif features['skills_count'] >= 5:
            score += 5
        
        # Formatting and achievements (15 points)
        if features['bullet_points_count'] >= 15:
            score += 5
        elif features['bullet_points_count'] >= 8:
            score += 3
        
        if features['action_verbs_count'] >= 8:
            score += 5
        elif features['action_verbs_count'] >= 4:
            score += 3
        
        if features['quantifiable_achievements'] >= 3:
            score += 5
        elif features['quantifiable_achievements'] >= 1:
            score += 3
        
        # Education bonus (10 points)
        if features['education_level'] >= 4:
            score += 10
        elif features['education_level'] >= 3:
            score += 5
        
        # Add some randomness to make it more realistic
        score += np.random.normal(0, 5)
        
        return max(0, min(100, score))
    
    def extract_features(self, parsed_resume: Dict) -> ResumeFeatures:
        """
        Extract features from parsed resume data.
        
        Args:
            parsed_resume (Dict): Parsed resume data from ResumeParser
            
        Returns:
            ResumeFeatures: Extracted features
        """
        text = parsed_resume.get('raw_text', '')
        
        # Basic text statistics
        text_length = len(text)
        word_count = len(text.split())
        sentence_count = len(text.split('.'))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Readability
        try:
            readability_score = textstat.flesch_reading_ease(text)
        except:
            readability_score = 50  # Default if calculation fails
        
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Skills and experience
        skills_count = len(parsed_resume.get('skills', []))
        experience_entries = parsed_resume.get('experience', [])
        experience_years = len(experience_entries) * 2  # Rough estimate
        
        # Education level (simplified)
        education_level = 3  # Default to Bachelor's level
        
        # Section presence
        sections = [section.title for section in parsed_resume.get('sections', [])]
        has_contact_info = any('contact' in section.lower() for section in sections)
        has_summary = any('summary' in section.lower() for section in sections)
        has_experience = any('experience' in section.lower() for section in sections)
        has_education = any('education' in section.lower() for section in sections)
        has_skills = any('skills' in section.lower() for section in sections)
        
        # Formatting features
        bullet_points_count = text.count('•') + text.count('-') + text.count('*')
        
        # Action verbs count
        action_verbs_count = sum(1 for verb in self.action_verbs if verb.lower() in text.lower())
        
        # Quantifiable achievements
        quantifiable_achievements = sum(1 for term in self.quantifiable_terms if term.lower() in text.lower())
        
        # Keyword density (simplified)
        keyword_density = len(set(text.lower().split())) / word_count if word_count > 0 else 0
        
        # Formatting score (simplified)
        formatting_score = 80  # Default score
        
        return ResumeFeatures(
            text_length=text_length,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score,
            skills_count=skills_count,
            experience_years=experience_years,
            education_level=education_level,
            has_contact_info=has_contact_info,
            has_summary=has_summary,
            has_experience=has_experience,
            has_education=has_education,
            has_skills=has_skills,
            bullet_points_count=bullet_points_count,
            action_verbs_count=action_verbs_count,
            quantifiable_achievements=quantifiable_achievements,
            keyword_density=keyword_density,
            formatting_score=formatting_score
        )
    
    def train_model(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the resume scoring model.
        
        Args:
            data (Optional[pd.DataFrame]): Training data, generates if not provided
        """
        if data is None:
            data = self.generate_training_data()
        
        # Prepare features and target
        X = data[self.feature_names]
        y = data['quality_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and choose the best one
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        best_score = -1
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            
            logger.info(f"{name} R² Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        logger.info(f"Selected {type(best_model).__name__} as best model with R² score: {best_score:.3f}")
    
    def predict_score(self, features: ResumeFeatures) -> ScoringResult:
        """
        Predict resume quality score.
        
        Args:
            features (ResumeFeatures): Resume features
            
        Returns:
            ScoringResult: Scoring results with breakdown
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create a dictionary for easy lookup
        features_dict = features.__dict__
        
        X = pd.DataFrame([features_dict], columns=self.feature_names)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Predict score
        predicted_score = self.model.predict(X_scaled)[0]
        
        # Clip score between 0 and 100
        predicted_score = np.clip(predicted_score, 0, 100)
        
        # Get feature importance and scores
        feature_importance = self._get_feature_importance()
        feature_scores = self._calculate_feature_scores(features)
        
        # Calculate confidence: higher for scores further from the average (50)
        confidence = 0.5 + (abs(predicted_score - 50) / 100.0)
        
        # Generate breakdown and suggestions
        breakdown = self._generate_breakdown(features, feature_scores)
        suggestions = self._generate_suggestions(features, predicted_score)
        
        return ScoringResult(
            overall_score=predicted_score,
            feature_scores=feature_scores,
            feature_importance=feature_importance,
            confidence=confidence,
            breakdown=breakdown,
            suggestions=suggestions
        )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances from the trained model"""
        importances = None
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif self.model and hasattr(self.model, 'coef_'):
            # For linear models, use the absolute value of coefficients
            importances = np.abs(self.model.coef_)

        if importances is not None:
            # The output of coef_ can be a 2D array, flatten it.
            importances = importances.flatten()
            
            # Normalize importances
            total_importance = np.sum(importances)
            if total_importance > 0:
                normalized_importances = importances / total_importance
                return dict(zip(self.feature_names, normalized_importances))
        
        return {}
    
    def _calculate_feature_scores(self, features: ResumeFeatures) -> Dict[str, float]:
        """Calculate scores for individual features"""
        scores = {}
        
        # Text quality scores
        scores['text_length'] = min(100, features.text_length / 20)
        scores['readability'] = min(100, max(0, features.readability_score))
        scores['skills'] = min(100, features.skills_count * 4)
        scores['experience'] = min(100, features.experience_years * 6)
        scores['formatting'] = features.formatting_score
        
        # Section completeness
        section_score = 0
        if features.has_contact_info:
            section_score += 20
        if features.has_summary:
            section_score += 20
        if features.has_experience:
            section_score += 20
        if features.has_education:
            section_score += 20
        if features.has_skills:
            section_score += 20
        scores['sections'] = section_score
        
        # Achievement indicators
        scores['achievements'] = min(100, features.action_verbs_count * 5 + features.quantifiable_achievements * 10)
        
        return scores
    
    def _generate_breakdown(self, features: ResumeFeatures, feature_scores: Dict[str, float]) -> Dict[str, str]:
        """Generate detailed breakdown of the score"""
        breakdown = {}
        
        # Text quality
        if features.readability_score >= 60:
            breakdown['Readability'] = f"Excellent ({features.readability_score:.0f})"
        elif features.readability_score >= 40:
            breakdown['Readability'] = f"Good ({features.readability_score:.0f})"
        else:
            breakdown['Readability'] = f"Needs improvement ({features.readability_score:.0f})"
        
        # Skills
        if features.skills_count >= 15:
            breakdown['Skills'] = f"Comprehensive ({features.skills_count} skills)"
        elif features.skills_count >= 8:
            breakdown['Skills'] = f"Good ({features.skills_count} skills)"
        else:
            breakdown['Skills'] = f"Limited ({features.skills_count} skills)"
        
        # Experience
        if features.experience_years >= 5:
            breakdown['Experience'] = f"Extensive ({features.experience_years:.1f} years)"
        elif features.experience_years >= 2:
            breakdown['Experience'] = f"Good ({features.experience_years:.1f} years)"
        else:
            breakdown['Experience'] = f"Entry-level ({features.experience_years:.1f} years)"
        
        # Achievements
        if features.action_verbs_count >= 10:
            breakdown['Achievements'] = f"Strong ({features.action_verbs_count} action verbs)"
        elif features.action_verbs_count >= 5:
            breakdown['Achievements'] = f"Good ({features.action_verbs_count} action verbs)"
        else:
            breakdown['Achievements'] = f"Limited ({features.action_verbs_count} action verbs)"
        
        return breakdown
    
    def _generate_suggestions(self, features: ResumeFeatures, overall_score: float) -> List[str]:
        """Generate improvement suggestions based on features and score"""
        suggestions = []
        
        if overall_score < 60:
            suggestions.append("Overall resume quality needs significant improvement.")
        
        if features.readability_score < 40:
            suggestions.append("Improve readability by using shorter sentences and simpler language.")
        
        if features.skills_count < 8:
            suggestions.append("Add more relevant skills to showcase your capabilities.")
        
        if features.experience_years < 2:
            suggestions.append("Consider adding internships, projects, or volunteer work to build experience.")
        
        if features.action_verbs_count < 5:
            suggestions.append("Use more action verbs to describe your achievements and responsibilities.")
        
        if features.quantifiable_achievements < 2:
            suggestions.append("Add quantifiable achievements with specific numbers and metrics.")
        
        if not features.has_summary:
            suggestions.append("Add a professional summary to highlight your key qualifications.")
        
        if features.bullet_points_count < 10:
            suggestions.append("Use more bullet points to organize information and highlight key points.")
        
        return suggestions
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # This section runs when the file is executed directly
    # It's useful for testing the scoring model
    
    scorer = ResumeScoringModel()
    
    # Train the model
    print("Training resume scoring model...")
    scorer.train_model()
    
    # Save the model
    scorer.save_model('models/resume_model.pkl')
    
    print("Resume Scoring Model trained and saved successfully!")
    print("To use this model, import it in your main application.") 