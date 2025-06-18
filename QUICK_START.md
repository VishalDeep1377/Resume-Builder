# 🚀 Quick Start Guide - AI Resume Evaluator

This guide will help you set up and run the AI Resume Evaluator in just a few minutes!

## 📋 Prerequisites

- **Python 3.8 or higher** (check with `python --version`)
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)

## 🛠️ Installation Steps

### Step 1: Clone or Download the Project

If you have Git:
```bash
git clone <repository-url>
cd Resume-Builder
```

Or simply download and extract the project files to a folder.

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: Automatic Setup (Recommended)**
```bash
python setup.py
```

**Option B: Manual Installation**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 4: Run the Application

```bash
streamlit run main_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 How to Use

### 1. Upload Your Resume
- Click "Browse files" and select your PDF resume
- Maximum file size: 10MB
- Supported format: PDF only

### 2. Enter Job Description
- Paste the complete job description in the text area
- This helps analyze how well your resume matches the job requirements

### 3. Analyze Resume
- Click the "🚀 Analyze Resume" button
- Wait for the analysis to complete (usually 30-60 seconds)

### 4. Review Results
The analysis provides:

**📊 Overall Assessment**
- ATS-Friendliness Score
- Job Matching Percentage
- Quality Score
- Average Score

**📋 Detailed Analysis**
- **ATS Analysis**: Formatting compliance, section presence, structure
- **Job Matching**: Keyword alignment, missing skills, similarity scores
- **Quality Score**: Overall resume effectiveness with feature breakdown
- **Content Analysis**: Extracted skills, experience, contact information
- **Recommendations**: Actionable improvement suggestions

### 5. Download Report
- Click "📄 Download Report" to save your analysis results

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
pip install -r requirements.txt
```

**2. spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**3. Port already in use**
```bash
streamlit run main_app.py --server.port 8502
```

**4. Memory issues with large PDFs**
- Try with a smaller PDF file
- Ensure you have at least 2GB of free RAM

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Ensure you're using Python 3.8 or higher
3. Try running the setup script again: `python setup.py`
4. Check the console output for error messages

## 📁 Project Structure

```
Resume-Builder/
├── main_app.py              # Main Streamlit application
├── resume_parser.py         # PDF parsing and text extraction
├── ats_checker.py          # ATS-friendliness analysis
├── jd_matcher.py           # Job description matching
├── scoring_model.py        # ML model for resume scoring
├── utils.py                # Utility functions
├── setup.py                # Setup script
├── requirements.txt        # Python dependencies
├── README.md              # Detailed documentation
├── QUICK_START.md         # This file
├── models/                # Trained ML models
└── data/                  # Sample data
    └── sample_resumes/
```

## 🎓 Learning Resources

This project demonstrates:

- **Machine Learning**: Training and deploying ML models
- **Natural Language Processing**: Text extraction and analysis
- **Web Development**: Building interactive web applications
- **Data Processing**: Working with PDFs and structured data
- **API Integration**: Optional OpenAI integration

## 🚀 Next Steps

After running the application:

1. **Experiment**: Try different resumes and job descriptions
2. **Customize**: Modify the code to add new features
3. **Deploy**: Learn how to deploy to Streamlit Cloud or other platforms
4. **Improve**: Train the model on your own resume dataset

## 📞 Support

If you need help or have questions:

1. Check the detailed README.md file
2. Review the code comments for explanations
3. Look at the troubleshooting section above
4. Create an issue in the project repository

---

**Happy resume analyzing! 🎉** 