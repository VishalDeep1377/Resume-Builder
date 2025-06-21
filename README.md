---
title: AI Resume Evaluator
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
---

# 🚀 AI-Powered Resume Evaluator & Job Match Suite

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-ff4b4b?logo=streamlit)](https://streamlit.io/) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced, AI-powered web app for resume evaluation, ATS scoring, job matching, LinkedIn analysis, cover letter generation, and more. Built with Streamlit and Python, featuring modular ML/NLP, GPT-powered improvements, and a beautiful, user-friendly UI.

---

## ✨ Key Features

- **ATS-Friendly Resume Template Download** (DOCX)
- **Section-by-Section Feedback** (Skills, Experience, Education)
- **Missing Section Alerts & Inline Tips**
- **Job Description Matching** (per-JD match scores, missing keywords, suggestions)
- **LinkedIn Profile Analyzer** (PDF or pasted text)
- **Cover Letter Generator** (AI or template-based)
- **One-Click AI Resume Rewriter** (DOCX, OpenAI-powered)
- **AI Chatbot Assistant** (OpenAI Q&A, persistent chat)
- **Modern, Responsive UI** (dark mode, mobile-friendly)
- **Step-by-Step OpenAI API Key Setup Guide**

---

## 📸 Demo

![Uploading image.png…]()


---

## 📚 Table of Contents
- [Quick Start](#-quick-start)
- [How to Use](#-how-to-use)
- [Features Explained](#-features-explained)
- [OpenAI API Integration](#-openai-api-integration)
- [Project Structure](#-project-structure)
- [Customization](#-customization)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Deployment](#-deployment)

---

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd Resume-Builder
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the App
```bash
streamlit run main_app.py
```
Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Deployment

This app is ready to be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

1. **Push to GitHub**: Make sure your code is pushed to a public GitHub repository.
2. **Sign up for Streamlit Community Cloud**: If you haven't already, sign up for a free account.
3. **Deploy**: Click "New app" from your workspace, connect your GitHub account, and select the repository and `main_app.py` file.

That's it! Your app will be live in a few minutes.

---

## 🧑‍💻 How to Use

1. **Upload Resume**: PDF only (max 10MB)
2. **Enter Job Description**: Paste the JD text
3. **Analyze**: Click "Analyze Resume"
4. **Review**: See scores, feedback, and suggestions
5. **Download**: Get ATS-friendly resume or cover letter (DOCX)
6. **LinkedIn Analyzer**: Upload your LinkedIn PDF or paste profile text
7. **AI Chatbot**: Ask resume/job search questions (OpenAI API key required)

---

## 🧩 Features Explained

### 📝 Resume Parsing & Section Feedback
- Extracts text, detects Skills/Experience/Education
- Alerts for missing sections, gives inline improvement tips

### 🏆 ATS Scoring
- Checks formatting, section presence, keyword usage
- Scores 0-100, with actionable feedback

### 🔍 Job Description Matching
- Compares resume to JD using ML/NLP (TF-IDF, transformers)
- Shows match %, missing keywords, and suggestions

### 🧑‍💼 LinkedIn Profile Analyzer
- Upload LinkedIn PDF or paste text (no URLs)
- Compares resume to LinkedIn profile, highlights gaps

### 📝 Cover Letter Generator
- Generates tailored cover letters (AI or template)
- Download or copy instantly

### ✨ AI Resume Rewriter
- One-click improved resume (DOCX)
- Uses OpenAI GPT if API key is provided

### 🤖 AI Chatbot Assistant
- Resume/job search Q&A
- Persistent chat history
- Powered by OpenAI (API key required)

---

## 🔑 OpenAI API Integration

Some features (AI rewriting, chatbot, advanced suggestions) require your own OpenAI API key.

**How to set up:**
1. [Get your OpenAI API key](https://platform.openai.com/account/api-keys)
2. Enter it in the app sidebar (step-by-step guide included)
3. Your key is never stored or sent anywhere except to OpenAI

---

## 🗂 Project Structure

```
Resume-Builder/
├── main_app.py           # Main Streamlit app
├── resume_parser.py      # Resume parsing logic
├── ats_checker.py        # ATS scoring logic
├── jd_matcher.py         # Job matching logic
├── scoring_model.py      # ML scoring model
├── utils.py              # Utilities
├── models/
│   └── resume_model.pkl  # Trained ML model
├── data/
│   └── sample_resumes/   # Example resumes
├── requirements.txt      # Dependencies
├── Procfile              # For deployment platforms
├── README.md             # This file
```

---

## 🛠️ Customization
- **Train your own model**: Edit `scoring_model.py`
- **Change ATS rules**: Edit `ats_checker.py`
- **Tweak UI/UX**: Edit `main_app.py`
- **Add features**: Use `utils.py` for helpers

For more, see this README.

---

## 🧩 Troubleshooting & FAQ

- **Module not found?**  
  `pip install -r requirements.txt`
- **spaCy model error?**  
  `python -m spacy download en_core_web_sm`
- **Port in use?**  
  `streamlit run main_app.py --server.port 8502`
- **Large PDF issues?**  
  Try a smaller file, ensure >2GB RAM
- **OpenAI errors?**  
  Check your API key and internet connection

For more, see this README.

---

## 🤝 Contributing

Pull requests, issues, and feature suggestions are welcome! Please open an issue to discuss major changes.

---

## 📄 License

MIT License.

---

**Built for job seekers, recruiters, and developers.** 
