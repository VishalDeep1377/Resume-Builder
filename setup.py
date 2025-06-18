"""
Setup Script for AI Resume Evaluator
====================================

This script helps set up the environment and install dependencies
for the AI Resume Evaluator application.

Author: AI Resume Evaluator
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up AI Resume Evaluator...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy language model"):
        return False
    
    # Create necessary directories
    print("📁 Creating directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/sample_resumes", exist_ok=True)
    print("✅ Directories created successfully!")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run: streamlit run main_app.py")
    print("3. Open your browser to the URL shown")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 