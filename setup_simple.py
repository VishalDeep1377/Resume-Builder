"""
Simplified Setup Script for AI Resume Evaluator
===============================================

This script helps set up the environment and install dependencies
for the AI Resume Evaluator application, optimized for Python 3.13.

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

def install_package(package, description):
    """Install a single package with error handling"""
    print(f"📦 Installing {description}...")
    try:
        result = subprocess.run(
            f"pip install {package}", 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up AI Resume Evaluator (Python 3.13 compatible)...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Install packages one by one to handle compatibility issues
    packages_to_install = [
        ("streamlit>=1.29.0", "Streamlit web framework"),
        ("pdfplumber>=0.10.0", "PDF processing library"),
        ("spacy>=3.7.0", "Natural language processing"),
        ("scikit-learn>=1.3.0", "Machine learning library"),
        ("sentence-transformers>=2.2.0", "Sentence embeddings"),
        ("textstat>=0.7.0", "Text statistics"),
        ("pandas>=2.0.0", "Data manipulation"),
        ("numpy>=1.24.0", "Numerical computing"),
        ("python-dotenv>=1.0.0", "Environment variables"),
        ("Pillow>=10.0.0", "Image processing"),
        ("openai>=1.3.0", "OpenAI API (optional)")
    ]
    
    failed_packages = []
    
    for package, description in packages_to_install:
        if not install_package(package, description):
            failed_packages.append(package)
            print(f"⚠️  Continuing with other packages...")
    
    # Download spaCy model
    print("🧠 Downloading spaCy language model...")
    try:
        result = subprocess.run(
            "python -m spacy download en_core_web_sm", 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        print("You can download it manually later with: python -m spacy download en_core_web_sm")
    
    # Create necessary directories
    print("📁 Creating directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/sample_resumes", exist_ok=True)
    print("✅ Directories created successfully!")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Setup Summary:")
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\n💡 You can try installing them manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    else:
        print("✅ All packages installed successfully!")
    
    print("\n🚀 To run the application:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run: streamlit run main_app.py")
    print("3. Open your browser to the URL shown")
    
    print("\n📚 If you encounter issues:")
    print("- Try installing packages individually")
    print("- Check Python version compatibility")
    print("- Consider using Python 3.11 or 3.12 for better compatibility")
    
    return len(failed_packages) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Setup completed with some issues. Check the output above.")
    else:
        print("\n🎉 Setup completed successfully!") 