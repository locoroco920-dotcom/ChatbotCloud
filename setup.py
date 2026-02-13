#!/usr/bin/env python
"""
Meadowlands Chatbot - Cross-Platform Setup Script
Run this script to install all dependencies and configure the chatbot.
Works on Windows, Mac, and Linux.
"""

import subprocess
import sys
import os
import platform

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(num, text):
    print(f"\n[{num}] {text}")

def print_ok(text):
    print(f"    [OK] {text}")

def print_error(text):
    print(f"    [ERROR] {text}")

def print_debug(text):
    print(f"    [DEBUG] {text}")

def install_package(package_name, pip_spec):
    """Install a single package with verbose output"""
    print(f"    Installing {package_name}...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_spec],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per package
        )
        if result.returncode == 0:
            print(f"    [OK] {package_name} installed")
            return True
        else:
            print(f"    [WARN] {package_name} had issues: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {package_name} took too long - skipping")
        return False
    except Exception as e:
        print(f"    [ERROR] {package_name}: {e}")
        return False

def main():
    print_header("MEADOWLANDS CHATBOT - SETUP")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version...")
    if sys.version_info < (3, 9):
        print_error("Python 3.9 or higher is required!")
        print("    Please install from https://www.python.org/downloads/")
        sys.exit(1)
    print_ok(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Step 2: Upgrade pip
    print_step(2, "Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    print_ok("pip upgraded")
    
    # Step 3: Install dependencies ONE BY ONE
    print_step(3, "Installing dependencies (one by one for debugging)...")
    
    packages = [
        ("fastapi", "fastapi>=0.100.0"),
        ("uvicorn", "uvicorn>=0.20.0"),
        ("openai", "openai>=1.0.0"),
        ("pydantic", "pydantic>=2.0.0"),
        ("python-multipart", "python-multipart>=0.0.6"),
        ("numpy", "numpy>=1.20.0"),
        ("pyngrok", "pyngrok>=5.0.0"),
    ]
    
    for name, spec in packages:
        install_package(name, spec)
    
    # Sentence transformers is big - install separately with more info
    print("\n    Installing sentence-transformers (LARGE package - may take 2-5 minutes)...")
    print("    This downloads PyTorch and transformer models...")
    install_package("sentence-transformers", "sentence-transformers>=2.0.0")
    
    print_ok("All dependencies installation attempted")
    
    # Step 4: Verify critical imports
    print_step(4, "Verifying installations...")
    
    imports_ok = True
    
    try:
        import fastapi
        print_ok(f"FastAPI {fastapi.__version__}")
    except ImportError as e:
        print_error(f"FastAPI not installed: {e}")
        imports_ok = False
    
    try:
        import openai
        print_ok(f"OpenAI {openai.__version__}")
    except ImportError as e:
        print_error(f"OpenAI not installed: {e}")
        imports_ok = False
    
    try:
        import uvicorn
        print_ok("Uvicorn installed")
    except ImportError as e:
        print_error(f"Uvicorn not installed: {e}")
        imports_ok = False
    
    try:
        import pyngrok
        print_ok("pyngrok installed")
    except ImportError as e:
        print_error(f"pyngrok not installed: {e}")
        imports_ok = False
    
    try:
        from sentence_transformers import SentenceTransformer
        print_ok("Sentence Transformers installed")
    except ImportError as e:
        print_error(f"Sentence Transformers not installed: {e}")
        imports_ok = False
    
    # Step 5: Check for required files
    print_step(5, "Checking required files...")
    
    required_files = ["start_chatbot.py", "faq_data.json"]
    for f in required_files:
        if os.path.exists(os.path.join(script_dir, f)):
            print_ok(f"{f} found")
        else:
            print_error(f"{f} missing!")
    
    # Step 6: Configure paths
    print_step(6, "Configuring paths for this PC...")
    print_debug(f"Script directory: {script_dir}")
    print_debug(f"Python executable: {sys.executable}")
    print_ok("Paths are configured dynamically in start_chatbot.py")
    
    # Step 7: Pre-download embedding model (optional)
    print_step(7, "Pre-downloading embedding model...")
    if imports_ok:
        try:
            from sentence_transformers import SentenceTransformer
            print("    Downloading model (this may take 1-2 minutes on first run)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print_ok("Embedding model ready")
        except Exception as e:
            print_error(f"Could not pre-load model: {e}")
    else:
        print("    Skipping - sentence-transformers not installed")
    
    # Done
    print_header("SETUP COMPLETE!")
    
    if imports_ok:
        print("""
All dependencies installed successfully!

To start the chatbot:

  Windows:  Double-click START_SERVER.bat
            Or run: python start_chatbot.py

  Mac/Linux: python start_chatbot.py

The server will display an ngrok URL you can use to access the chatbot.
""")
    else:
        print("""
Some dependencies failed to install. Try running:

  pip install -r requirements.txt

Or install missing packages manually.
""")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
