#!/usr/bin/env python
"""
Simple script to run the chatbot server in a way that keeps it running
"""
import subprocess
import sys
import os

def main():
    print("\n" + "="*60)
    print("         AI CHATBOT SERVER")
    print("="*60)
    print("\nStarting chatbot server...")
    print("DO NOT close this window while using chatbot!")
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run start_chatbot.py directly
    python_exe = r"C:\Users\locor\AppData\Local\spyder-6\envs\spyder-runtime\python.exe"
    
    try:
        subprocess.run([python_exe, "start_chatbot.py"], check=False)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
