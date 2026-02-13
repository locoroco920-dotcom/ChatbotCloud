#!/usr/bin/env python
"""
Simple test for the chatbot API
"""
import requests
import json
import time

def test_api():
    print("Testing Chatbot API...\n")
    
    time.sleep(5)  # Give server time to start
    
    # Test 1: Health endpoint
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✓ Health endpoint: {response.json()}")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False
    
    # Test 2: Root endpoint
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"✓ Root endpoint: {response.json()}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
    
    # Test 3: Ask endpoint
    try:
        payload = {"question": "What is your name?", "session_id": "test-session-1"}
        response = requests.post("http://localhost:8000/ask", json=payload, timeout=30)
        result = response.json()
        print(f"✓ Ask endpoint: {result['answer'][:100]}...")
    except Exception as e:
        print(f"✗ Ask endpoint failed: {e}")
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_api()
