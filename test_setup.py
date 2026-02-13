#!/usr/bin/env python
"""Simple test to verify the AI chatbot is working with Ollama"""

import requests
import json
import time

API_URL = "http://localhost:8000/ask"
OLLAMA_API = "http://localhost:11434/api/generate"

print("=" * 60)
print("Testing Chatbot Setup")
print("=" * 60)

# Test 1: Check if Ollama is running
print("\n1. Checking if Ollama is running...")
try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    if resp.status_code == 200:
        models = resp.json().get("models", [])
        print(f"   SUCCESS: Ollama is running with {len(models)} model(s)")
        for model in models:
            print(f"   - {model['name']}")
    else:
        print(f"   FAILED: Ollama returned status {resp.status_code}")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 2: Test Ollama directly
print("\n2. Testing Ollama API directly...")
try:
    resp = requests.post(
        OLLAMA_API,
        json={"model": "gemma3:4b", "prompt": "Hi", "stream": False},
        timeout=60
    )
    if resp.status_code == 200:
        answer = resp.json().get("response", "")
        print(f"   SUCCESS: Ollama responded with: {answer[:100]}...")
    else:
        print(f"   FAILED: Status {resp.status_code}")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 3: Test chatbot endpoint (if server is running)
print("\n3. Testing Chatbot API...")
try:
    response = requests.post(
        API_URL,
        json={"question": "test"},
        timeout=60
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   SUCCESS: Chatbot responded")
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   AI Generated: {data.get('is_ai_generated', False)}")
    else:
        print(f"   FAILED: Status {response.status_code}")
except Exception as e:
    print(f"   Chatbot not running yet (expected): {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
