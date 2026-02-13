import uvicorn
from pyngrok import ngrok
from main import app
import sys

def start_server():
    # Open a HTTP tunnel on the default port 8000
    # Note: You might need to sign up for ngrok and run `ngrok config add-authtoken <token>` 
    # if the session expires too quickly.
    try:
        public_url = ngrok.connect(8000).public_url
        print("\n" + "="*60)
        print(f" PUBLIC URL: {public_url}")
        print(f" Update your website code to use: {public_url}/ask")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        print("Ensure ngrok is installed or check your network connection.")

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
