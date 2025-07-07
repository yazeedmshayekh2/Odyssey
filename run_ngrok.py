#!/usr/bin/env python3
"""
Dedicated script for running the Car Verification API with ngrok tunnel.
This provides a simplified way to expose your local server to the internet.
"""

import uvicorn
import os
import sys
import time
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to run server with ngrok"""
    
    # Check if pyngrok is available
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("❌ Error: pyngrok not installed.")
        print("📦 Install it with: pip install pyngrok")
        print("📦 Or install all requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Car Verification API - Ngrok Mode")
    print("=" * 50)
    
    # Set ngrok auth token
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_auth_token and ngrok_auth_token != "your_ngrok_auth_token_here":
        ngrok.set_auth_token(ngrok_auth_token)
        print("✅ Ngrok auth token loaded from .env file")
    else:
        print("⚠️  Warning: No valid NGROK_AUTH_TOKEN found!")
        print("📝 Please:")
        print("   1. Copy .env.example to .env")
        print("   2. Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. Add it to your .env file")
        print("   4. Or continue with limited ngrok features")
        print()
    
    # Start FastAPI server in background thread
    print("🖥️  Starting FastAPI server...")
    
    def run_server():
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server to initialize...")
    time.sleep(5)
    
    # Create ngrok tunnel
    print("🌐 Creating ngrok tunnel...")
    
    try:
        # Check for custom domain
        ngrok_domain = os.getenv("NGROK_DOMAIN")
        tunnel_kwargs = {"addr": 8000}
        
        if ngrok_domain and ngrok_domain != "your-custom-domain.ngrok.io":
            tunnel_kwargs["hostname"] = ngrok_domain
            print(f"🏷️  Using custom domain: {ngrok_domain}")
        
        # Create the tunnel
        public_tunnel = ngrok.connect(**tunnel_kwargs)
        public_url = public_tunnel.public_url
        
        # Display connection info
        print("\n" + "🎉" + "=" * 48 + "🎉")
        print("           CAR VERIFICATION API READY!")
        print("=" * 50)
        print(f"🏠 Local Access:   http://localhost:8000")
        print(f"🌍 Public Access:  {public_url}")
        print("=" * 50)
        print("📱 Test Endpoints:")
        print(f"   • Home Page:    {public_url}")
        print(f"   • Health Check: {public_url}/health")
        print(f"   • API Docs:     {public_url}/docs")
        print("=" * 50)
        print("📊 Ngrok Dashboard: http://localhost:4040")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50 + "\n")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            
    except Exception as e:
        print(f"❌ Error creating ngrok tunnel: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Check your internet connection")
        print("   • Verify your ngrok auth token")
        print("   • Make sure port 8000 is not in use")
        print("   • Try running: ngrok http 8000 (manual test)")
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            ngrok.disconnect(public_url)
            ngrok.kill()
            print("✅ Ngrok tunnel closed")
        except:
            pass

if __name__ == "__main__":
    main() 