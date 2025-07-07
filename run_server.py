import uvicorn
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_local():
    """Run server locally only"""
    print("🚀 Starting FastAPI server (local only)...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Only accept connections from localhost
        port=8080,
        reload=True
    )

def run_with_ngrok():
    """Run server with ngrok tunnel"""
    try:
        from pyngrok import ngrok, conf
        
        # Set ngrok auth token if provided
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if ngrok_auth_token:
            ngrok.set_auth_token(ngrok_auth_token)
            print("✅ Ngrok auth token set from environment")
        else:
            print("⚠️  No NGROK_AUTH_TOKEN found in environment. Some features may be limited.")
        
        # Configure ngrok
        ngrok_config = conf.get_default()
        
        # Start uvicorn in background
        print("🚀 Starting FastAPI server...")
        import threading
        server_thread = threading.Thread(
            target=lambda: uvicorn.run(
                "main:app",
                host="0.0.0.0",  # Accept connections from anywhere for ngrok
                port=8000,
                reload=False  # Disable reload when using ngrok
            )
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Create ngrok tunnel
        print("🌐 Creating ngrok tunnel...")
        
        # Get custom domain if specified
        ngrok_domain = os.getenv("NGROK_DOMAIN")
        tunnel_kwargs = {"addr": 8000}
        
        if ngrok_domain:
            tunnel_kwargs["hostname"] = ngrok_domain
            print(f"🏷️  Using custom domain: {ngrok_domain}")
        
        # Create tunnel
        public_tunnel = ngrok.connect(**tunnel_kwargs)
        public_url = public_tunnel.public_url
        
        print("\n" + "="*60)
        print("🎉 SERVER READY!")
        print("="*60)
        print(f"🏠 Local URL:  http://localhost:8000")
        print(f"🌍 Public URL: {public_url}")
        print("="*60)
        print("📊 Ngrok Dashboard: http://localhost:4040")
        print("🛑 Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        # Keep the main thread alive
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            ngrok.disconnect(public_tunnel.public_url)
            ngrok.kill()
            
    except ImportError:
        print("❌ Error: pyngrok not installed. Run: pip install pyngrok")
        return
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        print("💡 Tip: Make sure you have ngrok installed and configured")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Verification API Server")
    parser.add_argument(
        "--ngrok", 
        action="store_true", 
        help="Run with ngrok tunnel for public access"
    )
    parser.add_argument(
        "--local", 
        action="store_true", 
        help="Run locally only (default)"
    )
    
    args = parser.parse_args()
    
    if args.ngrok:
        run_with_ngrok()
    else:
        run_local() 