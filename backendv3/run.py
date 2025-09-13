#!/usr/bin/env python
"""
Convenience script to run the application.

Usage:
    python run.py [--host HOST] [--port PORT]
"""
import argparse
import uvicorn
import time
import sys

def print_status_message(host, port):
    """Print a clear, formatted server status message"""
    print("\n" + "="*60)
    print(f"🚀 TRAFFIC MONITORING SERVER STARTING")
    print(f"📡 Server URL: http://{host}:{port}")
    print(f"📋 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Add a small loading animation
    for _ in range(3):
        sys.stdout.write("⏳ Initializing server")
        for i in range(4):
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(0.25)
        sys.stdout.write("\r")
        sys.stdout.flush()
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Traffic Monitoring API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Print a clearer status message
    print_status_message(args.host, args.port)
    
    try:
        uvicorn.run(
            "app.main:app", 
            host=args.host, 
            port=args.port, 
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("🛑 Server shutdown requested. Stopping...")
        print("="*60 + "\n")
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR: Server failed to start.")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Value: {e}")
        print(f"Exception Representation: {repr(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        sys.exit(1)