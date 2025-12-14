#!/usr/bin/env python3
"""
Enhanced server runner with better error handling and logging
"""

import os
import sys
from app import app

def main():
    print("Starting Deepfake Detection Server...")
    print("=" * 50)
    
    # Check if model exists
    model_path = "deepfake_model.h5"
    if os.path.exists(model_path):
        print("Pre-trained model found")
    else:
        print("No pre-trained model found - using demo mode")
    
    print("\nServer Configuration:")
    print(f"   Host: localhost")
    print(f"   Port: 5000")
    print(f"   Debug: True")
    print(f"   CORS: Enabled")
    
    print("\nAccess the application at:")
    print("   http://localhost:5000")
    
    print("\nFeatures Available:")
    print("   - File Upload (MP4, AVI, MOV, MKV, WebM)")
    print("   - YouTube URL Analysis")
    print("   - Direct Video URL Support")
    print("   - Advanced CNN + FFT Analysis")
    print("   - Real-time Results with Confidence")
    print("   - No Size or Duration Limits")
    
    print("\n" + "=" * 50)
    print("Server starting... Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        app.run(
            host='localhost',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False  # Prevent double startup in debug mode
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()