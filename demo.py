#!/usr/bin/env python3
"""
Quick demo script for deepfake detection system
"""

import cv2
import numpy as np
from deepfake_detector import DeepfakeDetector
import os

def create_sample_video(filename="sample_video.mp4", duration=3):
    """Create a sample video for testing"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    
    for i in range(duration * 20):  # 20 FPS
        # Create a simple animated frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving circle
        center_x = int(320 + 100 * np.sin(i * 0.1))
        center_y = int(240 + 50 * np.cos(i * 0.1))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f'Frame {i}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {filename}")

def demo_detection():
    """Run deepfake detection demo"""
    print("Deepfake Detection Demo")
    print("=" * 40)
    
    # Create sample video if it doesn't exist
    sample_video = "sample_video.mp4"
    if not os.path.exists(sample_video):
        print("Creating sample video...")
        create_sample_video(sample_video)
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = DeepfakeDetector()
    detector.build_model()
    
    # Extract frames
    print(f"\nAnalyzing video: {sample_video}")
    frames = detector.extract_frames(sample_video)
    print(f"   Extracted {len(frames)} frames")
    
    # FFT analysis demo
    print("\nPerforming FFT analysis...")
    fft_features = detector.fft_analysis(frames)
    print(f"   Generated FFT features: {fft_features.shape}")
    
    # Prediction (with untrained model)
    print("\nRunning prediction...")
    try:
        score = detector.predict_video(sample_video)
        print(f"   Authenticity Score: {score:.3f}")
        
        if score > 0.7:
            result = "LIKELY REAL"
        elif score > 0.4:
            result = "UNCERTAIN"
        else:
            result = "LIKELY FAKE"
            
        print(f"   Result: {result}")
        
    except Exception as e:
        print(f"   Prediction failed (model not trained): {e}")
        print("   Train the model with real data for accurate predictions")
    
    # Model info
    print(f"\nModel Information:")
    print(f"   Architecture: ResNet50 + Custom layers")
    print(f"   Input size: {detector.img_size}")
    print(f"   Parameters: {detector.model.count_params():,}")
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("   1. Run: streamlit run streamlit_app.py")
    print("   2. Train with DFDC dataset using train_model.py")
    print("   3. Upload real videos for testing")

if __name__ == "__main__":
    demo_detection()