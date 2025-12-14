import os
import cv2
import numpy as np
from deepfake_detector import DeepfakeDetector

def create_synthetic_training_data():
    """Create synthetic training data for demonstration"""
    
    # Create directories
    os.makedirs('data/real', exist_ok=True)
    os.makedirs('data/fake', exist_ok=True)
    
    print("Creating synthetic training data...")
    
    # Create real-looking videos (stable, consistent)
    for i in range(5):
        create_real_video(f'data/real/real_video_{i}.mp4')
    
    # Create fake-looking videos (artifacts, inconsistencies)
    for i in range(5):
        create_fake_video(f'data/fake/fake_video_{i}.mp4')
    
    print("Sample data created!")
    print("Real videos: data/real/")
    print("Fake videos: data/fake/")

def create_real_video(filename, duration=3):
    """Create a realistic-looking video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (224, 224))
    
    # Base face-like pattern
    base_face = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(base_face, (112, 112), 80, (220, 180, 160), -1)  # Face
    cv2.circle(base_face, (90, 90), 8, (50, 50, 50), -1)       # Left eye
    cv2.circle(base_face, (134, 90), 8, (50, 50, 50), -1)      # Right eye
    cv2.ellipse(base_face, (112, 130), (15, 8), 0, 0, 180, (180, 120, 120), -1)  # Mouth
    
    for frame_num in range(duration * 20):
        frame = base_face.copy()
        
        # Add natural variations (breathing, micro-movements)
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Slight movement
        shift_x = int(2 * np.sin(frame_num * 0.1))
        shift_y = int(1 * np.cos(frame_num * 0.15))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (224, 224))
        
        out.write(frame)
    
    out.release()

def create_fake_video(filename, duration=3):
    """Create a fake-looking video with artifacts"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (224, 224))
    
    for frame_num in range(duration * 20):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Inconsistent face generation
        face_color = (200 + np.random.randint(-30, 30), 
                     160 + np.random.randint(-20, 20), 
                     140 + np.random.randint(-20, 20))
        
        cv2.circle(frame, (112, 112), 80, face_color, -1)
        
        # Flickering eyes (deepfake artifact)
        if frame_num % 3 == 0:
            cv2.circle(frame, (90, 90), 8, (0, 0, 0), -1)
            cv2.circle(frame, (134, 90), 8, (0, 0, 0), -1)
        
        # Inconsistent mouth
        mouth_y = 130 + np.random.randint(-5, 5)
        cv2.ellipse(frame, (112, mouth_y), (15, 8), 0, 0, 180, (150, 100, 100), -1)
        
        # Add compression artifacts
        if frame_num % 5 == 0:
            # Blocky artifacts
            for y in range(0, 224, 8):
                for x in range(0, 224, 8):
                    if np.random.random() > 0.8:
                        cv2.rectangle(frame, (x, y), (x+8, y+8), 
                                    (np.random.randint(0, 255), 
                                     np.random.randint(0, 255), 
                                     np.random.randint(0, 255)), -1)
        
        # High frequency noise (deepfake signature)
        noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()

def train_with_sample_data():
    """Train the model with synthetic data"""
    print("Training model with sample data...")
    
    detector = DeepfakeDetector()
    
    # Prepare training data
    video_paths = []
    labels = []
    
    # Real videos
    for i in range(5):
        video_paths.append(f'data/real/real_video_{i}.mp4')
        labels.append(1)
    
    # Fake videos  
    for i in range(5):
        video_paths.append(f'data/fake/fake_video_{i}.mp4')
        labels.append(0)
    
    # Train model
    history = detector.train(video_paths, labels, epochs=10, batch_size=4)
    
    # Save trained model
    detector.save_model('deepfake_model.h5')
    print("Model trained and saved!")
    
    return detector

if __name__ == "__main__":
    create_synthetic_training_data()
    train_with_sample_data()