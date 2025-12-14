import os
import cv2
import numpy as np
from deepfake_detector import DeepfakeDetector
import random

def create_enhanced_training_data():
    """Create more sophisticated training data with better fake patterns"""
    
    os.makedirs('data/real', exist_ok=True)
    os.makedirs('data/fake', exist_ok=True)
    
    print("Creating enhanced training dataset...")
    
    # Create more real videos with natural variations
    for i in range(15):
        create_realistic_video(f'data/real/real_video_{i}.mp4', is_real=True)
    
    # Create more fake videos with deepfake artifacts
    for i in range(15):
        create_deepfake_video(f'data/fake/fake_video_{i}.mp4')
    
    print("Enhanced dataset created!")
    print(f"Real videos: 15")
    print(f"Fake videos: 15")

def create_realistic_video(filename, is_real=True, duration=5):
    """Create realistic video with natural human characteristics"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 25.0, (224, 224))
    
    # Create base face with natural skin tone variation
    base_color = (210 + random.randint(-20, 20), 
                  175 + random.randint(-15, 15), 
                  155 + random.randint(-15, 15))
    
    for frame_num in range(duration * 25):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Natural face with slight color variations
        face_color = tuple(max(0, min(255, c + random.randint(-5, 5))) for c in base_color)
        
        # Draw face
        cv2.circle(frame, (112, 112), 85, face_color, -1)
        
        # Natural eye movement and blinking
        blink = random.random() < 0.05  # 5% chance of blink
        if not blink:
            # Eyes with natural movement
            eye_offset = random.randint(-2, 2)
            cv2.circle(frame, (90 + eye_offset, 90), 6, (40, 40, 40), -1)
            cv2.circle(frame, (134 + eye_offset, 90), 6, (40, 40, 40), -1)
            # Eye shine
            cv2.circle(frame, (92 + eye_offset, 88), 2, (255, 255, 255), -1)
            cv2.circle(frame, (136 + eye_offset, 88), 2, (255, 255, 255), -1)
        else:
            # Closed eyes
            cv2.ellipse(frame, (90, 90), (8, 2), 0, 0, 180, (40, 40, 40), -1)
            cv2.ellipse(frame, (134, 90), (8, 2), 0, 0, 180, (40, 40, 40), -1)
        
        # Natural mouth movement (talking)
        mouth_y = 130 + int(3 * np.sin(frame_num * 0.3))
        mouth_width = 12 + int(3 * np.sin(frame_num * 0.2))
        cv2.ellipse(frame, (112, mouth_y), (mouth_width, 6), 0, 0, 180, (120, 80, 80), -1)
        
        # Natural head movement
        shift_x = int(2 * np.sin(frame_num * 0.05))
        shift_y = int(1 * np.cos(frame_num * 0.07))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (224, 224))
        
        # Add natural skin texture
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()

def create_deepfake_video(filename, duration=5):
    """Create video with deepfake artifacts"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 25.0, (224, 224))
    
    for frame_num in range(duration * 25):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Inconsistent face color (deepfake artifact)
        if frame_num % 10 == 0:  # Color shift every 10 frames
            base_color = (random.randint(180, 230), 
                         random.randint(140, 180), 
                         random.randint(120, 160))
        
        # Face with color inconsistencies
        cv2.circle(frame, (112, 112), 85, base_color, -1)
        
        # Unnatural eye behavior (deepfake artifact)
        if frame_num % 7 == 0:  # Eyes disappear/reappear
            eye_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        else:
            eye_color = (40, 40, 40)
        
        cv2.circle(frame, (90, 90), 6, eye_color, -1)
        cv2.circle(frame, (134, 90), 6, eye_color, -1)
        
        # Unnatural mouth (lip sync issues)
        mouth_phase = random.random()  # Random mouth positions
        if mouth_phase < 0.3:
            cv2.ellipse(frame, (112, 130), (15, 8), 0, 0, 180, (120, 80, 80), -1)
        elif mouth_phase < 0.6:
            cv2.circle(frame, (112, 130), 5, (0, 0, 0), -1)
        else:
            cv2.ellipse(frame, (112, 130), (8, 12), 0, 0, 180, (120, 80, 80), -1)
        
        # Compression artifacts (common in deepfakes)
        if frame_num % 5 == 0:
            # Add blocky artifacts
            for y in range(0, 224, 8):
                for x in range(0, 224, 8):
                    if random.random() > 0.9:
                        artifact_color = (random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255))
                        cv2.rectangle(frame, (x, y), (x+8, y+8), artifact_color, -1)
        
        # High frequency noise (deepfake signature)
        noise = np.random.normal(0, 20, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Temporal inconsistencies
        if random.random() < 0.1:  # 10% chance of frame corruption
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        out.write(frame)
    
    out.release()

def train_enhanced_model():
    """Train model with enhanced dataset"""
    print("Training enhanced deepfake detection model...")
    
    detector = DeepfakeDetector()
    
    # Prepare enhanced training data
    video_paths = []
    labels = []
    
    # Real videos
    for i in range(15):
        video_paths.append(f'data/real/real_video_{i}.mp4')
        labels.append(1)
    
    # Fake videos  
    for i in range(15):
        video_paths.append(f'data/fake/fake_video_{i}.mp4')
        labels.append(0)
    
    # Train with more epochs for better learning
    history = detector.train(video_paths, labels, epochs=25, batch_size=8)
    
    # Save enhanced model
    detector.save_model('deepfake_model.h5')
    print("Enhanced model trained and saved!")
    
    # Print training results
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\nTraining Results:")
    print(f"   Final Training Accuracy: {final_accuracy:.4f}")
    print(f"   Final Validation Accuracy: {final_val_accuracy:.4f}")
    
    return detector

if __name__ == "__main__":
    create_enhanced_training_data()
    train_enhanced_model()