import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime

class DeepfakeDetector:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        
    def extract_frames(self, video_path, max_frames=300):
        """Extract frames from video with facial region detection"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        face_frames = []
        frame_count = 0
        
        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Get total frame count for uniform sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            frame_indices = range(total_frames)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Original frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, self.img_size)
                frames.append(frame_resized)
                
                # Face detection and cropping
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Expand face region by 20%
                    margin = int(0.2 * max(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, self.img_size)
                        face_frames.append(face_resized)
                
                frame_count += 1
                
        cap.release()
        
        # Return both full frames and face crops
        return {
            'full_frames': np.array(frames),
            'face_frames': np.array(face_frames) if face_frames else np.array(frames[:10]),
            'total_extracted': len(frames),
            'faces_detected': len(face_frames)
        }
    
    def fft_analysis(self, frames):
        """Apply FFT to extract frequency domain features"""
        fft_features = []
        
        for frame in frames:
            # Convert to grayscale for FFT
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply 2D FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Log transform and normalize
            magnitude = np.log(magnitude + 1)
            magnitude = (magnitude - magnitude.mean()) / magnitude.std()
            
            # Resize to match input size
            magnitude = cv2.resize(magnitude, self.img_size)
            fft_features.append(magnitude)
            
        return np.array(fft_features)
    
    def build_model(self):
        """Build enhanced CNN model with ResNet50 backbone"""
        # Base ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        
        # Unfreeze last few layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add enhanced custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Use different learning rates for base and custom layers
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=0.0001)
        
        self.model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def preprocess_data(self, video_paths, labels):
        """Preprocess video data for training with enhanced frame extraction"""
        X, y = [], []
        
        for video_path, label in zip(video_paths, labels):
            try:
                frame_data = self.extract_frames(video_path, max_frames=50)
                
                # Handle both old and new format
                if isinstance(frame_data, dict):
                    frames = frame_data['full_frames']
                    face_frames = frame_data['face_frames']
                    
                    # Use face frames if available, otherwise full frames
                    if len(face_frames) > 0:
                        selected_frames = face_frames
                    else:
                        selected_frames = frames
                else:
                    selected_frames = frame_data
                
                if len(selected_frames) > 0:
                    # Use multiple frames for better training
                    num_samples = min(3, len(selected_frames))
                    indices = np.linspace(0, len(selected_frames)-1, num_samples, dtype=int)
                    
                    for idx in indices:
                        X.append(selected_frames[idx] / 255.0)  # Normalize
                        y.append(label)
                        
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
                
        return np.array(X), np.array(y)
    
    def train(self, video_paths, labels, epochs=10, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        X, y = self.preprocess_data(video_paths, labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict_video(self, video_path):
        """Enhanced video prediction with facial analysis and metadata"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract metadata
        metadata = self.extract_metadata(video_path)
        
        # Extract frames with face detection
        frame_data = self.extract_frames(video_path, max_frames=300)
        full_frames = frame_data['full_frames']
        face_frames = frame_data['face_frames']
        
        if len(full_frames) == 0:
            return {
                'score': 0.5, 'frames_extracted': 0, 'frames_analyzed': 0, 
                'frame_scores': [], 'metadata': metadata
            }
        
        # Analyze both full frames and face crops
        full_scores = self.analyze_frames(full_frames, max_analyze=200)
        face_scores = self.analyze_frames(face_frames, max_analyze=150) if len(face_frames) > 0 else []
        
        # Combine scores with weighted approach
        if len(face_scores) > 0:
            # Give more weight to face-focused analysis
            combined_scores = (np.array(full_scores) * 0.4 + 
                             np.array(face_scores[:len(full_scores)]) * 0.6)
        else:
            combined_scores = np.array(full_scores)
        
        # Enhanced confidence calculation
        final_score = self.calculate_enhanced_confidence(combined_scores, metadata)
        
        return {
            'score': float(final_score),
            'frames_extracted': frame_data['total_extracted'],
            'frames_analyzed': len(combined_scores),
            'faces_detected': frame_data['faces_detected'],
            'frame_scores': combined_scores.tolist(),
            'face_scores': face_scores,
            'fake_probability': float(1 - final_score),
            'authenticity_percentage': float(final_score * 100),
            'metadata': metadata
        }
    
    def analyze_frames(self, frames, max_analyze=200):
        """Analyze frames in optimized batches"""
        frame_scores = []
        frames_to_analyze = min(len(frames), max_analyze)
        
        batch_size = 16  # Larger batch for efficiency
        for i in range(0, frames_to_analyze, batch_size):
            batch_end = min(i + batch_size, frames_to_analyze)
            batch_frames = frames[i:batch_end]
            
            # Normalize and predict
            batch_normalized = batch_frames / 255.0
            batch_scores = self.model.predict(batch_normalized, verbose=0)
            
            frame_scores.extend([float(score[0]) for score in batch_scores])
        
        return frame_scores
    
    def extract_metadata(self, video_path):
        """Extract video metadata for analysis"""
        cap = cv2.VideoCapture(video_path)
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0,
            'codec': 'unknown',
            'bitrate': 0
        }
        
        if metadata['fps'] > 0:
            metadata['duration'] = metadata['frame_count'] / metadata['fps']
        
        # Check for suspicious metadata patterns
        metadata['suspicious_patterns'] = self.check_suspicious_metadata(metadata)
        
        cap.release()
        return metadata
    
    def check_suspicious_metadata(self, metadata):
        """Check for metadata patterns common in deepfakes"""
        suspicious = []
        
        # Unusual frame rates
        if metadata['fps'] < 15 or metadata['fps'] > 60:
            suspicious.append('unusual_fps')
        
        # Very short videos (common in deepfake samples)
        if metadata['duration'] < 2:
            suspicious.append('very_short')
        
        # Unusual resolutions
        if metadata['width'] % 16 != 0 or metadata['height'] % 16 != 0:
            suspicious.append('unusual_resolution')
        
        return suspicious
    
    def calculate_enhanced_confidence(self, scores, metadata):
        """Calculate final confidence with metadata consideration"""
        base_score = np.mean(scores)
        
        # Consistency factor
        consistency = 1 - np.std(scores)
        
        # Metadata adjustment
        metadata_penalty = len(metadata.get('suspicious_patterns', [])) * 0.05
        
        # Temporal stability
        temporal_stability = self.calculate_temporal_stability(scores)
        
        # Final score calculation
        confidence_factors = {
            'base': base_score * 0.6,
            'consistency': consistency * 0.2,
            'temporal': temporal_stability * 0.15,
            'metadata': -metadata_penalty * 0.05
        }
        
        final_score = sum(confidence_factors.values())
        return np.clip(final_score, 0.0, 1.0)
    
    def calculate_temporal_stability(self, scores):
        """Calculate temporal stability across frames"""
        if len(scores) < 3:
            return 1.0
        
        # Calculate frame-to-frame differences
        differences = np.abs(np.diff(scores))
        
        # Penalize large jumps in predictions
        large_jumps = np.sum(differences > 0.3) / len(differences)
        stability = 1 - large_jumps
        
        return max(0, stability)
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)