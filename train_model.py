import os
import pandas as pd
from deepfake_detector import DeepfakeDetector
import argparse

def train_deepfake_model(data_dir, epochs=20, batch_size=16):
    """
    Train deepfake detection model on DFDC dataset
    
    Expected directory structure:
    data_dir/
    ├── real/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── fake/
        ├── video1.mp4
        └── video2.mp4
    """
    
    print("🚀 Starting Deepfake Detection Model Training")
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Prepare data paths and labels
    video_paths = []
    labels = []
    
    # Real videos (label = 1)
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        for video_file in os.listdir(real_dir):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(real_dir, video_file))
                labels.append(1)
    
    # Fake videos (label = 0)
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        for video_file in os.listdir(fake_dir):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(fake_dir, video_file))
                labels.append(0)
    
    print(f"📊 Dataset Summary:")
    print(f"   Real videos: {sum(labels)}")
    print(f"   Fake videos: {len(labels) - sum(labels)}")
    print(f"   Total videos: {len(video_paths)}")
    
    if len(video_paths) == 0:
        print("❌ No videos found! Please check your data directory structure.")
        return
    
    # Train model
    print("\n🔥 Training model...")
    history = detector.train(video_paths, labels, epochs=epochs, batch_size=batch_size)
    
    # Save model
    model_path = "deepfake_model.h5"
    detector.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Print final metrics
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n📈 Training Results:")
    print(f"   Final Training Accuracy: {final_accuracy:.4f}")
    print(f"   Final Validation Accuracy: {final_val_accuracy:.4f}")
    
    return detector

def create_sample_structure(base_dir):
    """Create sample directory structure for DFDC dataset"""
    os.makedirs(os.path.join(base_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'fake'), exist_ok=True)
    
    print(f"📁 Created directory structure at {base_dir}")
    print("   Place real videos in: real/")
    print("   Place fake videos in: fake/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Directory containing real/ and fake/ subdirectories')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Training batch size')
    parser.add_argument('--create_structure', action='store_true',
                       help='Create sample directory structure')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_structure(args.data_dir)
    else:
        if not os.path.exists(args.data_dir):
            print(f"❌ Data directory {args.data_dir} not found!")
            print("Use --create_structure to create the directory structure first.")
        else:
            train_deepfake_model(args.data_dir, args.epochs, args.batch_size)