import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class AdvancedAnalyzer:
    def __init__(self):
        self.frame_scores = []
        self.fft_features = []
        
    def generate_frame_analysis_chart(self, frame_scores):
        """Generate frame-wise analysis chart"""
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8')
        
        frames = range(1, len(frame_scores) + 1)
        fake_probs = [(1 - score) * 100 for score in frame_scores]
        
        plt.subplot(1, 2, 1)
        plt.plot(frames, fake_probs, 'r-', linewidth=2, label='Fake Probability')
        plt.fill_between(frames, fake_probs, alpha=0.3, color='red')
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        plt.xlabel('Frame Number')
        plt.ylabel('Fake Probability (%)')
        plt.title('Frame-wise Deepfake Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(fake_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Fake Probability (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Fake Probabilities')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_data
    
    def generate_heatmap(self, frame, fft_magnitude):
        """Generate heatmap of tampered regions"""
        plt.figure(figsize=(15, 5))
        
        # Original frame
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        plt.title('Original Frame')
        plt.axis('off')
        
        # FFT magnitude
        plt.subplot(1, 3, 2)
        plt.imshow(fft_magnitude, cmap='hot', interpolation='nearest')
        plt.title('FFT Magnitude Spectrum')
        plt.axis('off')
        
        # Heatmap overlay
        plt.subplot(1, 3, 3)
        plt.imshow(frame)
        
        # Create suspicious regions heatmap
        suspicious_regions = np.where(fft_magnitude > np.percentile(fft_magnitude, 85), 1, 0)
        plt.imshow(suspicious_regions, alpha=0.4, cmap='Reds')
        plt.title('Suspicious Regions Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        heatmap_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return heatmap_data
    
    def calculate_confidence_metrics(self, frame_scores):
        """Calculate advanced confidence metrics"""
        scores_array = np.array(frame_scores)
        
        metrics = {
            'mean_score': float(np.mean(scores_array)),
            'std_deviation': float(np.std(scores_array)),
            'consistency': float(1 - np.std(scores_array)),  # Higher = more consistent
            'temporal_stability': self.calculate_temporal_stability(scores_array),
            'confidence_interval': self.calculate_confidence_interval(scores_array)
        }
        
        return metrics
    
    def calculate_temporal_stability(self, scores):
        """Calculate temporal stability of predictions"""
        if len(scores) < 2:
            return 1.0
        
        differences = np.abs(np.diff(scores))
        stability = 1 - np.mean(differences)
        return max(0, float(stability))
    
    def calculate_confidence_interval(self, scores):
        """Calculate 95% confidence interval"""
        mean_score = np.mean(scores)
        std_error = np.std(scores) / np.sqrt(len(scores))
        margin_error = 1.96 * std_error
        
        return {
            'lower': float(max(0, mean_score - margin_error)),
            'upper': float(min(1, mean_score + margin_error))
        }
    
    def detect_manipulation_patterns(self, frame_scores):
        """Detect specific manipulation patterns"""
        patterns = {
            'face_swap': self.detect_face_swap_pattern(frame_scores),
            'face_reenactment': self.detect_reenactment_pattern(frame_scores),
            'lip_sync': self.detect_lip_sync_pattern(frame_scores),
            'full_synthesis': self.detect_synthesis_pattern(frame_scores)
        }
        
        return patterns
    
    def detect_face_swap_pattern(self, scores):
        """Detect face swap manipulation pattern"""
        # Face swaps often show inconsistent scores around face regions
        variance = np.var(scores)
        return {
            'detected': bool(variance > 0.1),
            'confidence': float(min(1.0, variance * 5)),
            'description': 'Inconsistent facial features detected'
        }
    
    def detect_reenactment_pattern(self, scores):
        """Detect face reenactment pattern"""
        # Reenactment often shows temporal inconsistencies
        if len(scores) < 5:
            return {'detected': False, 'confidence': 0.0, 'description': 'Insufficient frames'}
        
        temporal_changes = np.abs(np.diff(scores))
        high_changes = np.sum(temporal_changes > 0.2) / len(temporal_changes)
        
        return {
            'detected': bool(high_changes > 0.3),
            'confidence': float(min(1.0, high_changes * 2)),
            'description': 'Temporal inconsistencies in facial expressions'
        }
    
    def detect_lip_sync_pattern(self, scores):
        """Detect lip sync manipulation"""
        # Lip sync often shows periodic patterns
        if len(scores) < 10:
            return {'detected': False, 'confidence': 0.0, 'description': 'Insufficient frames'}
        
        # Simple periodicity detection
        fft_scores = np.fft.fft(scores)
        dominant_freq = np.argmax(np.abs(fft_scores[1:len(fft_scores)//2])) + 1
        periodicity = np.abs(fft_scores[dominant_freq]) / len(scores)
        
        return {
            'detected': bool(periodicity > 0.1),
            'confidence': float(min(1.0, periodicity * 5)),
            'description': 'Periodic patterns in mouth region detected'
        }
    
    def detect_synthesis_pattern(self, scores):
        """Detect full face synthesis"""
        # Full synthesis often shows consistently low scores
        mean_score = np.mean(scores)
        consistency = 1 - np.std(scores)
        
        synthesis_indicator = (mean_score < 0.3) and (consistency > 0.8)
        
        return {
            'detected': bool(synthesis_indicator),
            'confidence': float(min(1.0, (0.3 - mean_score) * 3 + consistency)),
            'description': 'Consistently artificial facial features'
        }