from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import os
import tempfile
from video_downloader import VideoDownloader
from werkzeug.utils import secure_filename
from threading import RLock
import traceback

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = None
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'deepfake_detection_secret_key'
app.static_folder = 'static'

# Initialize lightweight components. Load compute-heavy libraries only when analysis is requested,
# so Render can serve the frontend within the free instance memory limit.
detector = None
analyzer = None
# Serialize model loading and inference to prevent concurrent requests from
# loading duplicate TensorFlow models or exceeding the free instance memory cap.
MODEL_LOCK = RLock()
downloader = VideoDownloader()
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_model.tflite")

def get_detector():
    """Thread-safe lazy loading of the saved trained model."""
    global detector
    with MODEL_LOCK:
        if detector is None:
            from deepfake_detector import DeepfakeDetector

            if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
                raise RuntimeError(
                    "Trained model weights are unavailable on the server. "
                    "Ensure deepfake_model.tflite is generated during the build before deployment."
                )

            candidate = DeepfakeDetector()
            try:
                candidate.load_model(MODEL_PATH)
            except Exception as exc:
                raise RuntimeError(
                    "The trained model file could not be loaded. "
                    "Check the generated TFLite model and LiteRT runtime compatibility."
                ) from exc
            detector = candidate
        return detector

def get_analyzer():
    global analyzer
    if analyzer is None:
        from advanced_features import AdvancedAnalyzer
        analyzer = AdvancedAnalyzer()
    return analyzer

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health():
    """Lightweight deployment and trained-model availability check."""
    model_size = os.path.getsize(MODEL_PATH) if os.path.isfile(MODEL_PATH) else 0
    return jsonify({
        'status': 'ok',
        'trained_model_available': model_size >= 1_000_000,
        'model_size_bytes': model_size,
        'model_loaded': detector is not None,
        'model_format': 'LiteRT TFLite (weight-quantized)'
    })

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/check_url', methods=['POST'])
def check_url():
    try:
        url = request.json.get('url', '').strip()
        if not url:
            return jsonify({'valid': False, 'error': 'No URL provided'})
        if not (url.startswith('http://') or url.startswith('https://')):
            return jsonify({'valid': False, 'error': 'Invalid URL format'})
        if 'VIDEO_ID' in url or 'example.com' in url:
            return jsonify({'valid': False, 'error': 'Please use a real video URL, not a placeholder'})
        return jsonify({'valid': True})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze_video():
    tmp_path = None
    try:
        video_title = "Unknown"

        # Handle upload or URL
        if 'url' in request.form and request.form['url'].strip():
            url = request.form['url'].strip()
            tmp_path, video_title = downloader.download_video(url)
        elif 'video' in request.files:
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            if file and allowed_file(file.filename):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                    video_title = file.filename
            else:
                return jsonify({'error': 'Invalid file format'})
        else:
            return jsonify({'error': 'No video file or URL provided'})

        if not tmp_path:
            return jsonify({'error': 'Failed to process video'})

        # Only one inference at a time on the memory-limited free instance.
        with MODEL_LOCK:
            analysis_result = get_detector().predict_video(tmp_path)

        # Extract basic results
        if isinstance(analysis_result, dict):
            score = analysis_result['score']
            frames_extracted = analysis_result['frames_extracted']
            frames_analyzed = analysis_result['frames_analyzed']
            fake_probability = analysis_result['fake_probability']
            authenticity_percentage = analysis_result['authenticity_percentage']
        else:
            score = analysis_result
            frames_extracted = 30
            frames_analyzed = 10
            fake_probability = 1 - score
            authenticity_percentage = score * 100

        faces_detected = analysis_result.get('faces_detected', 0) if isinstance(analysis_result, dict) else 0
        metadata = analysis_result.get('metadata', {}) if isinstance(analysis_result, dict) else {}

        # Threshold logic
        if faces_detected > frames_analyzed * 0.7:
            real_threshold = 0.65
            fake_threshold = 0.35
        else:
            real_threshold = 0.75
            fake_threshold = 0.25

        if score >= real_threshold:
            result = "REAL VIDEO"
            confidence = f"{min(99, int(88 + (score - real_threshold) * 30))}%"
            color = "success"
            prediction = "AUTHENTIC"
        elif score <= fake_threshold:
            result = "FAKE VIDEO (DEEPFAKE DETECTED)"
            confidence = f"{min(99, int(88 + (fake_threshold - score) * 30))}%"
            color = "danger"
            prediction = "DEEPFAKE"
        else:
            uncertainty_reason = "Insufficient confidence"
            if faces_detected < frames_analyzed * 0.3:
                uncertainty_reason = "Limited facial data"
            elif len(metadata.get('suspicious_patterns', [])) > 0:
                uncertainty_reason = "Suspicious metadata detected"
            result = f"UNCERTAIN - {uncertainty_reason}"
            confidence = f"{int(40 + abs(score - 0.5) * 40)}%"
            color = "warning"
            prediction = "NEEDS_REVIEW"

        # Keep server-side post-processing light. Plotting libraries are intentionally
        # not imported for each request because this free instance has a 512 MiB RAM cap.
        chart_data = None
        heatmap_data = None
        frame_scores = analysis_result.get("frame_scores", []) if isinstance(analysis_result, dict) else []
        advanced_metrics = {}
        if len(frame_scores) > 1:
            import numpy as np
            scores = np.asarray(frame_scores, dtype=np.float32)
            differences = np.abs(np.diff(scores))
            mean_score = float(np.mean(scores))
            std_deviation = float(np.std(scores))
            advanced_metrics = {
                "mean_score": mean_score,
                "std_deviation": std_deviation,
                "consistency": float(max(0.0, 1.0 - std_deviation)),
                "temporal_stability": float(max(0.0, 1.0 - float(np.mean(differences)))) if len(differences) else 1.0,
                "confidence_interval": {
                    "lower": float(max(0.0, mean_score - 1.96 * std_deviation / np.sqrt(len(scores)))),
                    "upper": float(min(1.0, mean_score + 1.96 * std_deviation / np.sqrt(len(scores))))
                }
            }

        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        return jsonify({
            'score': round(float(score), 3),
            'result': result,
            'prediction': prediction,
            'confidence': confidence,
            'color': color,
            'video_title': video_title,
            'frames_extracted': frames_extracted,
            'frames_analyzed': frames_analyzed,
            'faces_detected': faces_detected,
            'fake_probability': round(fake_probability * 100, 1),
            'authenticity_percentage': round(authenticity_percentage, 1),
            'model_info': 'ResNet50 + custom classifier',
            'frame_scores': frame_scores,
            'metadata': metadata,
            'analysis_quality': {
                'face_detection_rate': round(faces_detected / max(frames_analyzed, 1) * 100, 1),
                'temporal_stability': advanced_metrics.get('temporal_stability', 0.8) if advanced_metrics else 0.8,
                'metadata_suspicious': len(metadata.get('suspicious_patterns', []))
            },
            'advanced_metrics': advanced_metrics,
            'chart_data': chart_data,
            'heatmap_data': heatmap_data
        })

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        error_details = traceback.format_exc()
        print(f"Analysis error: {error_details}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'details': 'Please check the video format and try again'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5002)