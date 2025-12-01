"""
Flask Web API for Plant Disease Detection (WITH VIDEO SUPPORT)
================================================================
Wraps the YOLO + MobileNet pipeline for web deployment
Supports: Images and Video files
"""

from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
import json
import io
import base64
import tempfile

# Download models if not present (for Render deployment)
try:
    from download_models import download_models
    download_models()
except Exception as e:
    print(f"⚠️  Model download check: {e}")

# Import the pipeline functions
from yolo_mobilenet_pipeline import load_models, detect_and_classify, classify_image, process_video

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload for videos
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models (loaded once at startup)
yolo_model = None
mobilenet_model = None
class_labels = None

# Flag to track if models are being loaded
models_loading = False


def load_models_async():
    """Load models asynchronously after server starts"""
    global yolo_model, mobilenet_model, class_labels, models_loading
    
    if yolo_model is not None:
        return  # Already loaded
    
    if models_loading:
        return  # Already loading
    
    models_loading = True
    try:
        print("🚀 Loading models in background...")
        yolo_model, mobilenet_model, class_labels = load_models()
        print("✅ Models loaded successfully!")
        print(f"   YOLO model: {type(yolo_model)}")
        print(f"   MobileNet model: {type(mobilenet_model)}")
        print(f"   Class labels: {len(class_labels) if class_labels else 0} classes")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
    finally:
        models_loading = False


def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False


@app.route('/')
def index():
    """Serve the main dashboard"""
    # Trigger model loading on first request
    if yolo_model is None and not models_loading:
        import threading
        threading.Thread(target=load_models_async, daemon=True).start()
    return render_template('index_video.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    # Trigger model loading on health check
    if yolo_model is None and not models_loading:
        import threading
        threading.Thread(target=load_models_async, daemon=True).start()
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': yolo_model is not None and mobilenet_model is not None,
        'models_loading': models_loading,
        'features': ['image_prediction', 'video_processing']
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for images
    Accepts: multipart/form-data with 'image' file
    Returns: JSON with predictions
    """
    try:
        # Validate models are loaded
        if models_loading:
            return jsonify({'error': 'Models are still loading. Please wait a moment and try again.'}), 503
        
        if yolo_model is None or mobilenet_model is None or class_labels is None:
            return jsonify({'error': 'Models not loaded yet. Please refresh the page and try again.'}), 503
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'File type not allowed. Use: jpg, jpeg, png, bmp'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get processing mode (default: full pipeline)
        use_yolo = request.form.get('use_yolo', 'true').lower() == 'true'
        
        # Run prediction
        if use_yolo:
            results, annotated_img = detect_and_classify(
                filepath, yolo_model, mobilenet_model, class_labels, save_output=False
            )
        else:
            results, annotated_img = classify_image(
                filepath, mobilenet_model, class_labels, save_output=False
            )
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format results
        formatted_results = []
        for result in results:
            plant_disease = result['class'].split('___')
            plant = plant_disease[0] if len(plant_disease) > 0 else "Unknown"
            disease = plant_disease[1] if len(plant_disease) > 1 else "Unknown"
            is_healthy = "healthy" in disease.lower()
            
            formatted_results.append({
                'plant': plant,
                'disease': disease,
                'confidence': float(result['confidence']),
                'is_healthy': is_healthy,
                'bbox': result.get('bbox', None)
            })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'predictions': formatted_results,
            'annotated_image': f'data:image/jpeg;base64,{img_base64}',
            'num_detections': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-video', methods=['POST'])
def predict_video():
    """
    Video processing endpoint
    Accepts: multipart/form-data with 'video' file
    Returns: Processed video file
    """
    try:
        # Validate models are loaded
        if models_loading:
            return jsonify({'error': 'Models are still loading. Please wait a moment and try again.'}), 503
        
        if yolo_model is None or mobilenet_model is None or class_labels is None:
            return jsonify({'error': 'Models not loaded yet. Please refresh the page and try again.'}), 503
        
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'File type not allowed. Use: mp4, avi, mov, mkv'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Create output path
        output_filename = Path(filename).stem + '_result.mp4'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Process video
        process_video(input_path, yolo_model, mobilenet_model, class_labels, output_path)
        
        # Send processed video
        response = send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=output_filename
        )
        
        # Clean up files after sending (in a callback)
        @response.call_on_close
        def cleanup():
            try:
                os.remove(input_path)
                os.remove(output_path)
            except:
                pass
        
        return response
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/classes')
def get_classes():
    """Return all available disease classes"""
    return jsonify({
        'num_classes': len(class_labels),
        'classes': class_labels
    })


if __name__ == '__main__':
    # Models are already loaded at module level
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
