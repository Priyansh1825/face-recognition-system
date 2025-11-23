from flask import Flask, render_template, request, jsonify
import os
from config import BASE_DIR

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'project': 'Face Recognition System',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Face Recognition System...")
    print(f"📁 Project Directory: {BASE_DIR}")
    print("🌐 Starting web server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # Add these imports to app.py
from face_detection.face_detector import FaceDetector
import os
from werkzeug.utils import secure_filename

# Add these routes to app.py (before if __name__ == '__main__')

@app.route('/detect-faces', methods=['GET', 'POST'])
def detect_faces():
    """Face detection page"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detect_faces.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect_faces.html', error='No file selected')
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join('datasets', 'unknown_faces', filename)
            file.save(upload_path)
            
            # Detect faces
            detector = FaceDetector()
            face_locations, image = detector.detect_faces(upload_path)
            
            if face_locations:
                # Create output image with bounding boxes
                output_path = os.path.join('static', 'results', f'detected_{filename}')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                detected_image = detector.draw_face_boxes(image, face_locations, output_path)
                
                return render_template('detect_faces.html', 
                                    faces_detected=len(face_locations),
                                    result_image=f'results/detected_{filename}')
            else:
                return render_template('detect_faces.html', error='No faces detected')
    
    return render_template('detect_faces.html')

# Create the template directory for results
os.makedirs('static/results', exist_ok=True)