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

# Add to imports in app.py
from face_recognition.face_recognizer import FaceRecognizer
import pickle

# Add these routes to app.py:

@app.route('/recognize-faces', methods=['GET', 'POST'])
def recognize_faces():
    """Face recognition page"""
    recognizer = FaceRecognizer()
    
    # Try to load existing database
    database_path = os.path.join('models', 'face_database.pkl')
    if os.path.exists(database_path):
        recognizer.load_database(database_path)
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('recognize_faces.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('recognize_faces.html', error='No file selected')
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join('datasets', 'unknown_faces', filename)
            file.save(upload_path)
            
            # Recognize faces
            recognized_faces, result_image = recognizer.recognize_faces(upload_path, draw_results=True)
            
            if recognized_faces:
                # Save result image
                output_path = os.path.join('static', 'results', f'recognized_{filename}')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if result_image:
                    result_image.save(output_path)
                
                return render_template('recognize_faces.html', 
                                    faces=recognized_faces,
                                    result_image=f'results/recognized_{filename}',
                                    known_people=recognizer.known_face_names)
            else:
                return render_template('recognize_faces.html', error='No faces detected or recognized')
    
    return render_template('recognize_faces.html', known_people=recognizer.known_face_names)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train model page"""
    if request.method == 'POST':
        person_name = request.form.get('person_name')
        
        if not person_name:
            return render_template('train.html', error='Please enter a name')
        
        # In a real app, you'd handle file uploads here
        return render_template('train.html', 
                             success=f'Training interface for {person_name} would be implemented here')
    
    return render_template('train.html')