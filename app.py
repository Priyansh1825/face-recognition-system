import os
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import (
    BASE_DIR,
    KNOWN_FACES_DIR,
    UNKNOWN_FACES_DIR,
    MODELS_DIR,
    FACE_ENCODINGS_PATH,
    ALLOWED_EXTENSIONS,
)
try:
    from face_detection.face_detector import FaceDetector
except ImportError:
    FaceDetector = None

try:
    from face_recognition_module.face_recognizer import FaceRecognizer
except ImportError:
    FaceRecognizer = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face-recognition-system-secret-key'

# Ensure required runtime directories exist
RESULTS_DIR = os.path.join(BASE_DIR, 'static', 'results')
for path in [RESULTS_DIR, KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Home landing page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """About page with technical overview"""
    return render_template('about.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'project': 'Face Recognition System',
        'version': '1.0.0'
    })


@app.route('/detect-faces', methods=['GET', 'POST'])
def detect_faces():
    """Face detection page"""
    if request.method == 'POST':
        if FaceDetector is None:
            return render_template('detect_faces.html', error='OpenCV / face_recognition library is not installed in the Python environment. Please run: pip install -r requirements.txt')

        if 'file' not in request.files:
            return render_template('detect_faces.html', error='No file selected')

        file = request.files['file']
        if file.filename == '':
            return render_template('detect_faces.html', error='No file selected')

        if not is_allowed_file(file.filename):
            return render_template('detect_faces.html', error='Unsupported file type')

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        file.save(upload_path)

        detector = FaceDetector()
        face_locations, image = detector.detect_faces(upload_path)

        if face_locations and image is not None:
            output_filename = f'detected_{filename}'
            output_path = os.path.join(RESULTS_DIR, output_filename)
            detector.draw_face_boxes(image, face_locations, output_path)

            return render_template(
                'detect_faces.html',
                faces_detected=len(face_locations),
                result_image=f'results/{output_filename}',
                locations=face_locations,
                original_filename=filename
            )
        else:
            return render_template('detect_faces.html', error='No faces detected in the uploaded image')

    return render_template('detect_faces.html')


@app.route('/recognize-faces', methods=['GET', 'POST'])
def recognize_faces():
    """Face recognition page against the known face database"""
    if FaceRecognizer is None:
        return render_template('recognize_faces.html', error='face_recognition module dependencies are not installed. Run: pip install -r requirements.txt', known_people=[])

    recognizer = FaceRecognizer()

    # Load existing database if available, otherwise attempt loading from known_faces directory
    database_path = FACE_ENCODINGS_PATH
    if os.path.exists(database_path):
        recognizer.load_database(database_path)
    elif os.path.exists(KNOWN_FACES_DIR):
        recognizer.load_known_faces(KNOWN_FACES_DIR)
        recognizer.save_database(database_path)

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('recognize_faces.html', error='No file selected', known_people=recognizer.known_face_names)

        file = request.files['file']
        if file.filename == '':
            return render_template('recognize_faces.html', error='No file selected', known_people=recognizer.known_face_names)

        if not is_allowed_file(file.filename):
            return render_template('recognize_faces.html', error='Unsupported file type', known_people=recognizer.known_face_names)

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        file.save(upload_path)

        recognized_faces, result_image = recognizer.recognize_faces(upload_path, draw_results=True)

        if recognized_faces:
            output_filename = f'recognized_{filename}'
            output_path = os.path.join(RESULTS_DIR, output_filename)
            if result_image:
                result_image.save(output_path)

            return render_template(
                'recognize_faces.html',
                faces=recognized_faces,
                result_image=f'results/{output_filename}',
                known_people=recognizer.known_face_names,
                original_filename=filename
            )
        else:
            return render_template(
                'recognize_faces.html',
                error='No faces detected or recognized in the image',
                known_people=recognizer.known_face_names
            )

    return render_template('recognize_faces.html', known_people=recognizer.known_face_names)


@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train / register a new person with multiple reference photos"""
    if FaceRecognizer is None:
        return render_template('train.html', error='face_recognition module dependencies are not installed. Run: pip install -r requirements.txt', known_people=[])

    recognizer = FaceRecognizer()
    database_path = FACE_ENCODINGS_PATH

    if os.path.exists(database_path):
        recognizer.load_database(database_path)

    if request.method == 'POST':
        person_name = request.form.get('person_name', '').strip()

        if not person_name:
            return render_template('train.html', error='Please enter a valid person name', known_people=recognizer.known_face_names)

        uploaded_files = request.files.getlist('files')
        valid_files = [f for f in uploaded_files if f.filename and is_allowed_file(f.filename)]

        if not valid_files:
            return render_template('train.html', error='Please select at least one valid image file (PNG, JPG, JPEG)', known_people=recognizer.known_face_names)

        # Create target directory for this person
        clean_person_dir = os.path.join(KNOWN_FACES_DIR, secure_filename(person_name) or person_name)
        os.makedirs(clean_person_dir, exist_ok=True)

        saved_image_paths = []
        for file in valid_files:
            filename = secure_filename(file.filename)
            dest_path = os.path.join(clean_person_dir, filename)
            file.save(dest_path)
            saved_image_paths.append(dest_path)

        # Update database with new person encodings
        success = recognizer.add_new_person(person_name, saved_image_paths)
        if success:
            recognizer.save_database(database_path)
            return render_template(
                'train.html',
                success=f"Successfully enrolled '{person_name}' with {len(saved_image_paths)} image(s)!",
                known_people=recognizer.known_face_names
            )
        else:
            return render_template(
                'train.html',
                error=f"Could not extract facial encodings from the provided images for '{person_name}'. Ensure faces are clearly visible.",
                known_people=recognizer.known_face_names
            )

    return render_template('train.html', known_people=recognizer.known_face_names)


if __name__ == '__main__':
    print("🚀 Starting Face Recognition System...")
    print(f"📁 Project Directory: {BASE_DIR}")
    print("🌐 Starting web server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)