# Face Recognition System

A modular Computer Vision system and Flask web application for face detection, 128-dimensional biometric facial embedding generation, and real-time identity recognition.

---

## ⚡ Core Features

- **Face Detection**: Fast facial localization using HOG (CPU) and CNN (GPU) models via `dlib` and OpenCV.
- **Biometric Encodings**: Generates 128D invariant face embeddings per detected face.
- **Known Face Database**: Persistent face database stored in serialized pickle format (`models/face_encodings.pkl`) supporting multiple training images per subject.
- **Web UI & REST API**: Modern Flask interface for detecting faces, matching against enrolled people, and registering new subjects.

---

## 📁 Project Architecture

```
face-recognition-system/
├── app.py                      # Flask web application & API entrypoint
├── config.py                   # Central directory & tolerance configuration
├── check_versions.py           # Dependency and package checker
├── requirements.txt            # Python dependencies
├── face_detection/
│   └── face_detector.py        # HOG/CNN face detection & bounding box drawing
├── face_recognition_module/
│   └── face_recognizer.py      # Face embedding extraction, comparison, & DB management
├── notebooks/
│   ├── capture_training_images.py  # Webcam training image capture utility
│   ├── test_face_detection.py      # Face detection pipeline verification
│   └── test_face_recognition.py    # Face recognition pipeline verification
└── templates/
    ├── index.html              # Landing overview dashboard
    ├── about.html              # Technical specs & architecture
    ├── detect_faces.html       # Image upload & face detection view
    ├── recognize_faces.html    # Biometric match & verification view
    └── train.html              # Subject enrollment & embedding generation
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Optional: Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on `dlib` on Windows:**
> If `pip install dlib` fails due to CMake or C++ Build Tools, you can install a prebuilt binary wheel for your Python version, e.g.:
> `pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl`

### 2. Run Web Application

```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Testing

```bash
# Run detection test
python notebooks/test_face_detection.py

# Run recognition test
python notebooks/test_face_recognition.py
```
