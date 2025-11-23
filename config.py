import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
KNOWN_FACES_DIR = os.path.join(DATASET_DIR, 'known_faces')
UNKNOWN_FACES_DIR = os.path.join(DATASET_DIR, 'unknown_faces')
TRAINING_DIR = os.path.join(DATASET_DIR, 'training')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FACE_ENCODINGS_PATH = os.path.join(MODELS_DIR, 'face_encodings.pkl')

# Application settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Face recognition settings
FACE_DETECTION_MODEL = 'hog'  # 'hog' or 'cnn'
NUMBER_OF_TIMES_TO_UPSAMPLE = 1
FACE_DETECTION_CONFIDENCE = 0.6

# Create directories if they don't exist
for directory in [KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, TRAINING_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("✅ Configuration loaded successfully!")