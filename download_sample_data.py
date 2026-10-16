import os
import sys
import urllib.request
import ssl

# Set working directory to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, FACE_ENCODINGS_PATH
from face_recognition_module.face_recognizer import FaceRecognizer

# Create directories
obama_dir = os.path.join(KNOWN_FACES_DIR, "Barack_Obama")
biden_dir = os.path.join(KNOWN_FACES_DIR, "Joe_Biden")
os.makedirs(obama_dir, exist_ok=True)
os.makedirs(biden_dir, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Dataset URLs (from official face_recognition repo)
# Distinct images for Training vs Testing:
dataset_files = [
    # --- TRAINING DATA (Known references) ---
    {
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        "dest": os.path.join(obama_dir, "obama_train_1.jpg"),
        "type": "Training (Barack Obama)"
    },
    {
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        "dest": os.path.join(biden_dir, "biden_train_1.jpg"),
        "type": "Training (Joe Biden)"
    },
    # --- TEST DATA (Completely different images) ---
    {
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
        "dest": os.path.join(UNKNOWN_FACES_DIR, "test_obama_different_photo.jpg"),
        "type": "Test: Different photo of Obama (Should match Barack Obama)"
    },
    {
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg",
        "dest": os.path.join(UNKNOWN_FACES_DIR, "test_multi_face_obama_and_biden.jpg"),
        "type": "Test: Group photo with Obama & Biden (Should detect & recognize both)"
    },
    {
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.jpg",
        "dest": os.path.join(UNKNOWN_FACES_DIR, "test_unregistered_unknown.jpg"),
        "type": "Test: Unenrolled person (Should identify as Unknown)"
    }
]

# Configure SSL & User-Agent
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

print("📥 DOWNLOADING TRAINING & TESTING DATASETS")
print("=" * 60)

for item in dataset_files:
    print(f"\n[FETCH] Downloading {item['type']}...")
    try:
        req = urllib.request.Request(item['url'], headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
            with open(item['dest'], 'wb') as f:
                f.write(data)
            print(f"  ✅ Saved to: {os.path.relpath(item['dest'], BASE_DIR)} ({len(data):,} bytes)")
    except Exception as e:
        print(f"  ❌ Error downloading {item['url']}: {e}")

print("\n" + "=" * 60)
print("🧠 ENROLLING KNOWN FACES INTO DATABASE")
print("=" * 60)

recognizer = FaceRecognizer(tolerance=0.6)
success = recognizer.load_known_faces(KNOWN_FACES_DIR)
if success and recognizer.known_face_names:
    recognizer.save_database(FACE_ENCODINGS_PATH)
    print(f"\n🎉 Successfully enrolled: {recognizer.known_face_names}")
    print(f"💾 Embeddings database saved to: {FACE_ENCODINGS_PATH}")
else:
    print("\n❌ Failed to load face encodings from training images.")
