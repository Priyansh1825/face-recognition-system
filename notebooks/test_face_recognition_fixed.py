# notebooks/test_face_recognition_fixed.py
"""
FIXED VERSION - Face Recognition Test (No Import Conflicts)
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append('..')

# Import our custom modules directly
try:
    # Import the face recognizer class dynamically to avoid static import resolution errors
    import importlib
    module = importlib.import_module('face_recognition_module.face_recognizer')
    FaceRecognizer = module.FaceRecognizer
    print("✅ Successfully imported FaceRecognizer")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("🔧 Creating inline version...")
    
    # If import fails, define the class inline
    import face_recognition as fr
    import pickle
    
    class FaceRecognizer:
        def __init__(self, tolerance=0.6):
            self.tolerance = tolerance
            self.known_face_encodings = []
            self.known_face_names = []
            self.face_database = {}
            print(f"✅ Face Recognizer initialized (tolerance: {tolerance})")
        
        def load_known_faces(self, known_faces_dir):
            print(f"📁 Loading known faces from: {known_faces_dir}")
            
            if not os.path.exists(known_faces_dir):
                print(f"❌ Directory not found: {known_faces_dir}")
                return False
            
            self.known_face_encodings = []
            self.known_face_names = []
            self.face_database = {}
            
            for person_name in os.listdir(known_faces_dir):
                person_dir = os.path.join(known_faces_dir, person_name)
                
                if os.path.isdir(person_dir):
                    print(f"👤 Loading faces for: {person_name}")
                    person_encodings = []
                    
                    for image_file in os.listdir(person_dir):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(person_dir, image_file)
                            
                            try:
                                image = fr.load_image_file(image_path)
                                face_encodings = fr.face_encodings(image)
                                
                                if face_encodings:
                                    person_encodings.append(face_encodings[0])
                                    print(f"   ✅ {image_file}: Face encoded")
                                else:
                                    print(f"   ❌ {image_file}: No face detected")
                                    
                            except Exception as e:
                                print(f"   ❌ {image_file}: Error - {e}")
                    
                    if person_encodings:
                        primary_encoding = person_encodings[0]
                        self.known_face_encodings.append(primary_encoding)
                        self.known_face_names.append(person_name)
                        self.face_database[person_name] = person_encodings
                        print(f"   📊 {person_name}: {len(person_encodings)} face encoding(s) loaded")
                    else:
                        print(f"   ⚠️  {person_name}: No valid faces found")
            
            print(f"✅ Loaded {len(self.known_face_names)} people from database")
            return True
        
        def recognize_faces(self, image_path, draw_results=True):
            print(f"🔍 Recognizing faces in: {os.path.basename(image_path)}")
            
            try:
                unknown_image = fr.load_image_file(image_path)
                face_locations = fr.face_locations(unknown_image)
                face_encodings = fr.face_encodings(unknown_image, face_locations)
                
                print(f"📊 Found {len(face_encodings)} face(s) to recognize")
                
                recognized_faces = []
                
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    matches = fr.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=self.tolerance
                    )
                    
                    face_distances = fr.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1
                    
                    face_info = {
                        'face_number': i + 1,
                        'location': face_location,
                        'encoding': face_encoding,
                        'matches': matches,
                        'distances': face_distances,
                        'best_match_index': best_match_index
                    }
                    
                    if best_match_index >= 0 and matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        distance = face_distances[best_match_index]
                        face_info.update({
                            'name': name,
                            'confidence': 1 - distance,
                            'distance': distance,
                            'recognized': True
                        })
                        print(f"   ✅ Face {i+1}: Recognized as {name} (confidence: {1-distance:.2f})")
                    else:
                        face_info.update({
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'distance': float('inf'),
                            'recognized': False
                        })
                        print(f"   ❌ Face {i+1}: Unknown person")
                    
                    recognized_faces.append(face_info)
                
                if draw_results and recognized_faces:
                    result_image = self._draw_recognition_results(unknown_image, recognized_faces)
                    return recognized_faces, result_image
                else:
                    return recognized_faces, None
                   
            except Exception as e:
                print(f"❌ Error recognizing faces: {e}")
                return [], None
        
        def _draw_recognition_results(self, image, recognized_faces):
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            for face_info in recognized_faces:
                top, right, bottom, left = face_info['location']
                
                if face_info['recognized']:
                    color = "green"
                    label = f"{face_info['name']} ({face_info['confidence']:.2f})"
                else:
                    color = "red"
                    label = "Unknown"
                
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                text_bbox = draw.textbbox((left, bottom), label)
                text_width = text_bbox[2] - text_bbox[0]
                draw.rectangle([left, bottom, left + text_width + 10, bottom + 25], fill=color)
                draw.text((left + 5, bottom), label, fill="white")
            
            return pil_image
        
        def list_known_people(self):
            print("📋 KNOWN PEOPLE IN DATABASE:")
            print("=" * 30)
            
            for name, encodings in self.face_database.items():
                print(f"👤 {name}: {len(encodings)} face encoding(s)")
            
            if not self.face_database:
                print("📝 Database is empty. Add some people first!")

def test_face_recognition():
    """Test the complete face recognition pipeline"""
    print("🧪 FACE RECOGNITION TEST")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = FaceRecognizer(tolerance=0.6)
    
    # Load known faces
    known_faces_dir = "../datasets/known_faces"
    
    if not os.path.exists(known_faces_dir):
        print("❌ Known faces directory not found")
        print("💡 Please create directories and add training images:")
        print("   datasets/known_faces/person_name/image1.jpg")
        return
    
    # Load known faces
    success = recognizer.load_known_faces(known_faces_dir)
    
    if not success or not recognizer.known_face_names:
        print("❌ No known faces loaded. Please add training images.")
        print("💡 You can use the capture utility to add people.")
        return
    
    # List known people
    recognizer.list_known_people()
    
    # Test recognition on unknown images
    unknown_faces_dir = "../datasets/unknown_faces"
    
    if not os.path.exists(unknown_faces_dir):
        print("❌ Unknown faces directory not found")
        return
    
    image_files = [f for f in os.listdir(unknown_faces_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("📝 No test images found. Please add images to 'datasets/unknown_faces/'")
        return
    
    print(f"\n🔍 Testing recognition on {len(image_files)} image(s)")
    
    for image_file in image_files:
        image_path = os.path.join(unknown_faces_dir, image_file)
        print(f"\n📷 Processing: {image_file}")
        print("-" * 30)
        
        # Recognize faces
        recognized_faces, result_image = recognizer.recognize_faces(image_path, draw_results=True)
        
        if recognized_faces:
            # Display results
            if result_image:
                plt.figure(figsize=(12, 8))
                plt.imshow(result_image)
                plt.title(f'Recognition Results: {image_file}')
                plt.axis('off')
                plt.show()
            
            # Print detailed results
            print("\n📊 DETAILED RESULTS:")
            for face in recognized_faces:
                if face['recognized']:
                    print(f"   ✅ Face {face['face_number']}: {face['name']} "
                          f"(Confidence: {face['confidence']:.2f}, Distance: {face['distance']:.2f})")
                else:
                    print(f"   ❌ Face {face['face_number']}: Unknown "
                          f"(Min Distance: {min(face['distances']):.2f})")
        else:
            print("❌ No faces detected or recognized")

if __name__ == "__main__":
    test_face_recognition()