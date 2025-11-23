# notebooks/test_face_recognition.py
"""
Test script for Face Recognition System
"""

import sys
import os
sys.path.append('..')

from face_recognition.face_recognizer import FaceRecognizer
import matplotlib.pyplot as plt

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
    
    # Save database
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    recognizer.save_database(os.path.join(models_dir, "face_database.pkl"))

def add_new_person_demo():
    """Demo: How to add a new person to the database"""
    print("\n👤 DEMO: ADDING NEW PERSON")
    print("=" * 40)
    
    recognizer = FaceRecognizer()
    
    # Try to load existing database
    database_path = "../models/face_database.pkl"
    if os.path.exists(database_path):
        recognizer.load_database(database_path)
    
    # Example: Add a new person (you'll need actual image paths)
    print("💡 To add a new person, you need:")
    print("   1. Create directory: datasets/known_faces/PersonName/")
    print("   2. Add several photos of the person")
    print("   3. Run the recognition test again")
    
    recognizer.list_known_people()

if __name__ == "__main__":
    test_face_recognition()
    add_new_person_demo()