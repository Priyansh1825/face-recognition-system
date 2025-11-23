# notebooks/test_face_detection.py
"""
Comprehensive test script for face detection
"""

import sys
import os
sys.path.append('..')

from face_detection.face_detector import FaceDetector
import cv2
import matplotlib.pyplot as plt

def test_with_sample_images():
    """Test face detection with sample images"""
    print("🧪 COMPREHENSIVE FACE DETECTION TEST")
    print("=" * 50)
    
    # Initialize detector
    detector_hog = FaceDetector(model='hog')
    
    # Test with multiple images if available
    test_images_dir = "../datasets/unknown_faces"
    
    if not os.path.exists(test_images_dir):
        print("❌ Test images directory not found")
        return
    
    image_files = [f for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("📝 No test images found. Please run 'capture_test_images.py' first")
        return
    
    print(f"📁 Found {len(image_files)} test image(s)")
    
    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        print(f"\n🔍 Processing: {image_file}")
        print("-" * 30)
        
        # Detect faces
        face_locations, image = detector_hog.detect_faces(image_path)
        
        if face_locations:
            # Draw bounding boxes
            detected_image = detector_hog.draw_face_boxes(image, face_locations)
            
            # Display results
            detector_hog.display_results(image, detected_image, face_locations)
            
            # Test face encodings
            encodings = detector_hog.extract_face_encodings(image, face_locations)
            print(f"📊 Face encoding dimensions: {len(encodings[0]) if encodings else 0}")
        else:
            print("❌ No faces detected")
        
        print("✅ Test completed for", image_file)

def compare_detection_models():
    """Compare HOG vs CNN models"""
    print("\n🔬 COMPARING DETECTION MODELS")
    print("=" * 40)
    
    test_images_dir = "../datasets/unknown_faces"
    image_files = [f for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        return
    
    # Test first image with both models
    test_image = os.path.join(test_images_dir, image_files[0])
    
    # HOG model (faster, less accurate)
    detector_hog = FaceDetector(model='hog')
    hog_locations, image = detector_hog.detect_faces(test_image)
    
    # CNN model (slower, more accurate - requires GPU for best performance)
    detector_cnn = FaceDetector(model='cnn')
    cnn_locations, _ = detector_cnn.detect_faces(test_image)
    
    print(f"\n📊 MODEL COMPARISON:")
    print(f"   HOG Model:  {len(hog_locations)} faces detected")
    print(f"   CNN Model:  {len(cnn_locations)} faces detected")
    
    if len(hog_locations) != len(cnn_locations):
        print("   ⚠️  Models detected different number of faces!")

if __name__ == "__main__":
    test_with_sample_images()
    compare_detection_models()