# notebooks/capture_test_images.py
"""
Utility to capture test images using webcam
"""

import cv2
import os
from datetime import datetime

def capture_test_image():
    """Capture an image using webcam for testing"""
    
    # Create test directory if it doesn't exist
    test_dir = "../datasets/unknown_faces"
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    print("📷 Webcam activated - Press SPACE to capture, ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Display frame
        cv2.imshow('Press SPACE to capture, ESC to exit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("🚪 Exiting without capture")
            break
        elif key == 32:  # SPACE key
            # Save captured image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_face_{timestamp}.jpg"
            filepath = os.path.join(test_dir, filename)
            
            cv2.imwrite(filepath, frame)
            print(f"✅ Image saved: {filepath}")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def create_sample_dataset():
    """Create a simple sample dataset for testing"""
    print("📁 CREATING SAMPLE DATASET STRUCTURE")
    
    # Create known faces directory structure
    known_faces_dir = "../datasets/known_faces"
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Create sample person directories
    sample_people = ["person_a", "person_b", "person_c"]
    
    for person in sample_people:
        person_dir = os.path.join(known_faces_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        print(f"✅ Created directory: {person_dir}")
    
    print("📝 Please add training images to:")
    for person in sample_people:
        print(f"   📂 datasets/known_faces/{person}/")
    
    print("\n💡 You can:")
    print("   1. Use webcam to capture images (run this script)")
    print("   2. Add existing photos to the directories")
    print("   3. Download sample faces from the internet")

if __name__ == "__main__":
    print("🖼️ TEST IMAGE CAPTURE UTILITY")
    print("=" * 40)
    
    print("1. Capture test image using webcam")
    print("2. Create sample dataset structure")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        capture_test_image()
    elif choice == "2":
        create_sample_dataset()
    else:
        print("❌ Invalid choice")