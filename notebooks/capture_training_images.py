# notebooks/capture_training_images.py
"""
Utility to capture training images for known people
"""

import cv2
import os
from datetime import datetime

def capture_training_images(person_name, num_images=5):
    """
    Capture training images for a specific person
    
    Args:
        person_name (str): Name of the person
        num_images (int): Number of images to capture
    """
    
    # Create person directory
    person_dir = f"../datasets/known_faces/{person_name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    print(f"📷 Capturing {num_images} training images for {person_name}")
    print("💡 Press SPACE to capture, ESC to exit early")
    
    captured_count = 0
    
    while captured_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Display instructions on frame
        cv2.putText(frame, f"Capturing for: {person_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {captured_count}/{num_images}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture, ESC: Exit", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Capture Training Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("🚪 Exiting early")
            break
        elif key == 32:  # SPACE key
            # Save captured image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{timestamp}_{captured_count+1}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            cv2.imwrite(filepath, frame)
            print(f"✅ Image {captured_count+1} saved: {filename}")
            captured_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"🎉 Successfully captured {captured_count} images for {person_name}")
    print(f"📁 Images saved in: {person_dir}")

def main():
    """Main function for training image capture"""
    print("🎯 TRAINING IMAGE CAPTURE UTILITY")
    print("=" * 40)
    
    person_name = input("Enter person's name: ").strip()
    
    if not person_name:
        print("❌ Please enter a valid name")
        return
    
    try:
        num_images = int(input("Number of images to capture (default 5): ") or "5")
    except ValueError:
        num_images = 5
    
    print(f"\n📋 Summary:")
    print(f"   Person: {person_name}")
    print(f"   Images: {num_images}")
    print(f"   Location: datasets/known_faces/{person_name}/")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    
    if confirm == 'y':
        capture_training_images(person_name, num_images)
    else:
        print("❌ Cancelled")

if __name__ == "__main__":
    main()