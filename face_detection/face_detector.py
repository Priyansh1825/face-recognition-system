# face_detection/face_detector.py
"""
Face Detection Module using OpenCV and face_recognition
"""

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

class FaceDetector:
    def __init__(self, model='hog'):
        """
        Initialize Face Detector
        
        Args:
            model (str): 'hog' for CPU, 'cnn' for GPU (more accurate but slower)
        """
        self.model = model
        print(f"✅ Face Detector initialized with {model.upper()} model")
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of face locations [(top, right, bottom, left)]
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                image, 
                number_of_times_to_upsample=1, 
                model=self.model
            )
            
            print(f"✅ Detected {len(face_locations)} face(s) in {os.path.basename(image_path)}")
            return face_locations, image
            
        except Exception as e:
            print(f"❌ Error detecting faces: {e}")
            return [], None
    
    def draw_face_boxes(self, image, face_locations, output_path=None):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: numpy array image
            face_locations: List of face locations
            output_path (str): Path to save the output image
            
        Returns:
            PIL.Image: Image with bounding boxes
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw rectangle around face
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            
            # Add face number
            draw.text((left, top - 20), f"Face {i+1}", fill="red")
            
            print(f"   👤 Face {i+1}: Position (Top:{top}, Right:{right}, Bottom:{bottom}, Left:{left})")
        
        if output_path:
            pil_image.save(output_path)
            print(f"💾 Output saved to: {output_path}")
        
        return pil_image
    
    def extract_face_encodings(self, image, face_locations):
        """
        Extract face encodings (embeddings) for recognition
        
        Args:
            image: numpy array image
            face_locations: List of face locations
            
        Returns:
            list: Face encodings for each detected face
        """
        try:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            print(f"✅ Extracted {len(face_encodings)} face encoding(s)")
            return face_encodings
        except Exception as e:
            print(f"❌ Error extracting face encodings: {e}")
            return []
    
    def display_results(self, original_image, detected_image, face_locations):
        """
        Display original and detected images side by side
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Detected faces image
        ax2.imshow(detected_image)
        ax2.set_title(f'Detected Faces: {len(face_locations)}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

def test_face_detection():
    """Test the face detection functionality"""
    print("🧪 TESTING FACE DETECTION")
    print("=" * 50)
    
    # Initialize detector
    detector = FaceDetector(model='hog')
    
    # Test image path (we'll create a sample test)
    test_image_path = "datasets/unknown_faces/test_face.jpg"
    
    # If no test image exists, we'll use a placeholder message
    if not os.path.exists(test_image_path):
        print("📝 No test image found. Please add an image to 'datasets/unknown_faces/test_face.jpg'")
        print("📝 Or let's create a simple test with webcam...")
        return
    
    # Detect faces
    face_locations, image = detector.detect_faces(test_image_path)
    
    if face_locations:
        # Draw bounding boxes
        detected_image = detector.draw_face_boxes(image, face_locations)
        
        # Extract encodings
        face_encodings = detector.extract_face_encodings(image, face_locations)
        
        # Display results
        detector.display_results(image, detected_image, face_locations)
        
        return face_encodings
    else:
        print("❌ No faces detected in the test image")
        return None

if __name__ == "__main__":
    test_face_detection()