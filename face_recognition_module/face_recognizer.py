# face_recognition/face_recognizer.py
"""
Face Recognition Module - Encoding and Matching Faces
"""

import face_recognition
import numpy as np
import pickle
import os
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class FaceRecognizer:
    def __init__(self, tolerance=0.6):
        """
        Initialize Face Recognizer

        Args:
            tolerance (float): How much distance between faces to consider it a match.
                             Lower is more strict.
        """
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_database = {}

        print(f"✅ Face Recognizer initialized (tolerance: {tolerance})")

    def load_known_faces(self, known_faces_dir):
        """
        Load known faces from directory structure

        Args:
            known_faces_dir (str): Path to directory with person subdirectories
        """
        print(f"📁 Loading known faces from: {known_faces_dir}")

        if not os.path.exists(known_faces_dir):
            print(f"❌ Directory not found: {known_faces_dir}")
            return False

        self.known_face_encodings = []
        self.known_face_names = []
        self.face_database = {}

        # Each person should have their own directory
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)

            if os.path.isdir(person_dir):
                print(f"👤 Loading faces for: {person_name}")

                person_encodings = []

                # Load all images for this person
                for image_file in os.listdir(person_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir, image_file)

                        try:
                            # Load and encode face
                            image = face_recognition.load_image_file(image_path)
                            face_encodings = face_recognition.face_encodings(image)

                            if face_encodings:
                                person_encodings.append(face_encodings[0])
                                print(f"   ✅ {image_file}: Face encoded")
                            else:
                                print(f"   ❌ {image_file}: No face detected")

                        except Exception as e:
                            print(f"   ❌ {image_file}: Error - {e}")

                if person_encodings:
                    # Use the first encoding as primary, store all for this person
                    primary_encoding = person_encodings[0]
                    self.known_face_encodings.append(primary_encoding)
                    self.known_face_names.append(person_name)

                    # Store all encodings for this person in database
                    self.face_database[person_name] = person_encodings

                    print(f"   📊 {person_name}: {len(person_encodings)} face encoding(s) loaded")
                else:
                    print(f"   ⚠️  {person_name}: No valid faces found")

        print(f"✅ Loaded {len(self.known_face_names)} people from database")
        return True

    def recognize_faces(self, image_path, draw_results=True):
        """
        Recognize faces in an image

        Args:
            image_path (str): Path to the image file
            draw_results (bool): Whether to draw bounding boxes and labels

        Returns:
            dict: Recognition results
        """
        print(f"🔍 Recognizing faces in: {os.path.basename(image_path)}")

        try:
            # Load image
            unknown_image = face_recognition.load_image_file(image_path)

            # Find faces and their encodings
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

            print(f"📊 Found {len(face_encodings)} face(s) to recognize")

            recognized_faces = []

            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=self.tolerance
                )

                # Calculate face distances
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding
                )

                # Find best match
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
                        'confidence': 1 - distance,  # Convert distance to confidence
                        'distance': distance,
                        'recognized': True
                    })
                    print(f"   ✅ Face {i + 1}: Recognized as {name} (confidence: {1 - distance:.2f})")
                else:
                    face_info.update({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'distance': float('inf'),
                        'recognized': False
                    })
                    print(f"   ❌ Face {i + 1}: Unknown person")

                recognized_faces.append(face_info)

            # Draw results if requested
            if draw_results and recognized_faces:
                result_image = self._draw_recognition_results(unknown_image, recognized_faces)
                return recognized_faces, result_image
            else:
                return recognized_faces, None

        except Exception as e:
            print(f"❌ Error recognizing faces: {e}")
            return [], None

    def _draw_recognition_results(self, image, recognized_faces):
        """
        Draw recognition results on image

        Args:
            image: numpy array image
            recognized_faces: List of face recognition results

        Returns:
            PIL.Image: Image with recognition results
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for face_info in recognized_faces:
            top, right, bottom, left = face_info['location']

            # Choose color based on recognition
            if face_info['recognized']:
                color = "green"
                label = f"{face_info['name']} ({face_info['confidence']:.2f})"
            else:
                color = "red"
                label = "Unknown"

            # Draw rectangle
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

            # Draw label background
            text_bbox = draw.textbbox((left, bottom), label)
            text_width = text_bbox[2] - text_bbox[0]
            draw.rectangle([left, bottom, left + text_width + 10, bottom + 25], fill=color)

            # Draw label text
            draw.text((left + 5, bottom), label, fill="white")

        return pil_image

    def add_new_person(self, person_name, image_paths):
        """
        Add a new person to the face database

        Args:
            person_name (str): Name of the person
            image_paths (list): List of image paths for this person
        """
        print(f"👤 Adding new person: {person_name}")

        if person_name in self.known_face_names:
            print(f"⚠️  Person {person_name} already exists. Updating...")

        person_encodings = []

        for image_path in image_paths:
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    person_encodings.append(face_encodings[0])
                    print(f"   ✅ {os.path.basename(image_path)}: Face encoded")
                else:
                    print(f"   ❌ {os.path.basename(image_path)}: No face detected")

            except Exception as e:
                print(f"   ❌ {os.path.basename(image_path)}: Error - {e}")

        if person_encodings:
            # Update database
            primary_encoding = person_encodings[0]

            if person_name in self.known_face_names:
                # Update existing person
                index = self.known_face_names.index(person_name)
                self.known_face_encodings[index] = primary_encoding
            else:
                # Add new person
                self.known_face_encodings.append(primary_encoding)
                self.known_face_names.append(person_name)

            self.face_database[person_name] = person_encodings
            print(f"✅ {person_name}: Added {len(person_encodings)} face encoding(s)")
            return True
        else:
            print(f"❌ {person_name}: No valid faces found")
            return False

    def save_database(self, filepath):
        """
        Save face database to file

        Args:
            filepath (str): Path to save the database
        """
        try:
            database = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'full_database': self.face_database,
                'tolerance': self.tolerance
            }

            with open(filepath, 'wb') as f:
                pickle.dump(database, f)

            print(f"💾 Database saved to: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False

    def load_database(self, filepath):
        """
        Load face database from file

        Args:
            filepath (str): Path to load the database from
        """
        try:
            with open(filepath, 'rb') as f:
                database = pickle.load(f)

            self.known_face_encodings = database['encodings']
            self.known_face_names = database['names']
            self.face_database = database['full_database']
            self.tolerance = database.get('tolerance', 0.6)

            print(f"📂 Database loaded from: {filepath}")
            print(f"📊 Loaded {len(self.known_face_names)} people")
            return True

        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False

    def list_known_people(self):
        """List all known people in the database"""
        print("📋 KNOWN PEOPLE IN DATABASE:")
        print("=" * 30)

        for name, encodings in self.face_database.items():
            print(f"👤 {name}: {len(encodings)} face encoding(s)")

        if not self.face_database:
            print("📝 Database is empty. Add some people first!")