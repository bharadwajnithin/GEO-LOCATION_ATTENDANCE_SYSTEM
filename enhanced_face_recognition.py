"""
Enhanced Face Recognition System for Advanced Attendance Tracking
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import mediapipe as mp
from deepface import DeepFace
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedFaceRecognition:
    """
    Advanced face recognition system with liveness detection and multiple model support
    """
    
    def __init__(self, model_name: str = "Facenet", confidence_threshold: float = 0.95):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
    def extract_face_embeddings(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract facial embeddings using advanced models (FaceNet, DeepFace, etc.)
        """
        try:
            # Convert bytes to image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Convert BGR to RGB for DeepFace
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract embeddings using DeepFace
            embedding = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            return np.array(embedding[0]['embedding'])
            
        except Exception as e:
            logger.error(f"Error extracting face embeddings: {e}")
            return None
    
    def detect_liveness(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Perform liveness detection to prevent spoofing attacks
        """
        try:
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use MediaPipe for face detection
            results = self.face_detection.process(rgb_image)
            
            liveness_score = 0.0
            face_detected = False
            
            if results.detections:
                face_detected = True
                # Basic liveness checks
                liveness_score = self._calculate_liveness_score(image, results.detections[0])
            
            return {
                'is_live': liveness_score > 0.7,
                'liveness_score': liveness_score,
                'face_detected': face_detected,
                'confidence': results.detections[0].score[0] if results.detections else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            return {
                'is_live': False,
                'liveness_score': 0.0,
                'face_detected': False,
                'confidence': 0.0
            }
    
    def _calculate_liveness_score(self, image: np.ndarray, detection) -> float:
        """
        Calculate liveness score based on various factors
        """
        try:
            # Extract face region
            h, w, _ = image.shape
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_roi = image[y:y+height, x:x+width]
            
            # Basic liveness indicators
            liveness_indicators = []
            
            # 1. Face size consistency
            if width > 50 and height > 50:
                liveness_indicators.append(0.3)
            
            # 2. Image quality (blur detection)
            blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            if blur_score > 100:
                liveness_indicators.append(0.3)
            
            # 3. Color distribution analysis
            color_variance = np.var(face_roi)
            if 1000 < color_variance < 10000:
                liveness_indicators.append(0.4)
            
            return sum(liveness_indicators)
            
        except Exception as e:
            logger.error(f"Error calculating liveness score: {e}")
            return 0.0
    
    def verify_face(self, stored_embedding: List[float], live_embedding: np.ndarray) -> Dict[str, any]:
        """
        Verify face by comparing stored and live embeddings
        """
        try:
            stored_array = np.array(stored_embedding).reshape(1, -1)
            live_array = live_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(stored_array, live_array)[0][0]
            
            # Convert to percentage
            similarity_percentage = similarity * 100
            
            is_match = similarity_percentage >= (self.confidence_threshold * 100)
            
            return {
                'is_match': is_match,
                'similarity_score': similarity_percentage,
                'confidence_threshold': self.confidence_threshold * 100,
                'verification_passed': is_match
            }
            
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return {
                'is_match': False,
                'similarity_score': 0.0,
                'confidence_threshold': self.confidence_threshold * 100,
                'verification_passed': False
            }

class LiveCameraProcessor:
    """
    Handle live camera feed processing for registration and attendance
    """
    
    def __init__(self):
        self.face_recognition = AdvancedFaceRecognition()
        
    def process_registration_feed(self, video_frames: List[bytes]) -> Dict[str, any]:
        """
        Process multiple frames from live camera during registration
        """
        best_embedding = None
        best_liveness_score = 0.0
        best_frame_data = None
        
        for frame_bytes in video_frames:
            # Extract embeddings
            embedding = self.face_recognition.extract_face_embeddings(frame_bytes)
            if embedding is None:
                continue
            
            # Check liveness
            liveness_data = self.face_recognition.detect_liveness(frame_bytes)
            
            if liveness_data['liveness_score'] > best_liveness_score:
                best_liveness_score = liveness_data['liveness_score']
                best_embedding = embedding
                best_frame_data = liveness_data
        
        return {
            'embedding': best_embedding.tolist() if best_embedding is not None else None,
            'liveness_data': best_frame_data,
            'registration_successful': best_embedding is not None and best_liveness_score > 0.7
        }
    
    def process_attendance_feed(self, frame_bytes: bytes, stored_embedding: List[float]) -> Dict[str, any]:
        """
        Process single frame for attendance verification
        """
        # Extract live embedding
        live_embedding = self.face_recognition.extract_face_embeddings(frame_bytes)
        if live_embedding is None:
            return {
                'verification_passed': False,
                'error': 'No face detected in frame'
            }
        
        # Check liveness
        liveness_data = self.face_recognition.detect_liveness(frame_bytes)
        if not liveness_data['is_live']:
            return {
                'verification_passed': False,
                'error': 'Liveness detection failed'
            }
        
        # Verify face
        verification_result = self.face_recognition.verify_face(stored_embedding, live_embedding)
        
        return {
            'verification_passed': verification_result['verification_passed'],
            'similarity_score': verification_result['similarity_score'],
            'liveness_score': liveness_data['liveness_score'],
            'confidence': liveness_data['confidence']
        }
