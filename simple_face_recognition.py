"""
Simple Face Recognition System - Fallback when advanced models fail
Uses OpenCV and basic face detection/recognition
"""

import cv2
import numpy as np
import base64
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class SimpleFaceRecognition:
    """
    Simple face recognition using OpenCV Haar cascades and basic feature extraction
    """
    
    def __init__(self, confidence_threshold: float = 0.6, require_face: bool = True):
        self.confidence_threshold = confidence_threshold
        self.require_face = require_face
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def extract_face_embeddings(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract simple face features using OpenCV
        """
        try:
            # Convert bytes to image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more relaxed parameters
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            
            if len(faces) == 0:
                if self.require_face:
                    return None
                # Legacy fallback (not recommended): use center region as "face"
                h, w = gray.shape
                center_x, center_y = w // 2, h // 2
                face_size = min(w, h) // 2
                x1 = max(0, center_x - face_size // 2)
                y1 = max(0, center_y - face_size // 2)
                x2 = min(w, center_x + face_size // 2)
                y2 = min(h, center_y + face_size // 2)
                face_roi = gray[y1:y2, x1:x2]
            else:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Enhance contrast and reduce noise
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_roi = clahe.apply(face_roi)
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
            
            # Extract robust features: spatial LBP histograms + L2 normalize
            features = self._extract_lbp_spatial_hist(face_roi, grid_size=4)
            
            # Final L2 normalization
            features = features.astype(np.float32)
            norm = np.linalg.norm(features) + 1e-8
            features = features / norm
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting face embeddings: {e}")
            return None
    
    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern features
        """
        try:
            # Simple LBP implementation
            rows, cols = image.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image[i, j]
                    code = 0
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            # Create histogram of LBP codes
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-8)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting LBP features: {e}")
            # Fallback to simple pixel features
            return image.flatten().astype(np.float32) / 255.0

    def _compute_lbp_image(self, image: np.ndarray) -> np.ndarray:
        """
        Compute LBP code image (rows-2, cols-2)
        """
        rows, cols = image.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                lbp[i-1, j-1] = code
        return lbp

    def _extract_lbp_spatial_hist(self, image: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """
        Extract spatially pooled LBP histograms over a grid (grid_size x grid_size)
        Returns concatenated histogram vector
        """
        try:
            lbp = self._compute_lbp_image(image)
            h, w = lbp.shape  # expected 62x62 (since 64x64 -> minus borders)
            cell_h = h // grid_size
            cell_w = w // grid_size
            feats: List[np.ndarray] = []
            for gy in range(grid_size):
                for gx in range(grid_size):
                    y1 = gy * cell_h
                    x1 = gx * cell_w
                    # Ensure last cell includes remainder pixels
                    y2 = h if gy == grid_size - 1 else (gy + 1) * cell_h
                    x2 = w if gx == grid_size - 1 else (gx + 1) * cell_w
                    cell = lbp[y1:y2, x1:x2]
                    hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                    hist = hist.astype(np.float32)
                    # Normalize per cell to reduce illumination variance
                    hist = hist / (np.sum(hist) + 1e-8)
                    feats.append(hist)
            return np.concatenate(feats, axis=0)
        except Exception as e:
            logger.error(f"Error extracting spatial LBP features: {e}")
            # Fallback to global LBP histogram
            return self._extract_lbp_features(image)
    
    def verify_face(self, stored_embedding: np.ndarray, live_embedding: np.ndarray) -> float:
        """
        Verify face by comparing embeddings using correlation
        """
        try:
            # Ensure both embeddings are normalized
            stored_norm = stored_embedding / (np.linalg.norm(stored_embedding) + 1e-8)
            live_norm = live_embedding / (np.linalg.norm(live_embedding) + 1e-8)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(stored_norm, live_norm)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert to similarity score (0-1)
            similarity = (correlation + 1) / 2
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return 0.0

def get_face_recognizer():
    """
    Get the best available face recognizer
    """
    try:
        # Try to import advanced face recognition
        from enhanced_face_recognition import AdvancedFaceRecognition
        return AdvancedFaceRecognition(model_name='Facenet', confidence_threshold=0.95)
    except Exception:
        # Fallback to simple recognition
        logger.info("Using simple face recognition fallback")
        return SimpleFaceRecognition(confidence_threshold=0.6)
