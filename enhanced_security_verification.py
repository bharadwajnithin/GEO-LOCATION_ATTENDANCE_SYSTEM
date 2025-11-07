"""
Enhanced Security Verification System
Implements multiple security checks for face recognition
"""

import cv2
import numpy as np
import base64
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedSecurityVerification:
    """
    Multi-layered security verification system
    """
    
    def __init__(self):
        self.min_face_size = (50, 50)
        self.max_similarity_threshold = 0.99  # Very strict
        self.liveness_threshold = 25.0  # Lighting variation threshold
        
    def verify_face_quality(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Verify face image quality and detect potential attacks
        """
        try:
            # Decode image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'valid': False, 'reason': 'Invalid image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check 1: Image size and quality
            height, width = gray.shape
            if height < 100 or width < 100:
                return {'valid': False, 'reason': 'Image too small'}
            
            # Check 2: Lighting variation (liveness detection)
            lighting_std = np.std(gray)
            if lighting_std < self.liveness_threshold:
                return {'valid': False, 'reason': 'Insufficient lighting variation (possible photo)'}
            
            # Check 3: Edge detection (photos tend to have different edge characteristics)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            if edge_density < 0.02:  # Too few edges might indicate a flat photo
                return {'valid': False, 'reason': 'Insufficient edge detail (possible photo)'}
            
            # Check 4: Blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Too blurry
                return {'valid': False, 'reason': 'Image too blurry'}
            
            # Check 5: Color distribution (photos might have different color characteristics)
            if len(image.shape) == 3:
                color_std = np.std(image, axis=(0, 1))
                if np.mean(color_std) < 10:  # Too uniform color
                    return {'valid': False, 'reason': 'Insufficient color variation'}
            
            return {
                'valid': True,
                'lighting_std': float(lighting_std),
                'edge_density': float(edge_density),
                'blur_score': float(laplacian_var),
                'color_variation': float(np.mean(color_std)) if len(image.shape) == 3 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in face quality verification: {e}")
            return {'valid': False, 'reason': f'Processing error: {e}'}
    
    def enhanced_similarity_check(self, stored_embeddings: list, live_embedding: np.ndarray, 
                                user_id: str) -> Dict[str, any]:
        """
        Enhanced similarity checking with multiple criteria
        """
        try:
            if not stored_embeddings or live_embedding is None:
                return {'verified': False, 'reason': 'Missing embeddings'}
            
            # Normalize live embedding
            live_norm = live_embedding / (np.linalg.norm(live_embedding) + 1e-8)
            
            similarities = []
            
            for stored_emb in stored_embeddings:
                stored_arr = np.array(stored_emb)
                stored_norm = stored_arr / (np.linalg.norm(stored_arr) + 1e-8)
                
                # Calculate cosine similarity
                similarity = float(np.dot(stored_norm, live_norm))
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            min_similarity = min(similarities)
            
            # Enhanced verification criteria
            criteria_passed = 0
            total_criteria = 4
            
            # Criterion 1: Maximum similarity must be very high
            if max_similarity >= self.max_similarity_threshold:
                criteria_passed += 1
            
            # Criterion 2: Average similarity should be high
            if avg_similarity >= (self.max_similarity_threshold - 0.02):
                criteria_passed += 1
            
            # Criterion 3: Minimum similarity shouldn't be too low (consistency check)
            if min_similarity >= (self.max_similarity_threshold - 0.05):
                criteria_passed += 1
            
            # Criterion 4: Low variance in similarities (consistency)
            similarity_variance = np.var(similarities)
            if similarity_variance < 0.001:  # Low variance indicates consistency
                criteria_passed += 1
            
            # Require at least 3 out of 4 criteria to pass
            verified = criteria_passed >= 3
            
            return {
                'verified': verified,
                'max_similarity': max_similarity,
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'similarity_variance': float(similarity_variance),
                'criteria_passed': criteria_passed,
                'total_criteria': total_criteria,
                'threshold_used': self.max_similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced similarity check: {e}")
            return {'verified': False, 'reason': f'Similarity check error: {e}'}
    
    def comprehensive_verification(self, user, face_image_data: str) -> Dict[str, any]:
        """
        Perform comprehensive multi-layered verification
        """
        try:
            # Extract image bytes
            if face_image_data.startswith('data:image'):
                face_image_data = face_image_data.split(',')[1]
            image_bytes = base64.b64decode(face_image_data)
            
            # Step 1: Face quality and liveness check
            quality_result = self.verify_face_quality(image_bytes)
            if not quality_result['valid']:
                return {
                    'verified': False,
                    'reason': f"Quality check failed: {quality_result['reason']}",
                    'quality_details': quality_result
                }
            
            # Step 2: Extract face embedding
            from simple_face_recognition import SimpleFaceRecognition
            recognizer = SimpleFaceRecognition(confidence_threshold=0.6)
            live_embedding = recognizer.extract_face_embeddings(image_bytes)
            
            if live_embedding is None:
                return {
                    'verified': False,
                    'reason': 'Could not extract face embedding',
                    'quality_details': quality_result
                }
            
            # Step 3: Get stored embeddings
            if not user.face_data or not isinstance(user.face_data, dict):
                return {
                    'verified': False,
                    'reason': 'No stored face data for user',
                    'quality_details': quality_result
                }
            
            stored_embeddings = user.face_data.get('embeddings', [])
            if not stored_embeddings:
                return {
                    'verified': False,
                    'reason': 'No stored embeddings for user',
                    'quality_details': quality_result
                }
            
            # Step 4: Enhanced similarity verification
            similarity_result = self.enhanced_similarity_check(
                stored_embeddings, live_embedding, user.username
            )
            
            # Final verification decision
            final_verified = quality_result['valid'] and similarity_result['verified']
            
            return {
                'verified': final_verified,
                'quality_details': quality_result,
                'similarity_details': similarity_result,
                'user': user.username,
                'timestamp': str(np.datetime64('now'))
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive verification: {e}")
            return {
                'verified': False,
                'reason': f'Verification error: {e}'
            }

# Global instance for use in views
enhanced_security = EnhancedSecurityVerification()
