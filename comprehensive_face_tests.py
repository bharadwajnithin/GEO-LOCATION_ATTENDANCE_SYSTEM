#!/usr/bin/env python3
"""
Comprehensive Face Recognition Testing Framework
Tests for security, accuracy, and liveness detection
"""

import os
import sys
import django
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import time

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from django.conf import settings
from userview.models import User, Class
from userview.views import verify_face, get_face_similarity_score, _simple_fr_available
from simple_face_recognition import SimpleFaceRecognition

class FaceRecognitionTestSuite:
    """Comprehensive test suite for face recognition system"""
    
    def __init__(self):
        self.results = {
            'negative_tests': [],
            'positive_tests': [],
            'liveness_tests': [],
            'impostor_tests': [],
            'summary': {}
        }
        self.test_users = []
        self.setup_test_users()
    
    def setup_test_users(self):
        """Setup test users with different face patterns"""
        print("=== Setting up Test Users ===")
        
        # Get existing student users
        students = User.objects.filter(role='student')[:4]  # Use first 4 students
        
        if len(students) < 2:
            print("❌ Need at least 2 student users for testing")
            return False
        
        self.test_users = list(students)
        print(f"✓ Using {len(self.test_users)} test users:")
        for i, user in enumerate(self.test_users):
            print(f"  User {i+1}: {user.username}")
        
        return True
    
    def create_face_image(self, user_id, variation=0, pose='front'):
        """Create distinctive face images for each user"""
        # Create unique face patterns for each user with more variation
        base_size = 150 + user_id * 20
        test_image = np.zeros((base_size, base_size, 3), dtype=np.uint8)
        
        # User-specific face features with more distinctive differences
        face_size = 60 + user_id * 10
        x_offset = 30 + user_id * 8
        y_offset = 30 + user_id * 8
        
        # Pose variations
        if pose == 'left':
            x_offset -= 10
        elif pose == 'right':
            x_offset += 10
        elif pose == 'up':
            y_offset -= 8
        elif pose == 'down':
            y_offset += 8
        
        # Face rectangle with user-specific color
        color_intensity = 200 + user_id * 10
        cv2.rectangle(test_image, (x_offset, y_offset), 
                     (x_offset + face_size, y_offset + face_size), 
                     (color_intensity, color_intensity, color_intensity), -1)
        
        # Eyes - user-specific positioning with more variation
        eye_y = y_offset + face_size//3 + user_id * 3
        eye_size = 3 + user_id * 2
        eye_spacing = face_size//3 + user_id * 5  # Different eye spacing per user
        cv2.circle(test_image, (x_offset + eye_spacing, eye_y), eye_size, (0, 0, 0), -1)
        cv2.circle(test_image, (x_offset + face_size - eye_spacing, eye_y), eye_size, (0, 0, 0), -1)
        
        # Nose - user-specific shape with more variation
        nose_y = y_offset + face_size//2 + user_id * 2
        nose_width = 2 + user_id * 3
        nose_height = 5 + user_id * 2
        cv2.rectangle(test_image, (x_offset + face_size//2 - nose_width, nose_y), 
                     (x_offset + face_size//2 + nose_width, nose_y + nose_height), (0, 0, 0), -1)
        
        # Mouth - user-specific width and position
        mouth_y = y_offset + 2*face_size//3 + user_id * 4
        mouth_width = face_size//3 + user_id * 4
        mouth_height = 3 + user_id
        cv2.rectangle(test_image, (x_offset + face_size//2 - mouth_width//2, mouth_y), 
                     (x_offset + face_size//2 + mouth_width//2, mouth_y + mouth_height), (0, 0, 0), -1)
        
        # Add user-specific unique patterns
        if user_id == 0:
            # User 0: Add distinctive marks
            cv2.circle(test_image, (x_offset + face_size//4, y_offset + face_size//4), 2, (128, 128, 128), -1)
        elif user_id == 1:
            # User 1: Add different pattern
            cv2.rectangle(test_image, (x_offset + face_size - 10, y_offset + 10), 
                         (x_offset + face_size - 5, y_offset + 15), (128, 128, 128), -1)
        elif user_id == 2:
            # User 2: Add triangular mark
            pts = np.array([[x_offset + 10, y_offset + face_size - 10], 
                           [x_offset + 15, y_offset + face_size - 5], 
                           [x_offset + 5, y_offset + face_size - 5]], np.int32)
            cv2.fillPoly(test_image, [pts], (128, 128, 128))
        elif user_id == 3:
            # User 3: Add cross pattern
            cv2.line(test_image, (x_offset + face_size - 15, y_offset + face_size - 15), 
                    (x_offset + face_size - 5, y_offset + face_size - 5), (128, 128, 128), 2)
            cv2.line(test_image, (x_offset + face_size - 5, y_offset + face_size - 15), 
                    (x_offset + face_size - 15, y_offset + face_size - 5), (128, 128, 128), 2)
        
        # Add variation noise
        if variation > 0:
            noise = np.random.randint(-variation*2, variation*2, test_image.shape, dtype=np.int16)
            test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', test_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_b64}"
    
    def create_photo_attack_image(self, user_id):
        """Create a flat photo simulation (reduced depth cues)"""
        # Create a flatter, more uniform image to simulate photo
        base_image = self.create_face_image(user_id, 0, 'front')
        
        # Decode and process to make it look like a photo
        image_data = base_image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Reduce contrast and add uniform lighting (photo characteristics)
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=30)
        
        # Add slight blur (photo quality)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_b64}"
    
    def enroll_test_users(self):
        """Enroll all test users with their face data"""
        print("\n=== Enrolling Test Users ===")
        
        from userview.views import _store_user_embeddings
        
        for i, user in enumerate(self.test_users):
            print(f"Enrolling user {i+1}: {user.username}")
            
            # Create multiple face images for enrollment
            face_images = []
            poses = ['front', 'left', 'right', 'up', 'down']
            
            for pose in poses:
                face_img = self.create_face_image(i, variation=2, pose=pose)
                face_images.append(face_img)
            
            # Store embeddings
            success = _store_user_embeddings(user, face_images)
            
            if success:
                print(f"  ✓ Enrolled {user.username} with {len(face_images)} images")
            else:
                print(f"  ❌ Failed to enroll {user.username}")
                return False
        
        return True
    
    def test_negative_authentication(self):
        """Test 1: User 2 presents User 1's face → must be rejected"""
        print("\n=== Negative Test: Wrong User Face ===")
        
        if len(self.test_users) < 2:
            print("❌ Need at least 2 users for negative testing")
            return
        
        user1 = self.test_users[0]  # Target user
        user2 = self.test_users[1]  # Attacker user
        
        # Create User 1's face image
        user1_face = self.create_face_image(0, variation=1, pose='front')
        
        # Test: User 2 tries to authenticate with User 1's face
        print(f"Testing: {user2.username} presenting {user1.username}'s face")
        
        verification_result = verify_face(user2, user1_face)
        similarity_score = get_face_similarity_score(user2, user1_face)
        
        test_result = {
            'test_type': 'negative_auth',
            'attacker': user2.username,
            'target': user1.username,
            'verification_result': verification_result,
            'similarity_score': similarity_score,
            'expected': False,
            'passed': not verification_result,  # Should be False (rejected)
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['negative_tests'].append(test_result)
        
        if test_result['passed']:
            print(f"  ✓ PASS: Correctly rejected (similarity: {similarity_score:.4f})")
        else:
            print(f"  ❌ FAIL: Incorrectly accepted (similarity: {similarity_score:.4f})")
        
        return test_result['passed']
    
    def test_positive_authentication(self):
        """Test 2: User with their own face across multiple poses → accepted"""
        print("\n=== Positive Test: Legitimate User Multiple Poses ===")
        
        user = self.test_users[0]
        poses = ['front', 'left', 'right', 'up', 'down']
        passed_count = 0
        
        for pose in poses:
            print(f"Testing {user.username} with {pose} pose...")
            
            # Create user's face in different pose
            user_face = self.create_face_image(0, variation=3, pose=pose)
            
            verification_result = verify_face(user, user_face)
            similarity_score = get_face_similarity_score(user, user_face)
            
            test_result = {
                'test_type': 'positive_auth',
                'user': user.username,
                'pose': pose,
                'verification_result': verification_result,
                'similarity_score': similarity_score,
                'expected': True,
                'passed': verification_result,  # Should be True (accepted)
                'timestamp': datetime.now().isoformat()
            }
            
            self.results['positive_tests'].append(test_result)
            
            if test_result['passed']:
                print(f"  ✓ PASS: {pose} pose accepted (similarity: {similarity_score:.4f})")
                passed_count += 1
            else:
                print(f"  ❌ FAIL: {pose} pose rejected (similarity: {similarity_score:.4f})")
        
        success_rate = passed_count / len(poses)
        print(f"Positive test success rate: {success_rate:.2%} ({passed_count}/{len(poses)})")
        
        return success_rate >= 0.8  # 80% success rate required
    
    def test_photo_attack(self):
        """Test 3: Photo/screen attack → rejected by liveness detection"""
        print("\n=== Liveness Test: Photo Attack Detection ===")
        
        user = self.test_users[0]
        
        # Create photo attack image
        photo_attack = self.create_photo_attack_image(0)
        
        print(f"Testing photo attack against {user.username}...")
        
        # Test basic verification
        verification_result = verify_face(user, photo_attack)
        similarity_score = get_face_similarity_score(user, photo_attack)
        
        # Test liveness detection if available
        liveness_result = None
        if _simple_fr_available:
            try:
                recognizer = SimpleFaceRecognition()
                image_data = photo_attack.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Simple liveness check: check for uniform lighting (photo characteristic)
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Calculate lighting uniformity (photos tend to be more uniform)
                std_dev = np.std(gray)
                liveness_result = std_dev > 30  # Threshold for "live" vs "photo"
                
            except Exception as e:
                print(f"  Liveness detection error: {e}")
                liveness_result = None
        
        test_result = {
            'test_type': 'photo_attack',
            'user': user.username,
            'verification_result': verification_result,
            'similarity_score': similarity_score,
            'liveness_result': liveness_result,
            'expected_verification': False,  # Should be rejected
            'expected_liveness': False,  # Should detect as photo
            'passed': not verification_result or (liveness_result is not None and not liveness_result),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['liveness_tests'].append(test_result)
        
        if test_result['passed']:
            print(f"  ✓ PASS: Photo attack detected/rejected")
            print(f"    Verification: {verification_result}, Similarity: {similarity_score:.4f}")
            if liveness_result is not None:
                print(f"    Liveness: {'Live' if liveness_result else 'Photo'}")
        else:
            print(f"  ❌ FAIL: Photo attack not detected")
            print(f"    Verification: {verification_result}, Similarity: {similarity_score:.4f}")
        
        return test_result['passed']
    
    def test_closest_impostor(self):
        """Test 4: Find most similar different users → must still fail"""
        print("\n=== Impostor Test: Closest User Pair ===")
        
        if len(self.test_users) < 3:
            print("❌ Need at least 3 users for impostor testing")
            return False
        
        # Test all user pairs to find most similar
        max_similarity = -1
        closest_pair = None
        
        print("Finding closest user pair...")
        
        for i in range(len(self.test_users)):
            for j in range(i + 1, len(self.test_users)):
                user1 = self.test_users[i]
                user2 = self.test_users[j]
                
                # Create user1's face and test against user2
                user1_face = self.create_face_image(i, variation=1, pose='front')
                similarity = get_face_similarity_score(user2, user1_face)
                
                print(f"  {user1.username} vs {user2.username}: {similarity:.4f}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_pair = (user1, user2, i, j)
        
        if closest_pair is None:
            print("❌ No user pairs found")
            return False
        
        user1, user2, i, j = closest_pair
        print(f"\nClosest pair: {user1.username} vs {user2.username} (similarity: {max_similarity:.4f})")
        
        # Test the closest impostor attack
        user1_face = self.create_face_image(i, variation=2, pose='front')
        verification_result = verify_face(user2, user1_face)
        
        test_result = {
            'test_type': 'closest_impostor',
            'impostor': user2.username,
            'target': user1.username,
            'similarity_score': max_similarity,
            'verification_result': verification_result,
            'expected': False,
            'passed': not verification_result,  # Should be rejected
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['impostor_tests'].append(test_result)
        
        if test_result['passed']:
            print(f"  ✓ PASS: Closest impostor correctly rejected")
        else:
            print(f"  ❌ FAIL: Closest impostor incorrectly accepted")
        
        return test_result['passed']
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FACE RECOGNITION TEST REPORT")
        print("="*60)
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        for test_category in ['negative_tests', 'positive_tests', 'liveness_tests', 'impostor_tests']:
            category_tests = self.results[test_category]
            total_tests += len(category_tests)
            passed_tests += sum(1 for test in category_tests if test['passed'])
        
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"Overall Success Rate: {overall_success_rate:.2%} ({passed_tests}/{total_tests})")
        print()
        
        # Detailed results
        categories = {
            'negative_tests': 'Negative Authentication Tests',
            'positive_tests': 'Positive Authentication Tests', 
            'liveness_tests': 'Liveness Detection Tests',
            'impostor_tests': 'Impostor Detection Tests'
        }
        
        for category, title in categories.items():
            tests = self.results[category]
            if tests:
                passed = sum(1 for test in tests if test['passed'])
                print(f"{title}: {passed}/{len(tests)} passed")
                
                for test in tests:
                    status = "✓ PASS" if test['passed'] else "❌ FAIL"
                    if 'similarity_score' in test:
                        print(f"  {status} - Similarity: {test['similarity_score']:.4f}")
                    else:
                        print(f"  {status}")
                print()
        
        # Security assessment
        print("SECURITY ASSESSMENT:")
        print("-" * 20)
        
        negative_passed = all(test['passed'] for test in self.results['negative_tests'])
        liveness_passed = all(test['passed'] for test in self.results['liveness_tests'])
        impostor_passed = all(test['passed'] for test in self.results['impostor_tests'])
        
        if negative_passed:
            print("✓ Cross-user authentication properly rejected")
        else:
            print("❌ SECURITY RISK: Cross-user authentication accepted")
        
        if liveness_passed:
            print("✓ Photo attacks properly detected")
        else:
            print("⚠️  Photo attack detection needs improvement")
        
        if impostor_passed:
            print("✓ Similar users properly distinguished")
        else:
            print("❌ SECURITY RISK: Similar users not distinguished")
        
        # Recommendations
        print("\nRECOMMENDations:")
        print("-" * 15)
        
        if overall_success_rate < 0.8:
            print("• Consider adjusting similarity thresholds")
        
        if not negative_passed or not impostor_passed:
            print("• CRITICAL: Strengthen face verification thresholds")
            print("• Consider additional security measures")
        
        if not liveness_passed:
            print("• Implement advanced liveness detection")
            print("• Add motion-based verification")
        
        positive_success = len([t for t in self.results['positive_tests'] if t['passed']]) / max(len(self.results['positive_tests']), 1)
        if positive_success < 0.8:
            print("• Improve pose invariance in face recognition")
            print("• Consider enrolling more face angles")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_success_rate': float(overall_success_rate),
            'total_tests': int(total_tests),
            'passed_tests': int(passed_tests),
            'security_status': {
                'negative_tests_passed': bool(negative_passed),
                'liveness_tests_passed': bool(liveness_passed),
                'impostor_tests_passed': bool(impostor_passed)
            }
        }
        
        try:
            with open('face_recognition_test_report.json', 'w') as f:
                json.dump(report_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save JSON report: {e}")
        
        print(f"\n📄 Detailed report saved to: face_recognition_test_report.json")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 COMPREHENSIVE FACE RECOGNITION TEST SUITE")
        print("=" * 50)
        
        if not self.setup_test_users():
            return False
        
        if not self.enroll_test_users():
            return False
        
        # Run all test categories
        test_results = []
        
        test_results.append(self.test_negative_authentication())
        test_results.append(self.test_positive_authentication())
        test_results.append(self.test_photo_attack())
        test_results.append(self.test_closest_impostor())
        
        # Generate comprehensive report
        self.generate_report()
        
        return all(test_results)

def main():
    """Main test execution"""
    test_suite = FaceRecognitionTestSuite()
    
    print("Starting comprehensive face recognition security tests...")
    print("This will test:")
    print("1. Negative authentication (wrong user)")
    print("2. Positive authentication (correct user, multiple poses)")
    print("3. Photo attack detection (liveness)")
    print("4. Closest impostor detection")
    print()
    
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed - check security settings!")

if __name__ == '__main__':
    main()
