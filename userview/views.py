"""
Views for the userview app.
"""
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db import transaction
from .forms import UserRegistrationForm, UserLoginForm, FaceDataForm, ProfileUpdateForm
from .models import User, Class, Attendance, GeoFence
from django.conf import settings
from datetime import datetime, time, timedelta
import math
from typing import List

try:
    # Prefer advanced face recognition if available
    from enhanced_face_recognition import AdvancedFaceRecognition
    _advanced_fr_available = True
except Exception:
    _advanced_fr_available = False

# Import simple face recognition as fallback
try:
    from simple_face_recognition import SimpleFaceRecognition, get_face_recognizer
    _simple_fr_available = True
except Exception:
    _simple_fr_available = False


def register_view(request):
    """
    User registration view (simplified).
    """
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    # Enforce role restriction: only admin can create admin/staff
                    desired_role = form.cleaned_data.get('role')
                    is_admin_user = request.user.is_authenticated and getattr(request.user, 'role', None) == 'admin'
                    if not is_admin_user and desired_role in ['admin', 'staff']:
                        messages.error(request, 'Only administrators can create Admin or Staff accounts.')
                        # Re-render with restricted role choices
                        try:
                            form.fields['role'].choices = [(k, v) for k, v in form.fields['role'].choices if k == 'student']
                        except Exception:
                            pass
                        return render(request, 'userview/register.html', {'form': form})

                    # Persist user with enforced role when creator is not admin
                    user = form.save(commit=False)
                    if not is_admin_user:
                        user.role = 'student'
                    user.save()

                    # Optional: handle multiple face images for pre-training
                    face_images: List[str] = request.POST.getlist('face_images[]') or []
                    if face_images:
                        try:
                            embeddings_saved = _store_user_embeddings(user, face_images)
                            if embeddings_saved:
                                messages.success(request, 'Face data captured and trained successfully.')
                            else:
                                messages.warning(request, 'Face data could not be processed. You can add it later.')
                        except Exception as e:
                            messages.warning(request, f'Face data processing failed: {str(e)}')

                    messages.success(request, 'Registration successful! Please log in.')
                    return redirect('userview:login')
            except Exception as e:
                messages.error(request, f'Registration failed: {str(e)}')
    else:
        form = UserRegistrationForm()
        # Adjust available role choices based on current user
        is_admin_user = request.user.is_authenticated and getattr(request.user, 'role', None) == 'admin'
        if not is_admin_user:
            try:
                form.fields['role'].choices = [(k, v) for k, v in form.fields['role'].choices if k == 'student']
            except Exception:
                pass

    return render(request, 'userview/register.html', {
        'form': form,
    })


def _store_user_embeddings(user: User, face_images_b64: List[str]) -> bool:
    """
    Process multiple base64-encoded images, compute embeddings and store on user.face_data.
    Returns True if at least one embedding saved.
    """
    if not face_images_b64:
        return False

    embeddings: List[List[float]] = []

    # Force simple recognizer for consistent enrollment
    recognizer = None
    model_type = 'simple'
    if _simple_fr_available:
        try:
            recognizer = SimpleFaceRecognition(confidence_threshold=0.6)
            model_type = 'simple'
        except Exception:
            recognizer = None
    
    if recognizer is not None:
        for data_url in face_images_b64:
            try:
                if data_url.startswith('data:image'):
                    data_url = data_url.split(',')[1]
                img_bytes = base64.b64decode(data_url)
                embedding = recognizer.extract_face_embeddings(img_bytes)
                if embedding is not None and isinstance(embedding, (list, np.ndarray)):
                    # normalize and ensure list of floats
                    emb_arr = np.array(embedding)
                    norm = np.linalg.norm(emb_arr) + 1e-8
                    emb_arr = emb_arr / norm
                    emb_list = emb_arr.tolist()
                    embeddings.append(emb_list)
            except Exception:
                continue
    else:
        # Fallback to simple OpenCV-based encoding already present in this file
        for data_url in face_images_b64:
            try:
                if data_url.startswith('data:image'):
                    data_url = data_url.split(',')[1]
                img_bytes = base64.b64decode(data_url)
                encoding = generate_face_encoding_from_bytes(img_bytes)
                if encoding is not None:
                    # normalize and ensure list of floats
                    enc_arr = np.array(encoding)
                    norm = np.linalg.norm(enc_arr) + 1e-8
                    enc_arr = enc_arr / norm
                    emb_list = enc_arr.tolist()
                    embeddings.append(emb_list)
            except Exception:
                continue

    if not embeddings:
        return False

    user.face_data = {
        'user_id': user.username,
        'model': model_type,
        'embeddings': embeddings,
    }
    user.save(update_fields=['face_data'])
    return True


def login_view(request):
    """
    User login view.
    """
    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                if user.role == 'student':
                    return redirect('userview:student_home')
                elif user.role == 'staff':
                    return redirect('adminview:staff_home')
                elif user.role == 'admin':
                    return redirect('adminview:admin_dashboard')
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = UserLoginForm()
    
    return render(request, 'userview/login.html', {'form': form})


def logout_view(request):
    """
    User logout view.
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('userview:login')


@login_required
def profile_view(request):
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('userview:profile')
    else:
        form = ProfileUpdateForm(instance=request.user)

    return render(request, 'userview/profile.html', {'form': form})


@login_required
def student_home(request):
    """
    Student dashboard view.
    """
    if request.user.role != 'student':
        messages.error(request, 'Access denied. Students only.')
        return redirect('userview:login')
    
    # If first-time student without face data, redirect to enrollment
    try:
        fd = request.user.face_data or {}
        has_embeddings = isinstance(fd, dict) and fd.get('embeddings')
    except Exception:
        has_embeddings = False
    if not has_embeddings and request.path != reverse('userview:face_enroll'):
        messages.info(request, 'Please enroll your face once to enable face-based attendance.')
        return redirect('userview:face_enroll')
    
    # Determine enrollment completeness
    fd_dict = request.user.face_data or {}
    emb_count = 0
    try:
        if isinstance(fd_dict, dict) and fd_dict.get('embeddings'):
            emb_count = len(fd_dict.get('embeddings'))
    except Exception:
        emb_count = 0
    min_required = int(getattr(settings, 'FACE_MIN_ENROLL_SAMPLES', 8))
    enrollment_pending = emb_count < min_required

    # Get classes where student is enrolled and active
    # Work around djongo SQLDecodeError on boolean WHERE in M2M join by filtering in Python
    try:
        _enrolled_qs = request.user.classes_enrolled.all()
    except Exception:
        _enrolled_qs = []
    enrolled_classes = [c for c in _enrolled_qs if getattr(c, 'is_active', True)]
    
    # Get today's attendance status for each class
    today_attendance = {}
    for cls in enrolled_classes:
        today_attendance[cls.id] = get_today_attendance_status(request.user, cls)
    
    # Prepare read-only geofence overlays for enrolled classes
    enrolled_geofences = []
    for cls in enrolled_classes:
        # Safely access geolocation; it may refer to a deleted GeoFence
        geo = None
        try:
            geo = cls.geolocation
        except GeoFence.DoesNotExist:
            geo = None
        except Exception:
            geo = None
        if geo and getattr(geo, 'coordinates', None):
            enrolled_geofences.append({
                'class_id': cls.id,
                'class_name': cls.name,
                'geofence_name': geo.name,
                'coordinates': geo.coordinates,
            })
    
    return render(request, 'userview/student_home.html', {
        'enrolled_classes': enrolled_classes,
        'today_attendance': today_attendance,
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', ''),
        'enrolled_geofences_json': json.dumps(enrolled_geofences),
        'enrollment_pending': enrollment_pending,
        'enrollment_count': emb_count,
        'enrollment_min_required': min_required,
    })


@login_required
def face_enroll(request):
    """
    Student face enrollment page (webcam capture UI).
    """
    if request.user.role != 'student':
        messages.error(request, 'Access denied. Students only.')
        return redirect('userview:login')
    return render(request, 'userview/face_enroll.html', {})


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def save_face_enroll(request):
    """
    API to receive multiple webcam frames and store user embeddings.
    Expects JSON: { images: [dataUrl, ...] }
    """
    if request.user.role != 'student':
        return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)
    try:
        payload = json.loads(request.body)
        images = payload.get('images') or []
        if not images:
            return JsonResponse({'success': False, 'error': 'No images provided'}, status=400)
        saved = _store_user_embeddings(request.user, images)
        if not saved:
            return JsonResponse({'success': False, 'error': 'Could not extract embeddings'}, status=400)
        return JsonResponse({'success': True})
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def mark_attendance(request):
    """
    Enhanced API endpoint for marking student attendance with dual verification.
    """
    if request.user.role != 'student':
        return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)
    
    try:
        data = json.loads(request.body)
        class_id = data.get('class_id')
        face_image_data = data.get('face_image')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        accuracy = data.get('accuracy')
        accuracy = data.get('accuracy')
        
        if not all([class_id, face_image_data, latitude, longitude]):
            return JsonResponse({
                'success': False, 
                'error': 'Missing required data'
            }, status=400)
        
        # Get the class
        try:
            class_instance = Class.objects.get(id=class_id)
        except Class.DoesNotExist:
            return JsonResponse({
                'success': False, 
                'error': 'Class not found'
            }, status=404)
        if not getattr(class_instance, 'is_active', True):
            return JsonResponse({'success': False, 'error': 'This class is currently disabled'}, status=400)
        
        # Check if student is enrolled in this class
        if class_instance not in request.user.classes_enrolled.all():
            return JsonResponse({
                'success': False, 
                'error': 'Not enrolled in this class'
            }, status=403)
        
        # Step 1: Verify geo-location first (prerequisite for face verification)
        # Ensure class has a configured geofence
        if not class_instance.geolocation:
            return JsonResponse({
                'success': False,
                'error': 'This class has no geo-fence configured. Please contact staff.',
            }, status=400)
        # Enforce GPS accuracy if provided
        try:
            accuracy_val = float(accuracy) if accuracy is not None and accuracy != '' else None
        except Exception:
            accuracy_val = None
        max_acc = getattr(settings, 'LOCATION_MAX_ACCURACY_METERS', 50)
        if accuracy_val is not None and accuracy_val > max_acc:
            return JsonResponse({
                'success': False,
                'error': f'Location accuracy too low (>{max_acc}m). Move to an area with better GPS signal.',
                'location_verified': False,
                'face_verified': False
            }, status=400)
        location_verified = verify_location(class_instance.geolocation, latitude, longitude)

        # Additional check: polygon containment with buffer tolerance
        if class_instance.geolocation and class_instance.geolocation.coordinates:
            student_point = {'lat': float(latitude), 'lng': float(longitude)}
            is_inside_geofence = is_point_in_polygon(student_point, class_instance.geolocation.coordinates)
            if not is_inside_geofence:
                # Allow small buffer tolerance near boundary
                buffer_m = getattr(settings, 'GEOFENCE_ALLOWED_BUFFER_METERS', 15)
                distance_m = distance_to_polygon_meters(student_point, class_instance.geolocation.coordinates)
                location_verified = distance_m is not None and distance_m <= float(buffer_m)
        
        # Step 2: Only proceed with face verification if location is verified
        face_verified = False
        face_similarity_score = 0.0
        liveness_score = 0.0
        
        if location_verified:
            face_verified = verify_face(request.user, face_image_data)
            # Get detailed verification results
            face_similarity_score = get_face_similarity_score(request.user, face_image_data)
            liveness_score = get_liveness_score(face_image_data)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Location verification failed. You must be within the geo-fenced area to mark attendance.',
                'location_verified': False,
                'face_verified': False
            }, status=400)
        
        if face_verified and location_verified:
            # Check if attendance already exists for today - use date range instead of __date
            today = timezone.now().date()
            start_time = datetime.combine(today, time.min)
            end_time = start_time + timedelta(days=1)
            
            # Make timezone-aware
            tz = timezone.get_current_timezone()
            if timezone.is_naive(start_time):
                start_time = timezone.make_aware(start_time, tz)
                end_time = timezone.make_aware(end_time, tz)
            
            existing_attendance = Attendance.objects.filter(
                student=request.user,
                class_instance=class_instance,
                timestamp__gte=start_time,
                timestamp__lt=end_time
            ).first()
            
            if existing_attendance:
                existing_attendance.is_present = True
                existing_attendance.face_verified = True
                existing_attendance.location_verified = True
                existing_attendance.save()
            else:
                # Create new attendance record
                Attendance.objects.create(
                    student=request.user,
                    class_instance=class_instance,
                    is_present=True,
                    geolocation=class_instance.geolocation,
                    face_verified=True,
                    location_verified=True
                )
            
            return JsonResponse({
                'success': True,
                'message': 'Attendance marked successfully!',
                'verification_details': {
                    'face_similarity': round(face_similarity_score * 100, 2),
                    'liveness_score': round(liveness_score * 100, 2),
                    'location_accuracy': accuracy_val,
                    'verification_timestamp': timezone.now().isoformat()
                }
            })
        else:
            error_message = 'Verification failed'
            if not location_verified:
                error_message = 'Location verification failed. Please ensure you are within the geo-fenced area.'
            elif not face_verified:
                error_message = 'Face verification failed. Please ensure good lighting and clear face visibility.'
            
            return JsonResponse({
                'success': False,
                'error': error_message,
                'face_verified': face_verified,
                'location_verified': location_verified,
                'verification_details': {
                    'face_similarity': round(face_similarity_score * 100, 2),
                    'liveness_score': round(liveness_score * 100, 2),
                    'location_accuracy': accuracy_val
                }
            }, status=400)
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False, 
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False, 
            'error': str(e)
        }, status=500)


def generate_face_encoding(image_file):
    """
    Generate face encoding from uploaded image using OpenCV.
    """
    try:
        # Read image using OpenCV
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the first face detected
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for consistency
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Convert to feature vector (simplified encoding)
            encoding = face_roi.flatten().astype(np.float32)
            
            # Normalize the encoding
            encoding = (encoding - np.mean(encoding)) / np.std(encoding)
            
            return encoding.tolist()
        else:
            return None
    except Exception as e:
        print(f"Error generating face encoding: {e}")
        return None


def generate_face_encoding_from_bytes(image_bytes):
    """
    Generate face encoding from image bytes (e.g., from base64) using OpenCV.
    """
    try:
        # Convert bytes to numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the first face detected
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for consistency
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Convert to feature vector (simplified encoding)
            encoding = face_roi.flatten().astype(np.float32)
            
            # Normalize the encoding
            encoding = (encoding - np.mean(encoding)) / np.std(encoding)
            
            return encoding
        else:
            return None
            
    except Exception as e:
        print(f"Error generating face encoding from bytes: {e}")
        return None


def verify_face(user, face_image_data):
    """
    Verify user's face against stored face data using OpenCV.
    """
    try:
        fd = user.face_data if isinstance(user.face_data, dict) else {}
        embeddings = fd.get('embeddings') if fd else None
        if not embeddings:
            return False

        if isinstance(face_image_data, str) and face_image_data.startswith('data:image'):
            face_image_data = face_image_data.split(',')[1]
        image_bytes = base64.b64decode(face_image_data)

        live_embedding = None
        
        # Check what model was used for enrollment
        stored_model = fd.get('model', 'unknown')
        
        # Try to use the same model that was used for enrollment
        if stored_model == 'simple' and _simple_fr_available:
            try:
                recognizer = SimpleFaceRecognition(confidence_threshold=0.6)
                live_embedding = recognizer.extract_face_embeddings(image_bytes)
            except Exception:
                live_embedding = None
        elif stored_model == 'opencv_simplified':
            # Use the same OpenCV method that was used for enrollment
            encoding = generate_face_encoding_from_bytes(image_bytes)
            if encoding is not None:
                live_embedding = np.array(encoding)
        else:
            live_embedding = None
        
        # Fallback to OpenCV method
        if live_embedding is None:
            encoding = generate_face_encoding_from_bytes(image_bytes)
            if encoding is not None:
                live_embedding = np.array(encoding)
        
        if live_embedding is None:
            return False

        def _l2norm(v):
            v = np.array(v)
            return v / (np.linalg.norm(v) + 1e-8)

        live_embedding = _l2norm(live_embedding)

        best_score = -1.0
        for emb in embeddings:
            try:
                emb_arr = _l2norm(np.array(emb))
                score = float(np.dot(emb_arr, live_embedding))
                if score > best_score:
                    best_score = score
            except Exception:
                continue

        if best_score < 0:
            return False

        threshold = float(getattr(settings, 'FACE_RECOGNITION_TOLERANCE', 0.75))
        return best_score >= threshold
    except Exception:
        return False


def verify_location(geofence, latitude, longitude):
    """
    Verify if coordinates are within the geo-fence.
    """
    try:
        return geofence.is_point_inside(float(latitude), float(longitude))
    except Exception as e:
        print(f"Error in location verification: {e}")
        return False


def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    """
    if not polygon or len(polygon) < 3:
        return False
    # Treat longitude as X and latitude as Y
    x, y = float(point['lng']), float(point['lat'])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i]['lng']), float(polygon[i]['lat'])
        xj, yj = float(polygon[j]['lng']), float(polygon[j]['lat'])
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def haversine_meters(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def distance_point_to_segment_meters(p, a, b):
    """Distance from point p to line segment ab in meters (lat/lng dicts)."""
    # Convert to approximate meters using local scaling around point a
    # Project lat/lng to meters (simple equirectangular projection)
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(a['lat']))
    ax, ay = (a['lng'] * lon_scale, a['lat'] * lat_scale)
    bx, by = (b['lng'] * lon_scale, b['lat'] * lat_scale)
    px, py = (p['lng'] * lon_scale, p['lat'] * lat_scale)

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0:
        # a and b are the same point
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def distance_to_polygon_meters(point, polygon):
    """Minimum distance in meters from a point to polygon edges. Returns 0 if inside."""
    try:
        if is_point_in_polygon(point, polygon):
            return 0.0
        min_d = None
        n = len(polygon)
        if n < 2:
            return None
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            d = distance_point_to_segment_meters(point, a, b)
            if min_d is None or d < min_d:
                min_d = d
        return min_d
    except Exception:
        return None


def get_today_attendance_status(student, class_instance):
    """
    Check if attendance was already marked today for a specific class.
    """
    try:
        # Use date range instead of __date to avoid djongo issues
        today = timezone.now().date()
        start_time = datetime.combine(today, time.min)
        end_time = start_time + timedelta(days=1)
        
        # Make timezone-aware
        tz = timezone.get_current_timezone()
        if timezone.is_naive(start_time):
            start_time = timezone.make_aware(start_time, tz)
            end_time = timezone.make_aware(end_time, tz)
        
        attendance = Attendance.objects.get(
            student=student,
            class_instance=class_instance,
            timestamp__gte=start_time,
            timestamp__lt=end_time
        )
        return attendance.is_present
    except Attendance.DoesNotExist:
        return None
    except Exception as e:
        print(f"Error checking attendance status: {e}")
        return None


def get_face_similarity_score(user, face_image_data):
    """
    Get detailed face similarity score for verification feedback.
    """
    try:
        if not user.face_data:
            return 0.0

        # Decode base64 image data
        if isinstance(face_image_data, str) and face_image_data.startswith('data:image'):
            face_image_data = face_image_data.split(',')[1]
        image_bytes = base64.b64decode(face_image_data)

        # Compute live embedding using same model as enrollment
        fd = user.face_data if isinstance(user.face_data, dict) else {}
        stored_model = fd.get('model', 'unknown')
        
        live_embedding = None
        
        # Try to use the same model that was used for enrollment
        if stored_model == 'simple' and _simple_fr_available:
            try:
                recognizer = SimpleFaceRecognition(confidence_threshold=0.6)
                live_embedding = recognizer.extract_face_embeddings(image_bytes)
            except Exception:
                live_embedding = None
        elif stored_model == 'opencv_simplified':
            # Use the same OpenCV method that was used for enrollment
            encoding = generate_face_encoding_from_bytes(image_bytes)
            if encoding is not None:
                live_embedding = np.array(encoding)
        elif _advanced_fr_available:
            try:
                recognizer = AdvancedFaceRecognition(model_name='Facenet', confidence_threshold=0.95)
                live_embedding = recognizer.extract_face_embeddings(image_bytes)
            except Exception:
                live_embedding = None
        
        # Fallback to OpenCV method
        if live_embedding is None:
            encoding = generate_face_encoding_from_bytes(image_bytes)
            if encoding is not None:
                live_embedding = np.array(encoding)
        
        if live_embedding is None:
            return 0.0

        # Compare against stored embeddings
        fd = user.face_data if isinstance(user.face_data, dict) else {}
        embeddings = fd.get('embeddings')
        if not embeddings:
            return 0.0

        best_score = -1.0
        for emb in embeddings:
            try:
                emb_arr = np.array(emb)
                score = float(np.dot(emb_arr, live_embedding) / (np.linalg.norm(emb_arr) * np.linalg.norm(live_embedding) + 1e-8))
                if score > best_score:
                    best_score = score
            except Exception:
                continue
        return max(best_score, 0.0)
    except Exception:
        return 0.0


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def identify_mark_attendance(request):
    """
    Identify the student by face among students enrolled in the class and mark attendance.
    Requires: class_id, face_image (base64), latitude, longitude.
    Security: Ensures identified face matches the logged-in user.
    """
    try:
        data = json.loads(request.body)
        class_id = data.get('class_id')
        face_image_data = data.get('face_image')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if not all([class_id, face_image_data, latitude, longitude]):
            return JsonResponse({'success': False, 'error': 'Missing required data'}, status=400)

        try:
            class_instance = Class.objects.get(id=class_id)
        except Class.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Class not found'}, status=404)
        if not getattr(class_instance, 'is_active', True):
            return JsonResponse({'success': False, 'error': 'This class is currently disabled'}, status=400)

        # Verify location first (enforce accuracy threshold if provided)
        # Ensure class has a configured geofence
        if not class_instance.geolocation:
            return JsonResponse({'success': False, 'error': 'This class has no geo-fence configured. Please contact staff.'}, status=400)
        try:
            accuracy_val = float(accuracy) if accuracy is not None and accuracy != '' else None
        except Exception:
            accuracy_val = None
        max_acc = getattr(settings, 'LOCATION_MAX_ACCURACY_METERS', 50)
        if accuracy_val is not None and accuracy_val > max_acc:
            return JsonResponse({'success': False, 'error': f'Location accuracy too low (>{max_acc}m). Move to an area with better GPS signal.'}, status=400)
        location_verified = verify_location(class_instance.geolocation, latitude, longitude)

        # Additional polygon check with buffer tolerance
        if class_instance.geolocation and class_instance.geolocation.coordinates:
            student_point = {'lat': float(latitude), 'lng': float(longitude)}
            is_inside_geofence = is_point_in_polygon(student_point, class_instance.geolocation.coordinates)
            if not is_inside_geofence:
                buffer_m = getattr(settings, 'GEOFENCE_ALLOWED_BUFFER_METERS', 15)
                distance_m = distance_to_polygon_meters(student_point, class_instance.geolocation.coordinates)
                location_verified = distance_m is not None and distance_m <= float(buffer_m)
        if not location_verified:
            return JsonResponse({'success': False, 'error': 'Location verification failed'}, status=400)

        # Prepare live embedding
        # Remove data URL prefix if present
        if face_image_data.startswith('data:image'):
            face_image_data = face_image_data.split(',')[1]
        image_bytes = base64.b64decode(face_image_data)

        live_embedding = None
        
        # Use simple face recognition for consistency with enrollment
        if _simple_fr_available:
            try:
                recognizer = SimpleFaceRecognition(confidence_threshold=0.6)
                live_embedding = recognizer.extract_face_embeddings(image_bytes)
            except Exception:
                live_embedding = None
        
        # Fallback to OpenCV method
        if live_embedding is None:
            encoding = generate_face_encoding_from_bytes(image_bytes)
            if encoding is not None:
                live_embedding = np.array(encoding)

        if live_embedding is None:
            return JsonResponse({'success': False, 'error': 'No face detected'}, status=400)

        # Identify among students enrolled in this class
        candidates = class_instance.students.all()
        best_user = None
        best_score = -1.0

        for candidate in candidates:
            fd = candidate.face_data or {}
            embeddings = fd.get('embeddings') if isinstance(fd, dict) else None
            if not embeddings:
                continue
            # Compute max cosine similarity across candidate's embeddings
            try:
                for emb in embeddings:
                    emb_arr = np.array(emb)
                    score = float(np.dot(emb_arr, live_embedding) / (np.linalg.norm(emb_arr) * np.linalg.norm(live_embedding) + 1e-8))
                    if score > best_score:
                        best_score = score
                        best_user = candidate
            except Exception:
                continue

        # Thresholds
        threshold = float(getattr(settings, 'ADV_FACE_COSINE_THRESHOLD', 0.6)) if _advanced_fr_available else float(getattr(settings, 'FACE_RECOGNITION_TOLERANCE', 0.6))
        if best_user is None or best_score < threshold:
            return JsonResponse({'success': False, 'error': 'Face not recognized'}, status=400)

        # Ensure identified face matches logged-in user to prevent spoofing via other accounts
        if best_user.id != request.user.id:
            return JsonResponse({'success': False, 'error': 'Face does not match the logged-in user'}, status=403)

        # Mark attendance for identified user (same as logged-in user)
        today = timezone.now().date()
        start_time = datetime.combine(today, time.min)
        end_time = start_time + timedelta(days=1)
        tz = timezone.get_current_timezone()
        if timezone.is_naive(start_time):
            start_time = timezone.make_aware(start_time, tz)
            end_time = timezone.make_aware(end_time, tz)

        existing_attendance = Attendance.objects.filter(
            student=best_user,
            class_instance=class_instance,
            timestamp__gte=start_time,
            timestamp__lt=end_time
        ).first()

        if existing_attendance:
            existing_attendance.is_present = True
            existing_attendance.face_verified = True
            existing_attendance.location_verified = True
            existing_attendance.save()
        else:
            Attendance.objects.create(
                student=best_user,
                class_instance=class_instance,
                is_present=True,
                geolocation=class_instance.geolocation,
                face_verified=True,
                location_verified=True
            )

        return JsonResponse({
            'success': True,
            'message': 'Attendance marked successfully!',
            'verification_details': {
                'face_similarity': round(best_score * 100, 2),
                'location_accuracy': accuracy_val,
                'verification_timestamp': timezone.now().isoformat()
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
        # Convert stored face data back to numpy array
        stored_encoding = np.array(user.face_data)
        
        # Decode base64 image data and process
        if face_image_data.startswith('data:image'):
            face_image_data = face_image_data.split(',')[1]
        
        image_bytes = base64.b64decode(face_image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Generate current encoding
            current_encoding = face_roi.flatten().astype(np.float32)
            current_encoding = (current_encoding - np.mean(current_encoding)) / np.std(current_encoding)
            
            # Calculate similarity (cosine similarity)
            similarity = np.dot(stored_encoding, current_encoding) / (np.linalg.norm(stored_encoding) * np.linalg.norm(current_encoding))
            return max(0.0, similarity)  # Ensure non-negative
        
        return 0.0
        
    except Exception as e:
        print(f"Error calculating face similarity: {e}")
        return 0.0


def get_liveness_score(face_image_data):
    """
    Calculate liveness score for anti-spoofing.
    """
    try:
        # Decode base64 image data
        if face_image_data.startswith('data:image'):
            face_image_data = face_image_data.split(',')[1]
        
        image_bytes = base64.b64decode(face_image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic liveness indicators
        liveness_score = 0.0
        
        # 1. Image quality (blur detection)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score > 100:
            liveness_score += 0.3
        
        # 2. Face size consistency
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            if w > 50 and h > 50:
                liveness_score += 0.3
        
        # 3. Color distribution analysis
        color_variance = np.var(image)
        if 1000 < color_variance < 10000:
            liveness_score += 0.4
        
        return min(1.0, liveness_score)
        
    except Exception as e:
        print(f"Error calculating liveness score: {e}")
        return 0.0
