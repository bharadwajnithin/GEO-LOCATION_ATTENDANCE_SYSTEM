# Geo-Location + Face Recognition Attendance System (Django)

A Django-based attendance system that requires both physical presence within a geo-fence and identity verification via face recognition to mark attendance.

- Robust geo-fencing with polygon containment and buffer tolerance
- Face verification with OpenCV fallback and optional advanced models
- Roles: Admin/Staff for class and fence management, Students for attendance
- MongoDB database via Djongo

## Tech Stack
- Backend: Python 3.8+, Django 3.1
- Database: MongoDB (djongo + pymongo)
- APIs/Libs: Google Maps JS API, OpenCV, NumPy, Pillow
- Optional (advanced): MediaPipe, DeepFace, TensorFlow, PyTorch, dlib

## Repository Layout
- `manage.py`
- `requirements.txt` and `enhanced_requirements.txt`
- `enhanced_face_recognition.py` and `enhanced_geo_verification.py`
- `enhanced_views.py`
- `PROJECT_ARCHITECTURE.md` and `PROJECT_DOCUMENTATION.txt`
- `templates/`, `static/`, `media/`
- `.env`, `config.env`, `config.env.example`

## Features
- Dual verification: location + face, both must pass
- Configurable GPS accuracy and geo-fence buffer
- Student workflow: select class, preview fence on map, capture face, submit
- Staff/Admin: create/edit classes, assign geo-fences

## Prerequisites
- Python 3.8+ (recommended 3.8-3.10)
- MongoDB Community Server running locally (default: mongodb://localhost:27017)
- Google Cloud project with Maps JavaScript API enabled
- Windows/macOS/Linux

## Quick Start

1) Clone and create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2) Install dependencies

- Minimal runtime:
```bash
pip install -r requirements.txt
```

- Advanced face/geo stack (optional, heavy deps):
```bash
pip install -r enhanced_requirements.txt
```

3) Configure environment variables

Copy the example and fill values:
```bash
cp config.env.example .env  # use manual copy on Windows
```
Set the following in `.env` (or keep in `config.env` if your settings load it):
- `SECRET_KEY`
- `DEBUG` (True for dev)
- `DB_NAME`, `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`
- `GOOGLE_MAPS_API_KEY`
- `FACE_RECOGNITION_TOLERANCE` (e.g., 0.6)
- `GEO_FENCING_BUFFER` (meters)

4) Start MongoDB
- Ensure MongoDB is running locally on the configured port (default 27017).

5) Run the development server
```bash
python manage.py runserver
```
Visit http://127.0.0.1:8000

## Usage Overview

- Student flow
  - Go to student home (e.g., `/student-home/`).
  - Select a class to preview its polygon on the map.
  - Allow location access; readiness depends on polygon/buffer and GPS accuracy.
  - Start camera; capture face and submit attendance.

- Staff/Admin flow
  - Create/edit classes and assign geo-fences.
  - Review attendance analytics pages if enabled.

Details are in `PROJECT_DOCUMENTATION.txt` and `PROJECT_ARCHITECTURE.md`.

## Configuration Notes

- Location thresholds
  - `LOCATION_MAX_ACCURACY_METERS` (default ~50m): reject poor GPS fixes.
  - `GEOFENCE_ALLOWED_BUFFER_METERS` (default ~15m): near-boundary tolerance.

- Face verification
  - Default fallback uses OpenCV + cosine similarity against stored embeddings.
  - Optional advanced pipeline provided in `enhanced_face_recognition.py` (DeepFace/Facenet, MediaPipe liveness heuristics). Requires `enhanced_requirements.txt`.

- Maps integration
  - Ensure `GOOGLE_MAPS_API_KEY` is set and Maps JS API is enabled.
  - Allow localhost referrers in Google Cloud Console during development.

## Development Tips
- Use `requirements.txt` for a lightweight setup; switch to `enhanced_requirements.txt` only if you plan to run advanced recognition and analytics.
- When using Djongo, Django migrations exist but may behave differently; ensure compatible versions and test your model changes carefully.
- Large ML dependencies can be slow to install; consider using a GPU-enabled environment if you enable TensorFlow/PyTorch.

## Security
- Do not commit real secrets. Rotate any keys in the repo history and move secrets to a secure store.
- Keep `DEBUG=False` in production, set proper `ALLOWED_HOSTS`, and configure HTTPS.
- Enforce role-based access and CSRF protections (Django defaults help here).

## Troubleshooting
- Map not loading
  - Check `GOOGLE_MAPS_API_KEY`, refresh, and check Network tab for Maps script.
  - Ensure APIs are enabled and referrers are allowed.

- GPS accuracy too low
  - Move to an open area or raise `LOCATION_MAX_ACCURACY_METERS` for testing.

- Face not detected
  - Ensure good lighting and frontal face. Test with the fallback pipeline first.

## Scripts and Key Files
- `enhanced_geo_verification.py`: Polygon/circular containment, distance, confidence.
- `enhanced_face_recognition.py`: Embeddings, liveness heuristics, verification.
- `enhanced_views.py`: End-to-end registration and attendance flows using the enhanced modules.
- `requirements.txt`: Minimal deps to run core flows.
- `enhanced_requirements.txt`: Full stack for advanced recognition and analytics.

## License
Specify your license here (e.g., MIT). If not specified, the project is proprietary by default.

## Acknowledgements
- Django, Djongo, MongoDB
- OpenCV, NumPy, Pillow
- Google Maps JavaScript API
- Optional: MediaPipe, DeepFace/Facenet, TensorFlow/PyTorch, dlib
# -GAS-GEO-LOCATION-ATTENDANCE-TRACKING-SYSTEM
