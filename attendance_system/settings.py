"""
Django settings for attendance_system project.
"""

import os
from pathlib import Path
from decouple import AutoConfig, Config, RepositoryEnv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

"""
Environment loading order:
1) .env (preferred)
2) config.env (fallback)
"""
# Prefer .env via AutoConfig
config = AutoConfig(search_path=BASE_DIR)

# If essential keys are missing, fallback to explicit config.env repository
if not config('SECRET_KEY', default=None):
    try:
        config = Config(RepositoryEnv(BASE_DIR / 'config.env'))
    except Exception as e:
        print(f"Warning: Could not load config.env file: {e}")

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'userview',
    'adminview',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'attendance_system.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'attendance_system.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': config('DB_NAME', default='attendance_system'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default=27017, cast=int),
        'ENFORCE_SCHEMA': True,
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Custom User Model
AUTH_USER_MODEL = 'userview.User'

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

# Email settings (from .env / config.env). Defaults to console backend for dev.
EMAIL_BACKEND = config('EMAIL_BACKEND', default='django.core.mail.backends.console.EmailBackend')
EMAIL_HOST = config('EMAIL_HOST', default='localhost')
EMAIL_PORT = config('EMAIL_PORT', default=25, cast=int)
EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')
EMAIL_USE_TLS = config('EMAIL_USE_TLS', default=False, cast=bool)
EMAIL_USE_SSL = config('EMAIL_USE_SSL', default=False, cast=bool)
DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL', default='no-reply@example.com')

# CORS settings
CORS_ALLOW_ALL_ORIGINS = True  # Only for development
CORS_ALLOW_CREDENTIALS = True

# Login/Logout URLs
LOGIN_URL = '/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

# Face Recognition settings
FACE_RECOGNITION_TOLERANCE = config('FACE_RECOGNITION_TOLERANCE', default=0.6, cast=float)
FACE_RECOGNITION_MODEL = config('FACE_RECOGNITION_MODEL', default='hog')
ADV_FACE_COSINE_THRESHOLD = config('ADV_FACE_COSINE_THRESHOLD', default=0.5, cast=float)

# Strict one-to-one verification thresholds
FACE_MATCH_MIN_SIMILARITY = config('FACE_MATCH_MIN_SIMILARITY', default=0.6, cast=float)
LIVENESS_MIN_SCORE = config('LIVENESS_MIN_SCORE', default=0.4, cast=float)

# Geo-fencing settings
GEO_FENCING_BUFFER = config('GEO_FENCING_BUFFER', default=10, cast=int)
GEOFENCE_ALLOWED_BUFFER_METERS = config('GEOFENCE_ALLOWED_BUFFER_METERS', default=15, cast=int)
LOCATION_MAX_ACCURACY_METERS = config('LOCATION_MAX_ACCURACY_METERS', default=50, cast=int)

# Google Maps API - Try multiple sources
GOOGLE_MAPS_API_KEY = config('GOOGLE_MAPS_API_KEY', default='')

# Fallback: Try direct environment variable
if not GOOGLE_MAPS_API_KEY:
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')

# Fallback: Try reading from config.env file directly
if not GOOGLE_MAPS_API_KEY:
    try:
        config_file_path = BASE_DIR / 'config.env'
        if config_file_path.exists():
            with open(config_file_path, 'r') as f:
                for line in f:
                    if line.startswith('GOOGLE_MAPS_API_KEY='):
                        GOOGLE_MAPS_API_KEY = line.split('=', 1)[1].strip()
                        break
    except Exception as e:
        print(f"Error reading config.env directly: {e}")

# Debug: Print API key status (remove in production)
if DEBUG and not GOOGLE_MAPS_API_KEY:
    print("Warning: Google Maps API Key is empty - maps will not work!")

# Session settings
SESSION_COOKIE_AGE = 3600  # 1 hour
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
