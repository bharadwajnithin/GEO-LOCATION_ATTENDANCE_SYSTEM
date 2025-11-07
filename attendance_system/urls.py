"""
Main URL configuration for attendance_system project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('userview.urls')),
    path('adminview/', include('adminview.urls')),
    # PWA service worker at root scope
    path('service-worker.js', lambda request: HttpResponse(
        open(os.path.join(settings.BASE_DIR, 'static', 'service-worker.js'), 'rb').read(),
        content_type='application/javascript'
    )),
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
