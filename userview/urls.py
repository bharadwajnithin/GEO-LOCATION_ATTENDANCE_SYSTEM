"""
URL configuration for the userview app.
"""
from django.urls import path
from . import views

app_name = 'userview'

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('logout/', views.logout_view, name='logout'),
    path('student-home/', views.student_home, name='student_home'),
    path('face-enroll/', views.face_enroll, name='face_enroll'),
    path('api/save-face-enroll/', views.save_face_enroll, name='save_face_enroll'),
    path('mark-attendance/', views.mark_attendance, name='mark_attendance'),
    path('identify-mark-attendance/', views.identify_mark_attendance, name='identify_mark_attendance'),
]
