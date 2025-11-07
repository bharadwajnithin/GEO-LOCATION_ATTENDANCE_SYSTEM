"""
URL configuration for the adminview app.
"""
from django.urls import path
from . import views

app_name = 'adminview'

urlpatterns = [
    path('staff-home/', views.staff_home, name='staff_home'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    # User visibility pages
    path('users/staff/', views.staff_list, name='staff_list'),
    path('users/students/', views.student_list, name='student_list'),
    
    # Class management
    path('classes/', views.class_list, name='class_list'),
    path('classes/create/', views.class_create, name='class_create'),
    path('classes/<int:class_id>/edit/', views.class_edit, name='class_edit'),
    path('classes/<int:class_id>/delete/', views.class_delete, name='class_delete'),
    path('classes/<int:class_id>/enrollment/', views.class_enrollment, name='class_enrollment'),
    path('classes/<int:class_id>/remove-student/<int:student_id>/', views.remove_student_from_class, name='remove_student_from_class'),
    
    # Geo-fence management
    path('geofences/', views.geofence_list, name='geofence_list'),
    path('geofences/create/', views.geofence_create, name='geofence_create'),
    path('geofences/<int:geofence_id>/edit/', views.geofence_edit, name='geofence_edit'),
    path('geofences/<int:geofence_id>/delete/', views.geofence_delete, name='geofence_delete'),
    
    # Analytics and reporting
    path('analytics/', views.attendance_analytics, name='attendance_analytics'),
    
    # API endpoints
    path('api/save-geofence-coordinates/', views.save_geofence_coordinates, name='save_geofence_coordinates'),
    path('api/admin-dashboard-metrics/', views.admin_dashboard_metrics, name='admin_dashboard_metrics'),
    path('api/staff-home-metrics/', views.staff_home_metrics, name='staff_home_metrics'),
]
