"""
Models for the userview app.
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator


class User(AbstractUser):
    """
    Custom User model extending Django's AbstractUser.
    """
    ROLE_CHOICES = [
        ('student', 'Student'),
        ('staff', 'Staff'),
        ('admin', 'Administrator'),
    ]
    
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='student')
    department_id = models.CharField(max_length=50, blank=True, null=True)
    face_data = models.JSONField(null=True, blank=True)  # Store facial encodings
    
    # Fix reverse accessor conflicts
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_name='custom_user_set',
        related_query_name='custom_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='custom_user_set',
        related_query_name='custom_user',
    )
    
    class Meta:
        db_table = 'userview_user'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
    
    @property
    def is_student(self):
        return self.role == 'student'
    
    @property
    def is_staff_member(self):
        return self.role in ['staff', 'admin']
    
    @property
    def is_admin(self):
        return self.role == 'admin'


class GeoFence(models.Model):
    """
    Geo-fence model for defining attendance boundaries.
    """
    name = models.CharField(max_length=100)
    coordinates = models.JSONField()  # Store as [{lat: number, lng: number}]
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'userview_geofence'
        verbose_name = 'Geo Fence'
        verbose_name_plural = 'Geo Fences'
    
    def __str__(self):
        return self.name
    
    def is_point_inside(self, lat, lng):
        """
        Check if a point (lat, lng) is inside the geo-fence polygon.
        Uses the point-in-polygon algorithm.
        """
        if not self.coordinates or len(self.coordinates) < 3:
            return False
        # Build polygon as list of (x, y) where x=lng, y=lat for correct ray casting
        polygon_xy = [(coord['lng'], coord['lat']) for coord in self.coordinates]
        return self._point_in_polygon(lat, lng, polygon_xy)
    
    def _point_in_polygon(self, lat, lng, polygon_xy):
        """
        Ray casting algorithm for point-in-polygon test using x=lng and y=lat.
        polygon_xy is a list of (x, y) tuples where x=lng, y=lat.
        """
        n = len(polygon_xy)
        inside = False
        x, y = float(lng), float(lat)
        p1x, p1y = polygon_xy[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_xy[i % n]
            # Check if the edge crosses the horizontal ray to the right of (x, y)
            if ((p1y > y) != (p2y > y)):
                # Compute x-intersection of the edge with the horizontal line at y
                denom = (p2y - p1y) if (p2y - p1y) != 0 else 1e-12
                xinters = (p2x - p1x) * (y - p1y) / denom + p1x
                if x < xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside


class Class(models.Model):
    """
    Class model representing courses/classes.
    """
    name = models.CharField(max_length=100)
    staff = models.ForeignKey(User, on_delete=models.CASCADE, related_name='classes_taught')
    geolocation = models.ForeignKey(GeoFence, on_delete=models.SET_NULL, null=True, blank=True, related_name='classes')
    students = models.ManyToManyField(User, related_name='classes_enrolled', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'userview_class'
        verbose_name = 'Class'
        verbose_name_plural = 'Classes'
    
    def __str__(self):
        return f"{self.name} - {self.staff.username}"
    
    @property
    def student_count(self):
        return self.students.count()


class Attendance(models.Model):
    """
    Attendance model for tracking student attendance.
    """
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='attendance_records')
    class_instance = models.ForeignKey(Class, on_delete=models.CASCADE, related_name='attendance_records')
    timestamp = models.DateTimeField(auto_now_add=True)
    is_present = models.BooleanField(default=False)
    geolocation = models.ForeignKey(GeoFence, on_delete=models.SET_NULL, null=True, blank=True, related_name='attendance_records')
    face_verified = models.BooleanField(default=False)
    location_verified = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'userview_attendance'
        verbose_name = 'Attendance'
        verbose_name_plural = 'Attendance Records'
        unique_together = ['student', 'class_instance', 'timestamp']
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.student.username} - {self.class_instance.name} - {self.timestamp.date()}"
