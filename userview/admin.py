"""
Django admin configuration for the userview app.
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Class, GeoFence, Attendance


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """
    Custom admin interface for User model.
    """
    list_display = ('username', 'email', 'first_name', 'last_name', 'role', 'department_id', 'face_embeddings_count', 'is_active')
    list_filter = ('role', 'is_active', 'is_staff', 'is_superuser', 'date_joined')
    search_fields = ('username', 'first_name', 'last_name', 'email')
    ordering = ('username',)
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('role', 'department_id', 'face_data')}),
    )
    
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ('role', 'department_id')}),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()

    def face_embeddings_count(self, obj):
        try:
            if obj.face_data and isinstance(obj.face_data, dict):
                embs = obj.face_data.get('embeddings')
                return len(embs) if embs else 0
        except Exception:
            return 0
        return 0
    face_embeddings_count.short_description = 'Embeddings'

    actions = ['clear_face_data']

    def clear_face_data(self, request, queryset):
        updated = 0
        for user in queryset:
            user.face_data = None
            user.save(update_fields=['face_data'])
            updated += 1
        self.message_user(request, f"Cleared face data for {updated} user(s).")
    clear_face_data.short_description = 'Clear selected users\' face data'


@admin.register(GeoFence)
class GeoFenceAdmin(admin.ModelAdmin):
    """
    Admin interface for GeoFence model.
    """
    list_display = ('name', 'created_at', 'updated_at', 'coordinate_count')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('name',)
    ordering = ('name',)
    readonly_fields = ('created_at', 'updated_at')
    
    def coordinate_count(self, obj):
        """Display the number of coordinates in the geo-fence."""
        if obj.coordinates:
            return len(obj.coordinates)
        return 0
    coordinate_count.short_description = 'Coordinates'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name',)
        }),
        ('Geo-fence Coordinates', {
            'fields': ('coordinates',),
            'description': 'Enter coordinates as JSON array of {lat: number, lng: number} objects'
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Class)
class ClassAdmin(admin.ModelAdmin):
    """
    Admin interface for Class model.
    """
    list_display = ('name', 'staff', 'geolocation', 'student_count', 'created_at')
    list_filter = ('created_at', 'staff__role', 'geolocation')
    search_fields = ('name', 'staff__username', 'staff__first_name', 'staff__last_name')
    ordering = ('name',)
    readonly_fields = ('created_at', 'updated_at')
    filter_horizontal = ('students',)
    
    def student_count(self, obj):
        """Display the number of enrolled students."""
        return obj.students.count()
    student_count.short_description = 'Students'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'staff', 'geolocation')
        }),
        ('Students', {
            'fields': ('students',),
            'description': 'Select students to enroll in this class'
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('staff', 'geolocation')


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    """
    Admin interface for Attendance model.
    """
    list_display = ('student', 'class_instance', 'timestamp', 'is_present', 'face_verified', 'location_verified')
    list_filter = ('is_present', 'face_verified', 'location_verified', 'timestamp', 'class_instance', 'geolocation')
    search_fields = ('student__username', 'student__first_name', 'student__last_name', 'class_instance__name')
    ordering = ('-timestamp',)
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Attendance Information', {
            'fields': ('student', 'class_instance', 'timestamp', 'is_present')
        }),
        ('Verification Details', {
            'fields': ('face_verified', 'location_verified', 'geolocation'),
            'description': 'Verification status for face recognition and geo-location'
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('student', 'class_instance', 'geolocation')
    
    def has_add_permission(self, request):
        """Attendance records are created automatically by the system."""
        return False
    
    def has_change_permission(self, request, obj=None):
        """Allow editing of verification status."""
        return True
    
    def has_delete_permission(self, request, obj=None):
        """Allow deletion of attendance records."""
        return True


# Customize admin site
admin.site.site_header = "Attendance System Administration"
admin.site.site_title = "Attendance System Admin"
admin.site.index_title = "Welcome to Attendance System Administration"
