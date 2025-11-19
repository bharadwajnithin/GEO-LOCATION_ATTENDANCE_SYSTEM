"""
Views for the adminview app.
"""
import json
import logging
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Q, Count
from django.utils import timezone
from datetime import datetime, timedelta, time
from .forms import ClassForm, GeoFenceForm, StudentEnrollmentForm, AttendanceFilterForm
from userview.models import User, Class, Attendance, GeoFence
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


def _day_bounds(day_date):
    """Return timezone-aware (start, end) datetimes for a given date."""
    start = datetime.combine(day_date, time.min)
    end = start + timedelta(days=1)
    tz = timezone.get_current_timezone()
    if timezone.is_naive(start):
        start = timezone.make_aware(start, tz)
        end = timezone.make_aware(end, tz)
    return start, end


def is_staff_member(user):
    """
    Check if user is a staff member or admin.
    """
    return user.is_authenticated and user.role in ['staff', 'admin']


def is_admin(user):
    """
    Check if user is an admin.
    """
    return user.is_authenticated and user.role == 'admin'


@login_required
@user_passes_test(is_staff_member)
def staff_home(request):
    """
    Staff dashboard view.
    """
    # Get classes taught by the staff member
    if request.user.role == 'admin':
        classes = Class.objects.all()
    else:
        classes = Class.objects.filter(staff=request.user)
    
    # Get attendance statistics
    total_students = sum(cls.students.count() for cls in classes)
    today = timezone.now().date()
    start, end = _day_bounds(today)
    # Materialize to avoid djongo lookup edge cases
    today_records = list(Attendance.objects.filter(
        class_instance__in=classes,
        timestamp__gte=start,
        timestamp__lt=end
    ).select_related('class_instance'))
    today_attendance = sum(1 for r in today_records if r.is_present)
    
    # Prepare read-only geofence data for map (handle deleted geofences safely)
    geofences_payload = []
    for cls in classes:
        geo = None
        try:
            geo = cls.geolocation
        except GeoFence.DoesNotExist:
            geo = None
        except Exception:
            geo = None
        if geo and getattr(geo, 'coordinates', None):
            geofences_payload.append({
                'class_name': cls.name,
                'geofence_name': geo.name,
                'coordinates': geo.coordinates,
            })

    context = {
        'classes': classes,
        'total_students': total_students,
        'today_attendance': today_attendance,
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
        'class_geofences_json': json.dumps(geofences_payload),
    }
    
    return render(request, 'adminview/staff_home.html', context)

@login_required
@user_passes_test(is_staff_member)
def staff_home_metrics(request):
    """
    JSON metrics for staff dashboard auto-refresh.
    """
    # Classes for this user
    if request.user.role == 'admin':
        classes = Class.objects.all()
    else:
        classes = Class.objects.filter(staff=request.user)

    total_students = sum(cls.students.count() for cls in classes)
    today = timezone.now().date()
    start, end = _day_bounds(today)
    today_records = list(Attendance.objects.filter(
        class_instance__in=classes,
        timestamp__gte=start,
        timestamp__lt=end
    ))
    today_attendance = sum(1 for r in today_records if r.is_present)

    payload = {
        'classes_count': classes.count() if hasattr(classes, 'count') else len(list(classes)),
        'total_students': total_students,
        'today_attendance': today_attendance,
    }
    return JsonResponse(payload)


@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    """
    Admin dashboard view with comprehensive analytics.
    """
    # Get overall statistics
    total_users = User.objects.count()
    total_students = User.objects.filter(role='student').count()
    total_staff = User.objects.filter(role='staff').count()
    total_classes = Class.objects.count()
    total_geofences = GeoFence.objects.count()
    
    # Get attendance statistics for the last 30 days
    thirty_days_ago = timezone.now() - timedelta(days=30)
    # Materialize recent attendance and compute in Python to avoid djongo SQLDecodeError
    recent_attendance = list(
        Attendance.objects.filter(timestamp__gte=thirty_days_ago)
        .select_related('class_instance')
    )
    
    total_attendance_records = len(recent_attendance)
    present_records = sum(1 for r in recent_attendance if r.is_present)
    attendance_percentage = (present_records / total_attendance_records * 100) if total_attendance_records > 0 else 0
    
    # Get class-wise attendance
    class_attendance = []
    classes = list(Class.objects.all())
    for class_instance in classes:
        class_records = [r for r in recent_attendance if r.class_instance_id == class_instance.id]
        if class_records:
            present_count = sum(1 for r in class_records if r.is_present)
            total_count = len(class_records)
            percentage = (present_count / total_count * 100) if total_count > 0 else 0
            class_attendance.append({
                'class_name': class_instance.name,
                'present_count': present_count,
                'total_count': total_count,
                'percentage': round(percentage, 2)
            })
    
    # Get daily attendance trend (use date range instead of __date)
    daily_attendance = []
    for i in range(7):
        day = timezone.now().date() - timedelta(days=i)
        start, end = _day_bounds(day)
        day_records = [r for r in recent_attendance if start <= r.timestamp < end]
        present_count = sum(1 for r in day_records if r.is_present)
        total_count = len(day_records)
        daily_attendance.append({
            'date': day.strftime('%Y-%m-%d'),
            'present_count': present_count,
            'total_count': total_count
        })
    daily_attendance.reverse()  # Show oldest to newest
    
    context = {
        'total_users': total_users,
        'total_students': total_students,
        'total_staff': total_staff,
        'total_classes': total_classes,
        'total_geofences': total_geofences,
        'total_attendance_records': total_attendance_records,
        'present_records': present_records,
        'absent_records': (total_attendance_records - present_records),
        'attendance_percentage': round(attendance_percentage, 2),
        'class_attendance': class_attendance,
        'daily_attendance': daily_attendance,
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
    }
    
    return render(request, 'adminview/admin_dashboard.html', context)

@login_required
@user_passes_test(is_admin)
def admin_dashboard_metrics(request):
    """
    JSON metrics for admin dashboard auto-refresh.
    """
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_attendance = list(
        Attendance.objects.filter(timestamp__gte=thirty_days_ago)
        .select_related('class_instance')
    )
    total_attendance_records = len(recent_attendance)
    present_records = sum(1 for r in recent_attendance if r.is_present)
    absent_records = total_attendance_records - present_records
    attendance_percentage = round((present_records / total_attendance_records * 100) if total_attendance_records > 0 else 0, 2)

    # Recompute charts data
    classes = list(Class.objects.all())
    class_attendance = []
    for class_instance in classes:
        class_records = [r for r in recent_attendance if r.class_instance_id == class_instance.id]
        if class_records:
            present_count = sum(1 for r in class_records if r.is_present)
            total_count = len(class_records)
            percentage = (present_count / total_count * 100) if total_count > 0 else 0
            class_attendance.append({
                'class_name': class_instance.name,
                'present_count': present_count,
                'total_count': total_count,
                'percentage': round(percentage, 2)
            })

    daily_attendance = []
    for i in range(7):
        day = timezone.now().date() - timedelta(days=i)
        start, end = _day_bounds(day)
        day_records = [r for r in recent_attendance if start <= r.timestamp < end]
        present_count = sum(1 for r in day_records if r.is_present)
        total_count = len(day_records)
        daily_attendance.append({
            'date': day.strftime('%Y-%m-%d'),
            'present_count': present_count,
            'total_count': total_count
        })
    daily_attendance.reverse()

    payload = {
        'totals': {
            'total_users': User.objects.count(),
            'total_students': User.objects.filter(role='student').count(),
            'total_staff': User.objects.filter(role='staff').count(),
            'total_classes': Class.objects.count(),
            'total_geofences': GeoFence.objects.count(),
        },
        'metrics': {
            'total_attendance_records': total_attendance_records,
            'present_records': present_records,
            'absent_records': absent_records,
            'attendance_percentage': attendance_percentage,
        },
        'class_attendance': class_attendance,
        'daily_attendance': daily_attendance,
    }
    return JsonResponse(payload)


@login_required
@user_passes_test(is_admin)
def staff_list(request):
    """
    List all staff users (admin only).
    """
    q = request.GET.get('q', '').strip()
    qs = User.objects.filter(role='staff')
    if q:
        qs = qs.filter(
            Q(username__icontains=q) |
            Q(first_name__icontains=q) |
            Q(last_name__icontains=q) |
            Q(email__icontains=q) |
            Q(department_id__icontains=q)
        )
    staffs = qs.order_by('username')
    return render(request, 'adminview/staff_list.html', {'staffs': staffs, 'q': q})


@login_required
@user_passes_test(is_staff_member)
def student_list(request):
    """
    List all students (visible to staff and admin).
    """
    q = request.GET.get('q', '').strip()
    qs = User.objects.filter(role='student')
    if q:
        qs = qs.filter(
            Q(username__icontains=q) |
            Q(first_name__icontains=q) |
            Q(last_name__icontains=q) |
            Q(email__icontains=q) |
            Q(department_id__icontains=q)
        )
    students = qs.order_by('username')
    return render(request, 'adminview/student_list.html', {'students': students, 'q': q})


@login_required
@user_passes_test(is_staff_member)
def class_list(request):
    """
    View for listing and managing classes.
    """
    if request.user.role == 'admin':
        classes = Class.objects.all()
    else:
        classes = Class.objects.filter(staff=request.user)
    
    return render(request, 'adminview/class_list.html', {'classes': classes})


@login_required
@user_passes_test(is_staff_member)
def class_create(request):
    """
    View for creating new classes.
    """
    if request.method == 'POST':
        form = ClassForm(request.POST)
        # Hide staff field for non-admin and auto-assign
        if request.user.role != 'admin' and hasattr(form, 'fields') and 'staff' in form.fields:
            form.fields.pop('staff')
        if form.is_valid():
            class_instance = form.save(commit=False)
            if request.user.role != 'admin':
                class_instance.staff = request.user
            class_instance.save()
            messages.success(request, f'Class "{class_instance.name}" created successfully!')
            return redirect('adminview:class_list')
    else:
        form = ClassForm()
        # Hide staff field for non-admin on form display
        if request.user.role != 'admin' and hasattr(form, 'fields') and 'staff' in form.fields:
            form.fields.pop('staff')
    
    # Provide geofence data for map preview
    try:
        geofence_data = {}
        import json as _json
        for g in GeoFence.objects.all():
            coords = g.coordinates or []
            if isinstance(coords, str):
                try:
                    coords = _json.loads(coords)
                except Exception:
                    coords = []
            geofence_data[str(g.id)] = {'coordinates': coords}
    except Exception:
        geofence_data = {}
    return render(request, 'adminview/class_form.html', {
        'form': form,
        'title': 'Create New Class',
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
        'geofence_data_json': json.dumps(geofence_data),
    })


@login_required
@user_passes_test(is_staff_member)
def class_edit(request, class_id):
    """
    View for editing existing classes.
    """
    class_instance = get_object_or_404(Class, id=class_id)
    
    # Check if user has permission to edit this class
    if request.user.role != 'admin' and class_instance.staff != request.user:
        messages.error(request, 'You do not have permission to edit this class.')
        return redirect('adminview:class_list')
    
    if request.method == 'POST':
        form = ClassForm(request.POST, instance=class_instance)
        # Hide staff field for non-admin and prevent reassignment
        if request.user.role != 'admin' and hasattr(form, 'fields') and 'staff' in form.fields:
            form.fields.pop('staff')
        if form.is_valid():
            updated = form.save(commit=False)
            if request.user.role != 'admin':
                updated.staff = request.user
            updated.save()
            messages.success(request, f'Class "{class_instance.name}" updated successfully!')
            return redirect('adminview:class_list')
    else:
        form = ClassForm(instance=class_instance)
        # Hide staff field for non-admin on form display
        if request.user.role != 'admin' and hasattr(form, 'fields') and 'staff' in form.fields:
            form.fields.pop('staff')
    
    # Provide geofence data for map preview
    try:
        geofence_data = {str(g.id): {'coordinates': g.coordinates or []} for g in GeoFence.objects.all()}
    except Exception:
        geofence_data = {}
    return render(request, 'adminview/class_form.html', {
        'form': form,
        'title': f'Edit Class: {class_instance.name}',
        'class_instance': class_instance,
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
        'geofence_data_json': json.dumps(geofence_data),
    })


@login_required
@user_passes_test(is_staff_member)
def class_delete(request, class_id):
    """
    View for deleting classes.
    """
    class_instance = get_object_or_404(Class, id=class_id)
    
    # Check if user has permission to delete this class
    if request.user.role != 'admin' and class_instance.staff != request.user:
        messages.error(request, 'You do not have permission to delete this class.')
        return redirect('adminview:class_list')
    
    if request.method == 'POST':
        class_name = class_instance.name
        class_instance.delete()
        messages.success(request, f'Class "{class_name}" deleted successfully!')
        return redirect('adminview:class_list')
    
    return render(request, 'adminview/class_confirm_delete.html', {
        'class_instance': class_instance
    })


@login_required
@user_passes_test(is_staff_member)
@require_http_methods(["POST"])
def class_toggle_active(request, class_id):
    """
    Toggle a class active/inactive. Staff can toggle only their classes; admin can toggle any.
    """
    class_instance = get_object_or_404(Class, id=class_id)
    if request.user.role != 'admin' and class_instance.staff != request.user:
        messages.error(request, 'You do not have permission to modify this class.')
        return redirect('adminview:class_list')

    class_instance.is_active = not bool(class_instance.is_active)
    class_instance.save(update_fields=['is_active'])
    status = 'enabled' if class_instance.is_active else 'disabled'
    messages.success(request, f'Class "{class_instance.name}" {status} successfully!')
    return redirect('adminview:class_list')


@login_required
@user_passes_test(is_staff_member)
def class_enrollment(request, class_id):
    """
    View for managing student enrollment in classes.
    """
    try:
        class_instance = get_object_or_404(Class, id=class_id)
        
        # Check if user has permission to manage this class
        if request.user.role != 'admin' and class_instance.staff != request.user:
            messages.error(request, 'You do not have permission to manage this class.')
            return redirect('adminview:class_list')
        
        if request.method == 'POST':
            form = StudentEnrollmentForm(request.POST, class_instance=class_instance)
            # Limit choices to not-yet-enrolled students
            try:
                current_ids = list(class_instance.students.values_list('id', flat=True))
                form.fields['students'].queryset = User.objects.filter(role='student').exclude(id__in=current_ids)
            except Exception:
                pass
            if form.is_valid():
                selected_students = form.cleaned_data['students']
                # Simple enrollment without complex operations
                for student in selected_students:
                    try:
                        class_instance.students.add(student)
                    except Exception as e:
                        print(f"Warning: Could not enroll student {student.username}: {e}")
                messages.success(request, f'{len(selected_students)} students enrolled successfully!')
                return redirect('adminview:class_enrollment', class_id=class_id)
        else:
            try:
                form = StudentEnrollmentForm(class_instance=class_instance)
                # Limit choices to not-yet-enrolled students
                try:
                    current_ids = list(class_instance.students.values_list('id', flat=True))
                    form.fields['students'].queryset = User.objects.filter(role='student').exclude(id__in=current_ids)
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning: Could not create enrollment form: {e}")
                # Create a simple form without complex queryset operations
                from django import forms
                form = forms.Form()
                form.fields = {}
                messages.warning(request, 'Enrollment form could not be loaded due to a database issue.')
        
        # Get currently enrolled students - use safer approach for djongo
        try:
            # Use a simple query to avoid complex operations
            enrolled_students = []
            for student in class_instance.students.all():
                enrolled_students.append(student)
        except Exception as e:
            print(f"Warning: Could not fetch enrolled students: {e}")
            enrolled_students = []
        
        # Show only not-enrolled students in the form choices
        try:
            enrolled_ids = [s.id for s in enrolled_students]
            available_students = User.objects.filter(role='student').exclude(id__in=enrolled_ids)
            if hasattr(form, 'fields') and 'students' in getattr(form, 'fields', {}):
                form.fields['students'].queryset = available_students
        except Exception as e:
            print(f"Warning: Could not filter available students: {e}")
        
        return render(request, 'adminview/class_enrollment.html', {
            'class_instance': class_instance,
            'form': form,
            'enrolled_students': enrolled_students
        })
        
    except Exception as e:
        print(f"Error in class_enrollment view: {e}")
        messages.error(request, 'An error occurred while loading the enrollment page.')
        return redirect('adminview:class_list')


@login_required
@user_passes_test(is_staff_member)
def remove_student_from_class(request, class_id, student_id):
    """
    View for removing students from class enrollment.
    """
    try:
        class_instance = get_object_or_404(Class, id=class_id)
        student = get_object_or_404(User, id=student_id, role='student')
        
        # Check if user has permission to manage this class
        if request.user.role != 'admin' and class_instance.staff != request.user:
            messages.error(request, 'You do not have permission to manage this class.')
            return redirect('adminview:class_list')
        
        # Check if student is actually enrolled in this class
        if student in class_instance.students.all():
            class_instance.students.remove(student)
            messages.success(request, f'Student "{student.get_full_name()}" removed from class "{class_instance.name}" successfully!')
        else:
            messages.warning(request, f'Student "{student.get_full_name()}" is not enrolled in this class.')
            
    except Exception as e:
        print(f"Error removing student from class: {e}")
        messages.error(request, 'An error occurred while removing the student.')
    
    return redirect('adminview:class_enrollment', class_id=class_id)


@login_required
@user_passes_test(is_staff_member)
def geofence_list(request):
    """
    View for listing and managing geo-fences.
    """
    geofences = GeoFence.objects.all()
    return render(request, 'adminview/geofence_list.html', {'geofences': geofences})


@login_required
@user_passes_test(is_staff_member)
def geofence_create(request):
    """
    View for creating new geo-fences.
    """
    if request.method == 'POST':
        form = GeoFenceForm(request.POST)
        if form.is_valid():
            geofence = form.save()
            messages.success(request, f'Geo-fence "{geofence.name}" created successfully!')
            return redirect('adminview:geofence_list')
    else:
        form = GeoFenceForm()
    
    return render(request, 'adminview/geofence_form.html', {
        'form': form,
        'title': 'Create New Geo-fence',
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
    })


@login_required
@user_passes_test(is_staff_member)
def geofence_edit(request, geofence_id):
    """
    View for editing existing geo-fences.
    """
    geofence = get_object_or_404(GeoFence, id=geofence_id)
    
    if request.method == 'POST':
        form = GeoFenceForm(request.POST, instance=geofence)
        if form.is_valid():
            form.save()
            messages.success(request, f'Geo-fence "{geofence.name}" updated successfully!')
            return redirect('adminview:geofence_list')
    else:
        form = GeoFenceForm(instance=geofence)
    
    return render(request, 'adminview/geofence_form.html', {
        'form': form,
        'title': f'Edit Geo-fence: {geofence.name}',
        'geofence': geofence,
        'google_maps_api_key': getattr(settings, 'GOOGLE_MAPS_API_KEY', None),
    })


@login_required
@user_passes_test(is_staff_member)
def geofence_delete(request, geofence_id):
    """
    View for deleting geo-fences.
    """
    geofence = get_object_or_404(GeoFence, id=geofence_id)
    
    if request.method == 'POST':
        geofence_name = geofence.name
        geofence.delete()
        messages.success(request, f'Geo-fence "{geofence_name}" deleted successfully!')
        return redirect('adminview:geofence_list')
    
    return render(request, 'adminview/geofence_confirm_delete.html', {
        'geofence': geofence
    })


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics(request):
    """
    View for attendance analytics and reporting.
    """
    # Determine classes visible to this user (needed for sanitization below)
    if request.user.role == 'admin':
        visible_classes = list(Class.objects.all())
    else:
        visible_classes = list(Class.objects.filter(staff=request.user))

    # Always treat GET the same; default filters when not provided
    # Sanitize incoming IDs to avoid DoesNotExist during form cleaning and widget rendering
    params = None
    if request.GET:
        try:
            params = request.GET.copy()
            sid = params.get('student')
            if sid and not User.objects.filter(id=sid, role='student').exists():
                params.pop('student', None)
            cid = params.get('class_instance')
            if cid:
                # Must exist and be within visible classes for this user
                visible_ids = {c.id for c in visible_classes}
                if (not Class.objects.filter(id=cid).exists()) or (int(str(cid)) not in visible_ids):
                    params.pop('class_instance', None)
        except Exception:
            params = request.GET
    form = AttendanceFilterForm(params or None)

    # Restrict form class choices to visible classes
    try:
        if hasattr(form, 'fields') and 'class_instance' in form.fields:
            form.fields['class_instance'].queryset = Class.objects.filter(id__in=[c.id for c in visible_classes])
    except Exception:
        pass

    # Parse filters with sensible defaults
    try:
        is_valid = form.is_valid()
    except Exception:
        is_valid = False
    if is_valid:
        selected_class = form.cleaned_data.get('class_instance')
        selected_student = form.cleaned_data.get('student')
        start_date = form.cleaned_data.get('start_date')
        end_date = form.cleaned_data.get('end_date')
    else:
        selected_class = None
        selected_student = None
        start_date = None
        end_date = None

    # Constrain student choices based on selected class or visible classes
    try:
        if hasattr(form, 'fields') and 'student' in form.fields:
            if selected_class:
                # Students enrolled in the selected class
                try:
                    student_ids = list(selected_class.students.filter(role='student').values_list('id', flat=True))
                except Exception:
                    student_ids = []
                form.fields['student'].queryset = User.objects.filter(role='student', id__in=student_ids)
            else:
                # All students across visible classes
                agg_ids = set()
                for cls in visible_classes:
                    try:
                        for sid in cls.students.filter(role='student').values_list('id', flat=True):
                            agg_ids.add(sid)
                    except Exception:
                        continue
                form.fields['student'].queryset = User.objects.filter(role='student', id__in=list(agg_ids))
            # If current selected_student is not in queryset, ignore it
            if selected_student and not form.fields['student'].queryset.filter(id=selected_student.id).exists():
                selected_student = None
    except Exception:
        pass

    # Default to last 7 days if no date range provided
    today = timezone.now().date()
    if not start_date and not end_date:
        start_date = today - timedelta(days=6)
        end_date = today
    elif start_date and not end_date:
        end_date = today
    elif end_date and not start_date:
        # limit to a single day if only end provided
        start_date = end_date

    # Build inclusive date list
    day_list = []
    if start_date and end_date and start_date <= end_date:
        cur = start_date
        while cur <= end_date:
            day_list.append(cur)
            cur += timedelta(days=1)

    # Determine target classes
    if selected_class:
        target_classes = [selected_class] if selected_class in visible_classes else []
    else:
        target_classes = visible_classes

    # Fetch all attendance in range for target classes (and optionally student)
    if day_list and target_classes:
        day_start, _ = _day_bounds(day_list[0])
        _, day_end = _day_bounds(day_list[-1])
        attendance_qs = Attendance.objects.filter(
            class_instance__in=target_classes,
            timestamp__gte=day_start,
            timestamp__lt=day_end,
        ).select_related('student', 'class_instance', 'geolocation')
        if selected_student:
            attendance_qs = attendance_qs.filter(student=selected_student)
        attendance_records = list(attendance_qs)
    else:
        attendance_records = []

    # De-duplicate by (class_id, student_id, day): mark present if any present entry exists that day
    from collections import defaultdict
    present_map = defaultdict(bool)  # key -> present?
    raw_records = []  # keep some raw records for table display
    # Build sets of existing FK ids to avoid template DoesNotExist on orphan rows
    try:
        existing_user_ids = set(User.objects.values_list('id', flat=True))
    except Exception:
        existing_user_ids = set()
    try:
        existing_class_ids = set(Class.objects.values_list('id', flat=True))
    except Exception:
        existing_class_ids = set()

    for r in attendance_records:
        # Skip orphaned FK rows
        if r.student_id not in existing_user_ids or r.class_instance_id not in existing_class_ids:
            continue
        try:
            day_key = timezone.localtime(r.timestamp).date()
        except Exception:
            day_key = r.timestamp.date()
        key = (r.class_instance_id, r.student_id, day_key)
        if r.is_present:
            present_map[key] = True or present_map[key]
        # keep a few recent records for display
        raw_records.append(r)

    # Compute class-wise summary including unmarked as absent
    class_breakdown = {}
    class_breakdown_rows = []
    total_expected_marks = 0
    total_present_marks = 0
    for cls in target_classes:
        # enrolled students for this class
        try:
            enrolled_students = list(cls.students.filter(role='student'))
        except Exception:
            enrolled_students = []
        days_count = len(day_list)
        expected = len(enrolled_students) * days_count
        present = 0
        if days_count > 0 and enrolled_students:
            for d in day_list:
                for s in enrolled_students:
                    if present_map.get((cls.id, s.id, d), False):
                        present += 1
        absent = max(0, expected - present)
        perc = round((present / expected * 100) if expected > 0 else 0, 2)
        class_breakdown[cls.name] = {
            'present': present,
            'absent': absent,
            'expected': expected,
            'percentage': perc,
        }
        class_breakdown_rows.append({
            'id': cls.id,
            'name': cls.name,
            'present': present,
            'absent': absent,
            'expected': expected,
            'percentage': perc,
        })
        total_expected_marks += expected
        total_present_marks += present

    # Overall metrics
    attendance_percentage = round((total_present_marks / total_expected_marks * 100) if total_expected_marks > 0 else 0, 2)

    # Per-student breakdown when a specific class is selected
    student_breakdown = []
    if selected_class:
        try:
            students_in_class = list(selected_class.students.filter(role='student'))
        except Exception:
            students_in_class = []
        for s in students_in_class:
            present_days = 0
            for d in day_list:
                if present_map.get((selected_class.id, s.id, d), False):
                    present_days += 1
            total_days = len(day_list)
            absent_days = max(0, total_days - present_days)
            perc = round((present_days / total_days * 100) if total_days > 0 else 0, 2)
            student_breakdown.append({
                'student': s,
                'present_days': present_days,
                'absent_days': absent_days,
                'total_days': total_days,
                'percentage': perc,
            })

    # Keep a limited recent raw record list for the table (sorted desc)
    raw_records.sort(key=lambda r: r.timestamp, reverse=True)
    records_for_table = raw_records[:200]

    # Compute enrolled/present/absent for cards and charts
    single_day = len(day_list) == 1
    enrolled_students_count = 0
    try:
        if selected_class:
            enrolled_students_count = selected_class.students.filter(role='student').count()
        else:
            # Sum across visible/target classes
            enrolled_students_count = sum(
                cls.students.filter(role='student').count() for cls in target_classes
            )
    except Exception:
        enrolled_students_count = 0

    if selected_class and single_day:
        # Per-student counts for the selected day
        day = day_list[0] if day_list else None
        present_count = 0
        if day is not None:
            try:
                students_in_class = list(selected_class.students.filter(role='student'))
            except Exception:
                students_in_class = []
            for s in students_in_class:
                if present_map.get((selected_class.id, s.id, day), False):
                    present_count += 1
        absent_count = max(0, enrolled_students_count - present_count)
        percentage_for_cards = round((present_count / enrolled_students_count * 100) if enrolled_students_count > 0 else 0, 2)
    else:
        # Aggregate marks across the selected range
        present_count = total_present_marks
        absent_count = max(0, total_expected_marks - total_present_marks)
        percentage_for_cards = attendance_percentage

    # Build trend data (daily present counts) when a specific class is selected
    trend_data = []
    if selected_class and day_list:
        try:
            students_in_class_ids = list(selected_class.students.filter(role='student').values_list('id', flat=True))
        except Exception:
            students_in_class_ids = []
        for d in day_list:
            present_count_for_day = 0
            for sid in students_in_class_ids:
                if present_map.get((selected_class.id, sid, d), False):
                    present_count_for_day += 1
            trend_data.append({
                'date': d.strftime('%Y-%m-%d'),
                'present': present_count_for_day,
            })

    context = {
        'form': form,
        'attendance_records': records_for_table,
        'total_records': total_expected_marks,
        'present_records': total_present_marks,
        'attendance_percentage': attendance_percentage,
        'class_breakdown': class_breakdown,
        'class_breakdown_rows': class_breakdown_rows,
        'student_breakdown': student_breakdown,
        'date_range': {'start': start_date, 'end': end_date},
        # New metrics for UI cards and charts
        'enrolled_students': enrolled_students_count,
        'present_count': present_count,
        'absent_count': absent_count,
        'cards_percentage': percentage_for_cards,
        'analytics_data': {
            'present': present_count,
            'absent': absent_count,
        },
        'trend_data': trend_data,
        'has_selected_class': bool(selected_class),
        'query_string': request.META.get('QUERY_STRING', ''),
    }
    return render(request, 'adminview/attendance_analytics.html', context)


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics_api(request):
    """
    Consolidated analytics API
    GET params:
      class_id (optional, int) - restrict to one class
      start_date, end_date (YYYY-MM-DD)
      student_id (optional, int)
      granularity (daily|weekly|monthly) - currently only affects trend grouping; default daily
      lookback (optional, int) - number of periods to include in trend when dates omitted
      export=csv (optional) - stream CSV for class-wise breakdown in current filter

    Returns JSON:
      totals: {enrolled, expected, present, absent, percentage}
      trend: [{label, date, present}]
      classes: [{id,name,present,absent,expected,percentage}]
    """
    corr_id = str(uuid.uuid4())
    try:
        # Authorization scope: visible classes
        if request.user.role == 'admin':
            visible_classes = list(Class.objects.all())
        else:
            visible_classes = list(Class.objects.filter(staff=request.user))

        # Parse inputs
        def _parse_int(v):
            try:
                return int(v)
            except Exception:
                return None
        def _parse_date(s):
            try:
                return datetime.strptime(s, '%Y-%m-%d').date()
            except Exception:
                return None

        class_id = _parse_int(request.GET.get('class_id'))
        student_id = _parse_int(request.GET.get('student_id'))
        start_date = _parse_date(request.GET.get('start_date', ''))
        end_date = _parse_date(request.GET.get('end_date', ''))
        granularity = (request.GET.get('granularity') or 'daily').lower()
        if granularity not in ('daily','weekly','monthly'):
            return JsonResponse({'error': 'Invalid granularity', 'correlation_id': corr_id}, status=400)
        lookback = _parse_int(request.GET.get('lookback'))

        # Defaults for date range
        today = timezone.now().date()
        if not start_date and not end_date:
            if lookback and lookback > 0:
                start_date = today - timedelta(days=lookback-1)
                end_date = today
            else:
                start_date = today - timedelta(days=6)
                end_date = today
        elif start_date and not end_date:
            end_date = today
        elif end_date and not start_date:
            start_date = end_date

        if start_date and end_date and start_date > end_date:
            return JsonResponse({'error': 'start_date must be on or before end_date', 'correlation_id': corr_id}, status=400)

        # Resolve selected_class and student
        selected_class = None
        if class_id:
            try:
                selected_class = next(c for c in visible_classes if c.id == class_id)
            except StopIteration:
                return JsonResponse({'error': 'Class not found or not permitted', 'correlation_id': corr_id}, status=404)
        selected_student = None
        if student_id:
            try:
                s = User.objects.get(id=student_id, role='student')
            except User.DoesNotExist:
                return JsonResponse({'error': 'Student not found', 'correlation_id': corr_id}, status=404)
            # If class filter provided, ensure student belongs to it
            if selected_class and not selected_class.students.filter(id=s.id).exists():
                return JsonResponse({'error': 'Student not in selected class', 'correlation_id': corr_id}, status=400)
            selected_student = s

        # Cache key
        cache_key = f"analytics:{request.user.id}:{class_id}:{student_id}:{start_date}:{end_date}:{granularity}"
        cached = cache.get(cache_key)
        if cached and request.GET.get('export') != 'csv':
            cached['correlation_id'] = corr_id
            return JsonResponse(cached)

        # Build day list
        day_list = []
        if start_date and end_date:
            d = start_date
            while d <= end_date:
                day_list.append(d)
                d += timedelta(days=1)

        # Target classes
        target_classes = [selected_class] if selected_class else visible_classes

        # Attendance window
        if day_list and target_classes:
            day_start, _ = _day_bounds(day_list[0])
            _, day_end = _day_bounds(day_list[-1])
            qs = Attendance.objects.filter(
                class_instance__in=target_classes,
                timestamp__gte=day_start,
                timestamp__lt=day_end,
            ).values('class_instance_id', 'student_id', 'timestamp', 'is_present')
            if selected_student:
                qs = qs.filter(student_id=selected_student.id)
            rows = list(qs)
        else:
            rows = []

        # Build present_map keyed by (class_id, student_id, day)
        present_map = {}
        for r in rows:
            try:
                dkey = timezone.localtime(r['timestamp']).date()
            except Exception:
                dkey = r['timestamp'].date()
            if r['is_present']:
                present_map[(r['class_instance_id'], r['student_id'], dkey)] = True

        # Aggregations
        classes_payload = []
        total_expected = 0
        total_present = 0
        for cls in target_classes:
            try:
                student_ids = list(cls.students.filter(role='student').values_list('id', flat=True))
            except Exception:
                student_ids = []
            days_count = len(day_list)
            expected = len(student_ids) * days_count
            present = 0
            if days_count and student_ids:
                for d in day_list:
                    for sid in student_ids:
                        if present_map.get((cls.id, sid, d), False):
                            present += 1
            absent = max(0, expected - present)
            perc = round((present / expected * 100) if expected else 0, 2)
            classes_payload.append({
                'id': cls.id,
                'name': cls.name,
                'present': present,
                'absent': absent,
                'expected': expected,
                'percentage': perc,
            })
            total_expected += expected
            total_present += present

        # Marks-based totals (existing behavior)
        totals = {
            'enrolled': sum(
                cls.students.filter(role='student').count() for cls in target_classes
            ) if target_classes else 0,
            'expected': total_expected,
            'present': total_present,
            'absent': max(0, total_expected - total_present),
            'percentage': round((total_present / total_expected * 100) if total_expected else 0, 2)
        }

        # Headcount totals across selected range (unique students present at least once)
        enrolled_headcount = 0
        present_headcount = 0
        for cls in target_classes:
            try:
                student_ids = list(cls.students.filter(role='student').values_list('id', flat=True))
            except Exception:
                student_ids = []
            enrolled_headcount += len(student_ids)
            # students present at least once in range for this class
            present_once = 0
            for sid in student_ids:
                seen = False
                for d in day_list:
                    if present_map.get((cls.id, sid, d), False):
                        seen = True
                        break
                if seen:
                    present_once += 1
            present_headcount += present_once
        totals_headcount = {
            'enrolled': enrolled_headcount,
            'expected': enrolled_headcount,  # for headcount, expected equals enrolled
            'present': present_headcount,
            'absent': max(0, enrolled_headcount - present_headcount),
            'percentage': round((present_headcount / enrolled_headcount * 100) if enrolled_headcount else 0, 2)
        }

        # Trend with granularity grouping
        # Build per-day aggregates first
        by_day_present = {}
        for (cid, sid, d), is_p in present_map.items():
            if is_p:
                by_day_present[d] = by_day_present.get(d, 0) + 1

        def _week_start(dt):
            # Monday as week start
            dow = dt.weekday()  # 0=Mon
            return dt - timedelta(days=dow)

        def _month_start(dt):
            return dt.replace(day=1)

        trend = []
        if granularity == 'daily':
            series = day_list
            for d in series[-12:]:  # last up to 12 days
                trend.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'label': d.strftime('%b %d'),
                    'present': by_day_present.get(d, 0)
                })
        elif granularity == 'weekly':
            # group days by week start
            buckets = {}
            order = []
            for d in day_list:
                ws = _week_start(d)
                if ws not in buckets:
                    buckets[ws] = 0
                    order.append(ws)
                buckets[ws] += by_day_present.get(d, 0)
            for ws in order[-12:]:  # last up to 12 weeks
                trend.append({
                    'date': ws.strftime('%Y-%m-%d'),
                    'label': f"Wk of {ws.strftime('%b %d')}",
                    'present': buckets.get(ws, 0)
                })
        else:  # monthly
            buckets = {}
            order = []
            for d in day_list:
                ms = _month_start(d)
                if ms not in buckets:
                    buckets[ms] = 0
                    order.append(ms)
                buckets[ms] += by_day_present.get(d, 0)
            for ms in order[-12:]:  # last up to 12 months
                trend.append({
                    'date': ms.strftime('%Y-%m'),
                    'label': ms.strftime('%b %Y'),
                    'present': buckets.get(ms, 0)
                })

        # CSV export of class breakdown
        if request.GET.get('export') == 'csv':
            import csv
            resp = HttpResponse(content_type='text/csv')
            resp['Content-Disposition'] = 'attachment; filename="attendance_analytics.csv"'
            w = csv.writer(resp)
            w.writerow(['Class','Present','Absent','Expected','Percentage'])
            for row in classes_payload:
                w.writerow([row['name'], row['present'], row['absent'], row['expected'], row['percentage']])
            return resp

        payload = {
            'totals': totals,  # marks totals retained for backward compatibility
            'totals_marks': totals,
            'totals_headcount': totals_headcount,
            'trend': trend,
            'classes': classes_payload,
        }
        cache.set(cache_key, payload, timeout=300)
        payload['correlation_id'] = corr_id
        logger.info('attendance_analytics_api ok corr_id=%s user=%s', corr_id, request.user.id)
        return JsonResponse(payload)
    except Exception as e:
        logger.exception('attendance_analytics_api error corr_id=%s err=%s', corr_id, e)
        return JsonResponse({'error': 'Internal server error', 'correlation_id': corr_id}, status=500)


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics_drilldown(request):
    """
    Drilldown API returning student-level attendance records for a selected class or all visible classes.
    GET params: class_id (optional), start_date, end_date, page, page_size, q (student name contains), export=csv
    Returns: { results: [ {student_id, student_name, class_id, class_name, date, is_present} ], page, page_size, total }
    """
    corr_id = str(uuid.uuid4())
    try:
        # Visible classes
        if request.user.role == 'admin':
            visible_classes = list(Class.objects.all())
        else:
            visible_classes = list(Class.objects.filter(staff=request.user))
        visible_ids = {c.id for c in visible_classes}

        def _parse_int(v):
            try:
                return int(v)
            except Exception:
                return None
        def _parse_date(s):
            try:
                return datetime.strptime(s, '%Y-%m-%d').date()
            except Exception:
                return None

        class_id = _parse_int(request.GET.get('class_id'))
        start_date = _parse_date(request.GET.get('start_date',''))
        end_date = _parse_date(request.GET.get('end_date',''))
        q = (request.GET.get('q') or '').strip()
        page = _parse_int(request.GET.get('page')) or 1
        page_size = _parse_int(request.GET.get('page_size')) or 25
        page_size = max(1, min(page_size, 200))

        today = timezone.now().date()
        if not start_date and not end_date:
            start_date = today - timedelta(days=6)
            end_date = today
        elif start_date and not end_date:
            end_date = today
        elif end_date and not start_date:
            start_date = end_date
        if start_date > end_date:
            return JsonResponse({'error': 'start_date must be on or before end_date', 'correlation_id': corr_id}, status=400)

        # Classes scope
        if class_id:
            if class_id not in visible_ids:
                return JsonResponse({'error': 'Class not found or not permitted', 'correlation_id': corr_id}, status=404)
            target_ids = [class_id]
        else:
            target_ids = list(visible_ids)

        # Date window
        day_start, _ = _day_bounds(start_date)
        _, day_end = _day_bounds(end_date)

        qs = Attendance.objects.filter(
            class_instance_id__in=target_ids,
            timestamp__gte=day_start,
            timestamp__lt=day_end,
        ).select_related('student','class_instance').order_by('-timestamp')
        if q:
            qs = qs.filter(
                Q(student__first_name__icontains=q) |
                Q(student__last_name__icontains=q) |
                Q(student__username__icontains=q)
            )

        # CSV export
        if request.GET.get('export') == 'csv':
            import csv
            resp = HttpResponse(content_type='text/csv')
            resp['Content-Disposition'] = 'attachment; filename="attendance_drilldown.csv"'
            w = csv.writer(resp)
            w.writerow(['Time','Student','Class','Present'])
            for r in qs.iterator():
                try:
                    sname = r.student.get_full_name() or r.student.username
                except Exception:
                    sname = str(r.student_id)
                try:
                    cname = r.class_instance.name
                except Exception:
                    cname = str(r.class_instance_id)
                w.writerow([
                    timezone.localtime(r.timestamp).strftime('%Y-%m-%d %H:%M'),
                    sname,
                    cname,
                    'Yes' if r.is_present else 'No'
                ])
            return resp

        total = qs.count()
        start_idx = (page-1) * page_size
        end_idx = start_idx + page_size
        page_qs = list(qs[start_idx:end_idx])
        results = []
        for r in page_qs:
            try:
                sname = r.student.get_full_name() or r.student.username
            except Exception:
                sname = str(r.student_id)
            try:
                cname = r.class_instance.name
            except Exception:
                cname = str(r.class_instance_id)
            results.append({
                'student_id': r.student_id,
                'student_name': sname,
                'class_id': r.class_instance_id,
                'class_name': cname,
                'timestamp': timezone.localtime(r.timestamp).strftime('%Y-%m-%d %H:%M'),
                'is_present': bool(r.is_present),
            })

        payload = {
            'results': results,
            'page': page,
            'page_size': page_size,
            'total': total,
            'correlation_id': corr_id,
        }
        logger.info('attendance_analytics_drilldown ok corr_id=%s user=%s', corr_id, request.user.id)
        return JsonResponse(payload)
    except Exception as e:
        logger.exception('attendance_analytics_drilldown error corr_id=%s err=%s', corr_id, e)
        return JsonResponse({'error': 'Internal server error', 'correlation_id': corr_id}, status=500)


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics_export_class_csv(request):
    """
    Export class-wise breakdown as CSV for the current filters.
    """
    # Reuse filter form
    form = AttendanceFilterForm(request.GET or None)
    if form.is_valid():
        selected_class = form.cleaned_data.get('class_instance')
        start_date = form.cleaned_data.get('start_date')
        end_date = form.cleaned_data.get('end_date')
    else:
        selected_class = None
        start_date = None
        end_date = None

    today = timezone.now().date()
    if not start_date and not end_date:
        start_date = today - timedelta(days=6)
        end_date = today
    elif start_date and not end_date:
        end_date = today
    elif end_date and not start_date:
        start_date = end_date

    day_list = []
    if start_date and end_date and start_date <= end_date:
        cur = start_date
        while cur <= end_date:
            day_list.append(cur)
            cur += timedelta(days=1)

    # Determine visible classes
    if request.user.role == 'admin':
        visible_classes = list(Class.objects.all())
    else:
        visible_classes = list(Class.objects.filter(staff=request.user))

    if selected_class:
        target_classes = [selected_class] if selected_class in visible_classes else []
    else:
        target_classes = visible_classes

    # Build present_map like in analytics
    present_map = {}
    if day_list and target_classes:
        day_start, _ = _day_bounds(day_list[0])
        _, day_end = _day_bounds(day_list[-1])
        attendance_qs = Attendance.objects.filter(
            class_instance__in=target_classes,
            timestamp__gte=day_start,
            timestamp__lt=day_end,
        ).values('class_instance_id', 'student_id', 'timestamp', 'is_present')
        for r in attendance_qs:
            try:
                day_key = timezone.localtime(r['timestamp']).date()
            except Exception:
                day_key = r['timestamp'].date()
            key = (r['class_instance_id'], r['student_id'], day_key)
            if r['is_present']:
                present_map[key] = True

    # Compute class breakdown
    import csv
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="class_breakdown.csv"'
    writer = csv.writer(response)
    writer.writerow(['Class', 'Present', 'Absent', 'Expected', 'Percentage'])

    for cls in target_classes:
        try:
            enrolled_students = list(cls.students.filter(role='student').values_list('id', flat=True))
        except Exception:
            enrolled_students = []
        days_count = len(day_list)
        expected = len(enrolled_students) * days_count
        present = 0
        if days_count > 0 and enrolled_students:
            for d in day_list:
                for sid in enrolled_students:
                    if present_map.get((cls.id, sid, d), False):
                        present += 1
        absent = max(0, expected - present)
        perc = round((present / expected * 100) if expected > 0 else 0, 2)
        writer.writerow([cls.name, present, absent, expected, perc])

    return response


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics_export_student_csv(request):
    """
    Export per-student breakdown CSV for the selected class and current date range.
    """
    form = AttendanceFilterForm(request.GET or None)
    if not form.is_valid() or not form.cleaned_data.get('class_instance'):
        return HttpResponse('Class must be selected for per-student export', status=400)

    selected_class = form.cleaned_data.get('class_instance')
    start_date = form.cleaned_data.get('start_date')
    end_date = form.cleaned_data.get('end_date')

    today = timezone.now().date()
    if not start_date and not end_date:
        start_date = today - timedelta(days=6)
        end_date = today
    elif start_date and not end_date:
        end_date = today
    elif end_date and not start_date:
        start_date = end_date

    day_list = []
    if start_date and end_date and start_date <= end_date:
        cur = start_date
        while cur <= end_date:
            day_list.append(cur)
            cur += timedelta(days=1)

    # Build present map for selected class only
    present_map = {}
    if day_list:
        day_start, _ = _day_bounds(day_list[0])
        _, day_end = _day_bounds(day_list[-1])
        attendance_qs = Attendance.objects.filter(
            class_instance=selected_class,
            timestamp__gte=day_start,
            timestamp__lt=day_end,
        ).values('student_id', 'timestamp', 'is_present')
        for r in attendance_qs:
            try:
                day_key = timezone.localtime(r['timestamp']).date()
            except Exception:
                day_key = r['timestamp'].date()
            key = (selected_class.id, r['student_id'], day_key)
            if r['is_present']:
                present_map[key] = True

    # Prepare CSV
    import csv
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="per_student_breakdown.csv"'
    writer = csv.writer(response)
    writer.writerow(['Student', 'Present Days', 'Absent Days', 'Total Days', 'Percentage'])

    try:
        students_in_class = list(selected_class.students.filter(role='student'))
    except Exception:
        students_in_class = []

    total_days = len(day_list)
    for s in students_in_class:
        present_days = 0
        for d in day_list:
            if present_map.get((selected_class.id, s.id, d), False):
                present_days += 1
        absent_days = max(0, total_days - present_days)
        perc = round((present_days / total_days * 100) if total_days > 0 else 0, 2)
        writer.writerow([s.get_full_name() or s.username, present_days, absent_days, total_days, perc])

    return response


@login_required
@user_passes_test(is_staff_member)
def attendance_analytics_metrics(request):
    """
    JSON metrics for Attendance Analytics page to update without full reload.
    Requires: class_id (int). Optional: start_date (YYYY-MM-DD), end_date (YYYY-MM-DD).
    Returns: { enrolled, present, absent, percentage, trend: [{date,label,present}] }
    """
    try:
        class_id = int(request.GET.get('class_id', '0'))
    except Exception:
        return JsonResponse({'error': 'Invalid class_id'}, status=400)

    # Visible classes for this user
    if request.user.role == 'admin':
        visible_classes = list(Class.objects.all())
    else:
        visible_classes = list(Class.objects.filter(staff=request.user))
    visible_ids = {c.id for c in visible_classes}
    if class_id not in visible_ids:
        return JsonResponse({'error': 'Class not found or not permitted'}, status=404)

    try:
        selected_class = next(c for c in visible_classes if c.id == class_id)
    except StopIteration:
        return JsonResponse({'error': 'Class not found'}, status=404)

    # Dates
    def _parse_date(s):
        try:
            return datetime.strptime(s, '%Y-%m-%d').date()
        except Exception:
            return None

    start_date = _parse_date(request.GET.get('start_date', ''))
    end_date = _parse_date(request.GET.get('end_date', ''))
    today = timezone.now().date()
    if not start_date and not end_date:
        start_date = today - timedelta(days=6)
        end_date = today
    elif start_date and not end_date:
        end_date = today
    elif end_date and not start_date:
        start_date = end_date

    # Clamp excessive ranges (max 90 days)
    if start_date and end_date and (end_date - start_date).days > 90:
        start_date = end_date - timedelta(days=90)

    # Build day list
    day_list = []
    if start_date and end_date and start_date <= end_date:
        d = start_date
        while d <= end_date:
            day_list.append(d)
            d += timedelta(days=1)

    # Enrolled students count
    try:
        enrolled_ids = list(selected_class.students.filter(role='student').values_list('id', flat=True))
    except Exception:
        enrolled_ids = []
    enrolled = len(enrolled_ids)

    # Cache key & lookup with per-(class,day) stamp for invalidation
    stamp_vals = []
    if start_date and end_date:
        cur = start_date
        while cur <= end_date:
            stamp_key = f"analytics:stamp:{class_id}:{cur}"
            stamp_vals.append(str(cache.get(stamp_key, 0)))
            cur += timedelta(days=1)
    stamp_str = ":".join(stamp_vals)
    cache_key = f"analytics:metrics:v1:{request.user.role}:{request.user.id}:{class_id}:{start_date}:{end_date}:{stamp_str}"
    cached = cache.get(cache_key)
    if cached is not None:
        return JsonResponse(cached)

    # Attendance within range
    present_map = {}
    present = 0
    if day_list:
        # Prefer session_day date filter (indexed), fallback to timestamp range for legacy rows
        day_start, _ = _day_bounds(day_list[0])
        _, day_end = _day_bounds(day_list[-1])
        qs = Attendance.objects.filter(
            class_instance=selected_class
        ).filter(
            Q(session_day__gte=start_date, session_day__lte=end_date) |
            Q(session_day__isnull=True, timestamp__gte=day_start, timestamp__lt=day_end)
        ).values('student_id', 'timestamp', 'session_day', 'is_present')
        for r in qs:
            # Derive day key from session_day if present; otherwise from timestamp
            day_key = r.get('session_day')
            if not day_key:
                try:
                    day_key = timezone.localtime(r['timestamp']).date()
                except Exception:
                    day_key = r['timestamp'].date()
            key = (r['student_id'], day_key)
            if r['is_present']:
                present_map[key] = True
        # Count present marks across days for enrolled students
        for d in day_list:
            for sid in enrolled_ids:
                if present_map.get((sid, d), False):
                    present += 1

    expected = enrolled * len(day_list)
    absent = max(0, expected - present)
    percentage = round((present / expected * 100) if expected > 0 else 0, 2)

    # Trend: last up to 5 sessions (days with any record) within range
    # Build a sorted unique list of days that had any records for this class within range
    days_with_records = []
    if day_list:
        # Map day -> present count for that day
        by_day_present = {d: 0 for d in day_list}
        for (sid, d), is_p in present_map.items():
            if is_p and d in by_day_present:
                by_day_present[d] += 1
        days_with_records = [d for d in day_list if by_day_present.get(d, 0) or True]
        # Take last 5 days from range
        recent_days = days_with_records[-5:]
        trend = [{'date': d.strftime('%Y-%m-%d'), 'label': d.strftime('%b %d'), 'present': by_day_present.get(d, 0)} for d in recent_days]
    else:
        trend = []

    payload = {
        'enrolled': enrolled,
        'present': present,
        'absent': absent,
        'percentage': percentage,
        'trend': trend,
    }
    cache.set(cache_key, payload, timeout=60 if end_date >= today else 120)
    return JsonResponse(payload)


@login_required
@user_passes_test(is_staff_member)
def attendance_daily_summary(request):
    """
    Return daily present/absent lists for a class and date.
    GET: class_id (required), date (YYYY-MM-DD, default today)
    Response: {
      present: [{id,name,roll,check_in_time}],
      absent: [{id,name,roll}],
      counts: {present, absent, total}
    }
    """
    def _parse_int(v):
        try:
            return int(v)
        except Exception:
            return None
    def _parse_date(s):
        try:
            return datetime.strptime(s, '%Y-%m-%d').date()
        except Exception:
            return None
    class_id = _parse_int(request.GET.get('class_id'))
    if not class_id:
        return JsonResponse({'error': 'class_id is required'}, status=400)

    # Authorization: restrict to classes visible to user
    if request.user.role == 'admin':
        visible_ids = set(Class.objects.values_list('id', flat=True))
    else:
        visible_ids = set(Class.objects.filter(staff=request.user).values_list('id', flat=True))
    if class_id not in visible_ids:
        return JsonResponse({'error': 'Class not found or not permitted'}, status=404)

    target_date = _parse_date(request.GET.get('date', '')) or timezone.now().date()
    day_start, day_end = _day_bounds(target_date)

    # Enrolled students
    try:
        enrolled = list(User.objects.filter(
            role='student', classes_enrolled__id=class_id
        ).only('id', 'first_name', 'last_name', 'username', 'department_id').distinct())
    except Exception:
        enrolled = []
    enrolled_ids = [s.id for s in enrolled]

    # Attendance rows for that day
    rows = list(
        Attendance.objects.filter(
            class_instance_id=class_id
        ).filter(
            Q(session_day=target_date) |
            Q(session_day__isnull=True, timestamp__gte=day_start, timestamp__lt=day_end)
        ).select_related('student').only('student_id', 'timestamp', 'is_present')
    )

    # Determine first check-in per student and presence
    first_checkin = {}
    present_ids = set()
    for r in rows:
        if r.is_present:
            present_ids.add(r.student_id)
            ts = r.timestamp
            prev = first_checkin.get(r.student_id)
            if not prev or ts < prev:
                first_checkin[r.student_id] = ts

    def _name(u):
        try:
            n = (u.first_name or '').strip() + ' ' + (u.last_name or '').strip()
            return n.strip() or u.username
        except Exception:
            return getattr(u, 'username', str(getattr(u, 'id', '')))

    present_list = []
    absent_list = []
    for u in enrolled:
        item = {
            'id': u.id,
            'name': _name(u),
            'roll': getattr(u, 'department_id', '') or '',
        }
        if u.id in present_ids:
            ts = first_checkin.get(u.id)
            item['check_in_time'] = timezone.localtime(ts).strftime('%H:%M') if ts else None
            present_list.append(item)
        else:
            absent_list.append(item)

    present_list.sort(key=lambda x: (x.get('check_in_time') or '99:99', x['name'].lower()))
    absent_list.sort(key=lambda x: x['name'].lower())

    payload = {
        'present': present_list,
        'absent': absent_list,
        'counts': {
            'present': len(present_list),
            'absent': len(absent_list),
            'total': len(enrolled_ids),
        }
    }
    return JsonResponse(payload)


@login_required
@user_passes_test(is_staff_member)
def attendance_student_history(request):
    """
    Return per-student presence over last N days with counts and timeline.
    GET: student_id (required), class_id (optional to restrict to a class), days (7/15/30/60; default 30)
    Response: { counts: {present, absent, total_days}, series: [{date, present}], student: {id,name,roll} }
    """
    def _parse_int(v):
        try:
            return int(v)
        except Exception:
            return None
    student_id = _parse_int(request.GET.get('student_id'))
    if not student_id:
        return JsonResponse({'error': 'student_id is required'}, status=400)
    class_id = _parse_int(request.GET.get('class_id'))
    days = _parse_int(request.GET.get('days')) or 30
    days = days if days in (7, 15, 30, 60) else 30

    # Fetch student and authorize
    try:
        student = User.objects.get(id=student_id, role='student')
    except User.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)

    if request.user.role == 'admin':
        visible_classes = set(Class.objects.values_list('id', flat=True))
    else:
        visible_classes = set(Class.objects.filter(staff=request.user).values_list('id', flat=True))

    if class_id and class_id not in visible_classes:
        return JsonResponse({'error': 'Class not found or not permitted'}, status=404)

    # Build date range (inclusive)
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=days-1)

    # Build day list
    day_list = []
    d = start_date
    while d <= end_date:
        day_list.append(d)
        d += timedelta(days=1)

    # Query attendance
    qs = Attendance.objects.filter(student_id=student_id)
    if class_id:
        qs = qs.filter(class_instance_id=class_id)
    day_start, _ = _day_bounds(start_date)
    _, day_end = _day_bounds(end_date)
    qs = qs.filter(
        Q(session_day__gte=start_date, session_day__lte=end_date) |
        Q(session_day__isnull=True, timestamp__gte=day_start, timestamp__lt=day_end)
    ).values('timestamp','session_day','is_present')

    present_days = set()
    for r in qs:
        dk = r.get('session_day')
        if not dk:
            try:
                dk = timezone.localtime(r['timestamp']).date()
            except Exception:
                dk = r['timestamp'].date()
        if r['is_present']:
            present_days.add(dk)

    series = []
    present_count = 0
    for d in day_list:
        is_p = d in present_days
        if is_p:
            present_count += 1
        series.append({'date': d.strftime('%Y-%m-%d'), 'present': 1 if is_p else 0})

    payload = {
        'student': {
            'id': student.id,
            'name': (student.get_full_name() or student.username),
            'roll': getattr(student, 'department_id', '') or '',
        },
        'counts': {
            'present': present_count,
            'absent': max(0, len(day_list) - present_count),
            'total_days': len(day_list),
        },
        'series': series,
    }
    return JsonResponse(payload)


@login_required
@user_passes_test(is_admin)
@csrf_exempt
@require_http_methods(["POST"])
def save_geofence_coordinates(request):
    """
    API endpoint for saving geo-fence coordinates from Google Maps.
    """
    try:
        data = json.loads(request.body)
        geofence_id = data.get('geofence_id')
        coordinates = data.get('coordinates')
        
        if not coordinates:
            return JsonResponse({'success': False, 'error': 'Coordinates are required'}, status=400)
        
        if geofence_id:
            # Update existing geo-fence
            try:
                geofence = GeoFence.objects.get(id=geofence_id)
                geofence.coordinates = coordinates
                geofence.save()
                return JsonResponse({'success': True, 'message': 'Geo-fence updated successfully', 'geofence_id': geofence.id})
            except GeoFence.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Geo-fence not found'}, status=404)
        else:
            # Create new geo-fence (coordinates will be saved when form is submitted)
            return JsonResponse({'success': True, 'message': 'Coordinates saved for new geo-fence'})
            
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
