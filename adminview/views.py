"""
Views for the adminview app.
"""
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Q, Count
from django.utils import timezone
from datetime import datetime, timedelta, time
from .forms import ClassForm, GeoFenceForm, StudentEnrollmentForm, AttendanceFilterForm
from userview.models import User, Class, Attendance, GeoFence
from django.conf import settings


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
    
    # Get recent attendance records
    recent_attendance = Attendance.objects.filter(
        class_instance__in=classes
    ).select_related('student', 'class_instance').order_by('-timestamp')[:10]
    
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
        'recent_attendance': recent_attendance,
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
    # Always treat GET the same; default filters when not provided
    form = AttendanceFilterForm(request.GET or None)

    # Determine classes visible to this user
    if request.user.role == 'admin':
        visible_classes = list(Class.objects.all())
    else:
        visible_classes = list(Class.objects.filter(staff=request.user))

    # Restrict form class choices to visible classes
    try:
        if hasattr(form, 'fields') and 'class_instance' in form.fields:
            form.fields['class_instance'].queryset = Class.objects.filter(id__in=[c.id for c in visible_classes])
    except Exception:
        pass

    # Parse filters with sensible defaults
    if form.is_valid():
        selected_class = form.cleaned_data.get('class_instance')
        selected_student = form.cleaned_data.get('student')
        start_date = form.cleaned_data.get('start_date')
        end_date = form.cleaned_data.get('end_date')
    else:
        selected_class = None
        selected_student = None
        start_date = None
        end_date = None

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
    for r in attendance_records:
        day_key = r.timestamp.astimezone(timezone.get_current_timezone()).date()
        key = (r.class_instance_id, r.student_id, day_key)
        if r.is_present:
            present_map[key] = True or present_map[key]
        # keep a few recent records for display
        raw_records.append(r)

    # Compute class-wise summary including unmarked as absent
    class_breakdown = {}
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

    context = {
        'form': form,
        'attendance_records': records_for_table,
        'total_records': total_expected_marks,
        'present_records': total_present_marks,
        'attendance_percentage': attendance_percentage,
        'class_breakdown': class_breakdown,
        'student_breakdown': student_breakdown,
        'date_range': {'start': start_date, 'end': end_date},
    }
    return render(request, 'adminview/attendance_analytics.html', context)


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
                return JsonResponse({'success': True, 'message': 'Geo-fence updated successfully'})
            except GeoFence.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Geo-fence not found'}, status=404)
        else:
            # Create new geo-fence (coordinates will be saved when form is submitted)
            return JsonResponse({'success': True, 'message': 'Coordinates saved for new geo-fence'})
            
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
