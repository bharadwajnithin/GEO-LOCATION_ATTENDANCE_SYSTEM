"""
Forms for the adminview app.
"""
from django import forms
from userview.models import Class, GeoFence, User


class ClassForm(forms.ModelForm):
    """
    Form for creating and editing classes.
    """
    class Meta:
        model = Class
        fields = ['name', 'staff', 'geolocation']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'staff': forms.Select(attrs={'class': 'form-control'}),
            'geolocation': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filter staff members only
        self.fields['staff'].queryset = User.objects.filter(role__in=['staff', 'admin'])


class GeoFenceForm(forms.ModelForm):
    """
    Form for creating and editing geo-fences.
    """
    coordinates_json = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        help_text="Geo-fence coordinates in JSON format"
    )
    
    class Meta:
        model = GeoFence
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
        }
    
    def clean_coordinates_json(self):
        """
        Validate and parse coordinates JSON.
        """
        import json
        coordinates_json = self.cleaned_data.get('coordinates_json')
        
        if not coordinates_json:
            raise forms.ValidationError("Coordinates are required.")
        
        try:
            coordinates = json.loads(coordinates_json)
            if not isinstance(coordinates, list) or len(coordinates) < 3:
                raise forms.ValidationError("At least 3 coordinate points are required.")
            
            for coord in coordinates:
                if not isinstance(coord, dict) or 'lat' not in coord or 'lng' not in coord:
                    raise forms.ValidationError("Invalid coordinate format.")
                
                try:
                    lat = float(coord['lat'])
                    lng = float(coord['lng'])
                    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                        raise forms.ValidationError("Invalid latitude or longitude values.")
                except (ValueError, TypeError):
                    raise forms.ValidationError("Invalid coordinate values.")
            
            return coordinates
            
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format.")
    
    def save(self, commit=True):
        """
        Save the geo-fence with parsed coordinates.
        """
        instance = super().save(commit=False)
        if self.cleaned_data.get('coordinates_json'):
            instance.coordinates = self.cleaned_data['coordinates_json']
        
        if commit:
            instance.save()
        return instance


class StudentEnrollmentForm(forms.Form):
    """
    Form for enrolling students in classes.
    """
    students = forms.ModelMultipleChoiceField(
        queryset=User.objects.filter(role='student'),
        widget=forms.CheckboxSelectMultiple,
        required=False
    )
    
    def __init__(self, *args, **kwargs):
        class_instance = kwargs.pop('class_instance', None)
        super().__init__(*args, **kwargs)
        
        # Don't try to filter enrolled students here - handle it in the view
        # This avoids complex queryset operations that cause djongo issues


class AttendanceFilterForm(forms.Form):
    """
    Form for filtering attendance records.
    """
    class_instance = forms.ModelChoiceField(
        queryset=Class.objects.all(),
        required=False,
        empty_label="All Classes",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    start_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    end_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    student = forms.ModelChoiceField(
        queryset=User.objects.filter(role='student'),
        required=False,
        empty_label="All Students",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
