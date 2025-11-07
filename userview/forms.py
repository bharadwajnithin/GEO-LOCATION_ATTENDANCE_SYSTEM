"""
Forms for the userview app.
"""
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import User
import re


class UserRegistrationForm(forms.ModelForm):
    """
    Simplified registration form with only:
    - role (Admin/Staff/Student)
    - username (User ID used for login)
    - display_name (stored in first_name)
    - password (used for login)
    """
    role = forms.ChoiceField(choices=User.ROLE_CHOICES, required=True)
    username = forms.CharField(required=True)
    display_name = forms.CharField(required=True, label='User Name')
    password = forms.CharField(required=True, widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'role')

    def clean_username(self):
        username = (self.cleaned_data.get('username') or '').strip()
        if not username:
            raise forms.ValidationError('Please enter a User ID.')
        if not re.match(r'^[\w.@+-]+$', username):
            raise forms.ValidationError('Enter a valid User ID. Only letters, numbers and @/./+/-/_ allowed.')
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError('A user with that User ID already exists.')
        return username

    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data['username'].strip()
        user.first_name = self.cleaned_data['display_name']
        user.last_name = ''
        user.email = ''
        user.role = self.cleaned_data['role']
        # set password
        user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
        return user


class UserLoginForm(AuthenticationForm):
    """
    Custom user login form.
    """
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'User ID'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )


class FaceDataForm(forms.Form):
    """
    Deprecated in simplified registration flow. Kept for compatibility but unused.
    """
    pass
