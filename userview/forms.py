"""
Forms for the userview app.
"""
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import PasswordResetForm as DjangoPasswordResetForm
from .models import User
import re


class UserRegistrationForm(forms.ModelForm):
    """
    Simplified registration form with only:
    - role (Admin/Staff/Student)
    - username (User ID used for login)
    - display_name (stored in first_name)
    - password (used for login)
    - email (used for password reset)
    """
    role = forms.ChoiceField(choices=User.ROLE_CHOICES, required=True)
    username = forms.CharField(required=True)
    display_name = forms.CharField(required=True, label='User Name')
    password = forms.CharField(required=True, widget=forms.PasswordInput)
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'role', 'email')

    def clean_username(self):
        username = (self.cleaned_data.get('username') or '').strip()
        if not username:
            raise forms.ValidationError('Please enter a User ID.')
        if not re.match(r'^[\w.@+-]+$', username):
            raise forms.ValidationError('Enter a valid User ID. Only letters, numbers and @/./+/-/_ allowed.')
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError('A user with that User ID already exists.')
        return username

    def clean_email(self):
        email = (self.cleaned_data.get('email') or '').strip().lower()
        if not email:
            raise forms.ValidationError('Please enter an email address.')
        # Optional uniqueness check; comment out if you allow duplicates
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('A user with that email already exists.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data['username'].strip()
        user.first_name = self.cleaned_data['display_name']
        user.last_name = ''
        user.email = self.cleaned_data['email']
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


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'email', 'department_id')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['first_name'].label = 'User Name'
        self.fields['first_name'].required = True
        self.fields['email'].required = True
        self.fields['first_name'].widget.attrs.update({'class': 'form-control'})
        self.fields['email'].widget.attrs.update({'class': 'form-control'})
        self.fields['department_id'].required = False
        self.fields['department_id'].widget.attrs.update({'class': 'form-control'})

    def clean_email(self):
        email = (self.cleaned_data.get('email') or '').strip().lower()
        if not email:
            raise forms.ValidationError('Please enter an email address.')
        try:
            for u in User.objects.all():
                try:
                    if u.pk == self.instance.pk:
                        continue
                    if (u.email or '').strip().lower() == email:
                        raise forms.ValidationError('A user with that email already exists.')
                except forms.ValidationError:
                    raise
                except Exception:
                    continue
        except forms.ValidationError:
            raise
        except Exception:
            pass
        return email


class PasswordResetForm(DjangoPasswordResetForm):
    """
    Custom PasswordResetForm to avoid djongo iLIKE issues by performing
    case-insensitive email matching in Python after fetching active users.
    """
    def get_users(self, email):
        UserModel = get_user_model()
        try:
            email_norm = (email or "").strip().lower()
            # Fetch only active users to reduce dataset; avoid email__iexact (iLIKE) in djongo
            active_users = UserModel._default_manager.filter(is_active=True)
            for user in active_users:
                try:
                    if (user.email or "").strip().lower() == email_norm:
                        yield user
                except Exception:
                    continue
        except Exception:
            return []
