from django import forms
from .models import Video
from django.contrib.auth.forms import AuthenticationForm

class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ('title', 'video_file')
        widgets = {
            'title': forms.TextInput(attrs={'required': True})
        }

# account_app 에서 대체
# class LoginForm(AuthenticationForm):
#     username = forms.CharField(label='ID')
#     password = forms.CharField(label='Password', widget=forms.PasswordInput)
