from django import forms

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from account_app.models import UserModel


class CreateAdminForm(UserCreationForm):
	email = forms.CharField(label="이메일")
	username = forms.CharField(label="이름")
	password1 = forms.CharField(label="비밀번호", widget=forms.PasswordInput)
	password2 = forms.CharField(label="비밀번호 확인", widget=forms.PasswordInput)
	MANAGEMENT_LOCATIONS_CHOICES = [
		('option1', '구역 1'),
		('option2', '구역 2'),
		('option3', '구역 3'),
	]
	management_locations = forms.ChoiceField(label="관리 위치", choices=MANAGEMENT_LOCATIONS_CHOICES)
	phone = forms.CharField(label="휴대전화")

	class Meta:
		model = UserModel
		fields = ('email', 'username', 'password1', 'password2', 'management_locations', 'phone')
		labels = {
			'username': '아이디',
			'password1': '비밀번호',
			'password2': '비밀번호 확인',
		}
		help_texts = {
			'username': None,
			'password1': None,
			'password2': None,
			'email': None,
		}


class AdminUpdateForm(CreateAdminForm):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.fields['username'].disabled = True


class AdminDetailForm(AdminUpdateForm):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		del self.fields['password1']
		del self.fields['password2']

		for field in self.fields:
			self.fields[field].disabled = True