from django.forms import ModelForm
from siteadmin_app.models import SiteAdmin


class SiteAdminCreationForm(ModelForm):
	class Meta:
		model = SiteAdmin
		fields = ['name', 'image', 'management_locations', 'phone']




class SiteAdminDetailForm(SiteAdminCreationForm):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for field in self.fields:
			self.fields[field].disabled = True

