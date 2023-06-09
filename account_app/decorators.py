from django.http import HttpResponseForbidden

from account_app.models import UserModel

def admin_ownership_required(func):
	def decorated(request, *args, **kwargs):
		user = UserModel.objects.get(pk=kwargs['pk'])
		if user.pk != request.user.pk:
			return HttpResponseForbidden()
		return func(request, *args, **kwargs)
	return decorated