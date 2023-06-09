from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from account_app.views import CreateAdminView, AdminProfileView, AdminProfileUpdateView, AdminDeleteView


app_name = 'account_app'


urlpatterns = [
	path('create/', CreateAdminView.as_view(), name='create'),  # name = html  라우팅 참조값  #  reverse_lazy('account_app:main')
	path('login/', LoginView.as_view(template_name='account_app/login.html'), name='login'),
	path('logout/', LogoutView.as_view(), name='logout'),
	path('profile/<int:pk>', AdminProfileView.as_view(), name='profile'),
	path('update/<int:pk>', AdminProfileUpdateView.as_view(), name='update'),
	path('delete/<int:pk>', AdminDeleteView.as_view(), name='delete'),

]
