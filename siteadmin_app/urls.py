from django.urls import path
from siteadmin_app.views import SiteAdminListView, SiteAdminCreateView, SiteAdminDetailView
from siteadmin_app.views import SiteAdminUpdateView, SiteAdminDeleteView

app_name = 'siteadmin_app'

urlpatterns = [
    path('list/', SiteAdminListView.as_view(), name='list'),
    path('create/', SiteAdminCreateView.as_view(), name='create'),
    path('detail/<int:pk>', SiteAdminDetailView.as_view(), name='detail'),
    path('update/<int:pk>', SiteAdminUpdateView.as_view(), name='update'),
    path('delete/<int:pk>', SiteAdminDeleteView.as_view(), name='delete'),
]
