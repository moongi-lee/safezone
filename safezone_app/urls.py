"""safezone URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from safezone_app import views
from django.conf import settings
from django.conf.urls.static import static

# 모듈 불러오는 방식 변경
# safezone app 이름 등록
# url 사용할때 {% url 'video_detail' %} X
# {% url 'safezone_app:upload_video' %} O
app_name = 'safezone_app'
urlpatterns = [
    path('', views.main, name='main'),
    path('settings/<str:memberid>', views.settings, name='settings'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('video/', views.video, name='video'),
    path('video_analyze/', views.video_analyze, name='video_analyze'),
    path('video_detail/<int:fileNo>/', views.video_detail, name='video_detail'),
    path('yolov5_webcam/', views.yolov5_webcam, name='yolov5_webcam'),
    path('run_yolov5_webcam/', views.run_yolov5_webcam, name='run_yolov5_webcam'),
    path('video_feed/<int:id>', views.video_feed, name='video_feed'),
    path('get_log/', views.get_log, name='get_log'),
]
