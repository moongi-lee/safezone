from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.contrib.auth.models import Group, Permission
import datetime
# Create your models here.

# account_app 에서 대체
# class UserManager(BaseUserManager):
#     def create_user(self, email, password=None, **extra_fields):
#         if not email:
#             raise ValueError("The Email field must be set")
#         email = self.normalize_email(email)
#         user = self.model(email=email, **extra_fields)
#         user.set_password(password)
#         user.save(using=self._db)
#         return user
#
#     def create_superuser(self, email, password=None, **extra_fields):
#         extra_fields.setdefault('is_staff', True)
#         extra_fields.setdefault('is_superuser', True)
#         return self.create_user(email, password, **extra_fields)
#
# class User(AbstractBaseUser, PermissionsMixin):
#     email = models.EmailField(unique=True)
#     first_name = models.CharField(max_length=30)
#     last_name = models.CharField(max_length=30)
#     is_active = models.BooleanField(default=True)
#     regdate = models.DateTimeField(auto_now_add=True)
#
#     groups = models.ManyToManyField(
#         Group,
#         verbose_name='groups',
#         blank=True,
#         help_text='The groups this user belongs to.',
#         related_name='user_set+'  # related_name 을 'user_set+'으로 설정
#     )
#     user_permissions = models.ManyToManyField(
#         Permission,
#         verbose_name='user permissions',
#         blank=True,
#         help_text='Specific permissions for this user.',
#         related_name='user_set+'  # related_name 을 'user_set+'으로 설정
#     )
#
#     USERNAME_FIELD = 'email'
#     REQUIRED_FIELDS = ['first_name', 'last_name']
#
#     objects = UserManager()
#
#     def get_full_name(self):
#         return f'{self.first_name} {self.last_name}'
#
#     def get_short_name(self):
#         return self.first_name


class Video(models.Model):
    fileNo = models.AutoField(primary_key=True, verbose_name='파일번호')
    filepath = models.CharField(max_length=100, verbose_name='파일 경로')
    regdate = models.DateField(default=datetime.date.today, verbose_name='등록날짜')
    title = models.CharField(max_length=100, verbose_name='제목', blank=True)
    video_file = models.FileField(upload_to='videos/', verbose_name='비디오 파일')

    class Meta:
        db_table = 'Video'
        verbose_name = 'Video Upload'
        verbose_name_plural = 'Video Upload'
    
    def __str__(self):
        return f'파일번호: {self.fileNo}'
    
class Setting(models.Model):
    CAM_MODE_CHOICES = (
        (1, 'WebCam'),
        (2, 'USBCam'),
        (3, 'IPCam'),
    )

    ALARM_MODE_CHOICES = (
        (1, 'SMS 문자메세지'),
        (2, '디스코드'),
        (3, '이메일'),
    )

    setno = models.AutoField(primary_key=True, verbose_name='설정번호')
    cammode = models.IntegerField(choices=CAM_MODE_CHOICES, verbose_name='카메라 모드')
    camIP = models.CharField(max_length=20, null=True, blank=True, verbose_name='카메라 IP')
    camport = models.IntegerField(null=True, blank=True, verbose_name='카메라 포트')
    alarmmode = models.CharField(choices=ALARM_MODE_CHOICES, max_length=20, verbose_name='알람 모드')
    alarmsend = models.CharField(max_length=30, verbose_name='알람 전송 대상')
    logpath = models.CharField(max_length=50, null=True, blank=True, verbose_name='로그 저장 경로')
    videorecordlength = models.IntegerField(null=True, blank=True, verbose_name='영상 저장 길이')

    class Meta:
        db_table = 'Setting'
        verbose_name = '설정'
        verbose_name_plural = '설정'

    def __str__(self):
        return f'설정번호: {self.setno}'
    
class LogEntry(models.Model):
    SOURCE_CHOICES = [
        ('webcam', 'Webcam'),
    ]

    source = models.CharField(max_length=10, choices=SOURCE_CHOICES)
    execution_num = models.IntegerField()
    event_type = models.CharField(max_length=50)
    event_time = models.DateTimeField()

    class Meta:
        db_table = 'LogEntry'
        verbose_name = '로그'
        verbose_name_plural = '로그'

    def __str__(self):
        return f"LogEntry - Source: {self.source}, Execution Num: {self.execution_num}, Event Type: {self.event_type}, Event Time: {self.event_time}"
    
    