# Generated by Django 4.1.5 on 2023-06-02 03:55

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LogEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source', models.CharField(choices=[('webcam', 'Webcam')], max_length=10)),
                ('execution_num', models.IntegerField()),
                ('event_type', models.CharField(max_length=50)),
                ('event_time', models.DateTimeField()),
            ],
            options={
                'verbose_name': '로그',
                'verbose_name_plural': '로그',
                'db_table': 'LogEntry',
            },
        ),
        migrations.CreateModel(
            name='Setting',
            fields=[
                ('setno', models.AutoField(primary_key=True, serialize=False, verbose_name='설정번호')),
                ('cammode', models.IntegerField(choices=[(1, 'WebCam'), (2, 'USBCam'), (3, 'IPCam')], verbose_name='카메라 모드')),
                ('camIP', models.CharField(blank=True, max_length=20, null=True, verbose_name='카메라 IP')),
                ('camport', models.IntegerField(blank=True, null=True, verbose_name='카메라 포트')),
                ('alarmmode', models.CharField(choices=[(1, 'SMS 문자메세지'), (2, '디스코드'), (3, '이메일')], max_length=20, verbose_name='알람 모드')),
                ('alarmsend', models.CharField(max_length=30, verbose_name='알람 전송 대상')),
                ('logpath', models.CharField(blank=True, max_length=50, null=True, verbose_name='로그 저장 경로')),
                ('videorecordlength', models.IntegerField(blank=True, null=True, verbose_name='영상 저장 길이')),
            ],
            options={
                'verbose_name': '설정',
                'verbose_name_plural': '설정',
                'db_table': 'Setting',
            },
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('fileNo', models.AutoField(primary_key=True, serialize=False, verbose_name='파일번호')),
                ('filepath', models.CharField(max_length=100, verbose_name='파일 경로')),
                ('regdate', models.DateField(default=datetime.date.today, verbose_name='등록날짜')),
                ('title', models.CharField(blank=True, max_length=100, verbose_name='제목')),
                ('video_file', models.FileField(upload_to='videos/', verbose_name='비디오 파일')),
            ],
            options={
                'verbose_name': 'Video Upload',
                'verbose_name_plural': 'Video Upload',
                'db_table': 'Video',
            },
        ),
    ]