from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, HttpResponse, get_object_or_404
from django.http import Http404, StreamingHttpResponse, HttpResponseServerError, JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators import gzip
import datetime
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage, default_storage
from django.core.files.base import ContentFile
from account_app.decorators import admin_ownership_required
import cv2
import os
from torchvision import transforms
from PIL import Image
from django.urls import reverse
from .forms import VideoForm
from .models import Video, LogEntry
from yolov5.models.experimental import *
import subprocess
import json
from collections import deque
import base64
import time
# Create your views here.
# @login_required
def main(request):
    #rtsp_url = 'rtsp://210.99.70.120:1935/live/cctv001.stream'
    #display_rtsp_video(rtsp_url)
    return render(request, 'main.html')

def settings(request):
    if request.method == 'POST':
        log_interval = request.POST.get('log-interval')
        log_location = request.POST.get('log-location')
        video_location = request.POST.get('video-location')
        
        # 데이터 처리 및 저장 로직 작성
        # 기존 설정이 있는지 확인합니다.
        settings = get_object_or_404(Video, title='settings')
        
        # 기존 설정이 없으면 새로운 객체를 생성합니다.
        if not settings:
            settings = Video(title='settings')
        
        # 새로운 설정값을 업데이트합니다.
        settings.log_interval = log_interval
        settings.log_location = log_location
        settings.video_location = video_location
        
        # 설정을 저장합니다.
        settings.save()
        
        return HttpResponse('Settings updated successfully!')
    
    return render(request, 'settings.html')
    


def upload_video(request):
    if request.method == 'POST':                            # form 으로 Method=POST 로 받아와서 작업
        form = VideoForm(request.POST, request.FILES)       # forms.py 에서 작업하기위해 POST 로 요청, FILES 를 불러옴 
        if form.is_valid():                                 # form 의 title, video_file 형식으로 들어오는지 유효성 검사
            
            video_file = request.FILES.get('video_file')    # <label for="video_file">Title:</label>
            video_name = video_file.name                    # <input type="file" class="form-control-file" id="video_file" name="video_file">
                                                            # 에서 받아온 video_file 을 get files 화
            video = form.save(commit=False)                 # file 들어온 값들을 form.save 적용은 안하고 video 에 입력
            
            video.title = video_name                        # video_name 을 title 에 입력
            video.filepath = 'videos/' + video_name         # 'videos/' + video_name 으로 파일 경로 입력
            video.video_file = 'videos/' + video_name
            
            video.save()                                    # DB 적용 video 의 값을

            return redirect('video_detail', fileNo=video.fileNo)    # DB 적용이 완료되면 video_detail 을 불러와, video_detail/fileNo
                                                                    # 으로 redirect
        else:
            print(form.errors)                                      # 유효성 검사 틀리면 프린트
    if request.method == 'POST':                            # form 으로 Method=POST 로 받아와서 작업
        form = VideoForm(request.POST, request.FILES)       # forms.py 에서 작업하기위해 POST 로 요청, FILES 를 불러옴
        if form.is_valid():                                 # form 의 title, video_file 형식으로 들어오는지 유효성 검사

            video_file = request.FILES.get('video_file')    # <label for="video_file">Title:</label>
            video_name = video_file.name                    # <input type="file" class="form-control-file" id="video_file" name="video_file">
                                                            # 에서 받아온 video_file 을 get files 화
            video = form.save(commit=False)                 # file 들어온 값들을 form.save 적용은 안하고 video 에 입력

            video.title = video_name                        # video_name 을 title 에 입력
            video.filepath = 'videos/' + video_name         # 'videos/' + video_name 으로 파일 경로 입력
            video.video_file = 'videos/' + video_name

            video.save()                                    # DB 적용 video 의 값을

            return redirect('video_detail', fileNo=video.fileNo)    # DB 적용이 완료되면 video_detail 을 불러와, video_detail/fileNo
                                                                    # 으로 redirect
        else:
            print(form.errors)                                      # 유효성 검사 틀리면 프린트
    else:
        form = VideoForm()                                          # POST 아니면 화면 다시 띄우기

    return render(request, 'upload_video.html', fileNo=video.fileNo)

def video(request):    
    return render(request, 'upload_video.html')

def video_analyze(request): 
    if request.method == 'POST':
        video_file = request.FILES['video_file']
        upload = default_storage.save(video_file.name,ContentFile(video_file.read()))

        # command = 'python /home/safezone/media/yolov5/detect.py --source /home/safezone/media/' + video_file.name + ' --weights /home/project/best.pt --exist-ok'
        command = 'python D:/Project_by_me/safezone/media/yolov5/detect.py --source D:/Project_by_me/safezone/media/' + video_file.name + ' --weights D:/safezone/best.pt --exist-ok'

        print(command)
        try:
            subprocess.run(command, shell=True, check=True)            
        except subprocess.CalledProcessError as e:
            print(e)

        detect_video_file = '/media/yolov5/runs/detect/exp/'+video_file.name
        # detect_txt_file = '/home/safezone' + detect_video_file.split('.mp4')[0] + '.txt'
        detect_txt_file = 'D:/Project_by_me/safezone/' + detect_video_file.split('.mp4')[0] + '.txt'

        f = open(detect_txt_file,'r')
        text_data = f.read()
        f.close()
        return render(request,'video_analyze.html',{'video_filename':detect_video_file,'text_data':text_data})
    return render(request, 'video_analyze.html')  


def video_detail(request, fileNo):
    video = get_object_or_404(Video, pk=fileNo)
    return render(request, 'video_detail.html', {'video': video})

num = 0
def yolov5_webcam(request):
    global num
    print(num)
    log_entries = LogEntry.objects.filter(execution_num=num)
    log_text = ""
    if log_entries.exists():
        text_space = '\n'
        for entry in log_entries:
            log_text += f"Source: {entry.source} {text_space} Event Type: {entry.event_type} | Event Time: {entry.event_time} {text_space}" 

    else:
        log_text = []
    print(log_text)
    return render(request, 'yolov5_webcam.html', {'log_text': log_text})

def get_log(request):
    global num  # num 변수를 전역 변수로 사용
    num += 1  # num 값 증가
    if request.method == 'GET':
        # exp_dir = '/home/safezone/media/yolov5/runs/detect/'
        exp_dir = 'D:/Project_by_me/safezone/media/yolov5/runs/detect/'

        subdirs = [f.path for f in os.scandir(exp_dir) if f.is_dir() and f.name.startswith('exp') and f.name[3:].isdigit()]

        if subdirs:
            max_exp_dir = max(subdirs, key=lambda x: int(x.split('exp')[-1]))
            detect_txt_file = os.path.join(max_exp_dir, 'detect_log.txt')
            print(max_exp_dir)
            if os.path.exists(detect_txt_file):  # 로그 파일이 존재하는 경우에만 읽어옴
                with open(detect_txt_file, 'r') as f:
                    lines = f.readlines()
                    
                # DB 에 저장할 로그 정보 추출
                log_entries = []
                for line in lines:
                    # 로그 파일에서 필요한 정보 추출
                    log_data = line.strip().split('|')
                    event_time_str = log_data[0].strip()  # 이벤트 시간 문자열
                    event_type = log_data[1].strip()  # 이벤트 유형
                    print(event_type)
                    # 이벤트 시간을 datetime 객체로 변환
                    event_time = datetime.datetime.strptime(event_time_str, '%Y%m%d_%H%M%S')

                    # DB 에 저장할 로그 엔트리 생성
                    log_entry = LogEntry(source='webcam', execution_num=num, event_type=event_type, event_time=event_time)
                    log_entries.append(log_entry)
                print(log_entry)
                # DB 에 일괄 저장
                LogEntry.objects.bulk_create(log_entries)

                log_content = "Log entries saved to the database."
            else:
                log_content = ""  # 로그 파일이 없는 경우 빈 문자열로 설정
        else:
            log_content = "No exp folders found."

        return HttpResponse(log_content)
# Ajax 요청 처리

@csrf_exempt
def run_yolov5_webcam(request):
    if request.method == 'POST':

        #command = '/Users/seoyoobin/Desktop/MLP_AI Engineer Camp/safezone/safezone/safezone_app/yolov5/best.pt'
        # command = 'python /home/safezone/media/yolov5/detect.py --weights /home/project/best.pt --save-txt --save-conf --conf-thres 0.60 --source 0 --alarm SMS'
        command = 'python D:/Project_by_me/safezone/media/yolov5/detect.py --weights D:/safezone/best.pt --save-txt --save-conf --conf-thres 0.60 --source 0 --alarm SMS'


        try:
            subprocess.run(command, shell=True, check=True)
            return HttpResponse("Detection completed successfully.")
        except subprocess.CalledProcessError as e:
            return HttpResponse(f"Error occurred while running detection: {e}")
        # 웹캠 캡처 객체 생성

        cap = cv2.VideoCapture(0)  # 0 은 기본 웹캠을 나타냄
        #count = count * 30
        #start_file_number = count
        #end_file_number = count + 299

        # detect.py 스크립트 실행
        os.system(command)

        # 일정 시간마다 출력 결과 확인
        
            # 특정 클래스 객체 검출 수 확인
        
        
        # detect.py 실행을 중지하는지 확인
        if 'stop_flag' in request.POST and request.POST['stop_flag'] == 'true':
            # 웹캠 캡처 객체 해제
            cap.release()
            return JsonResponse({'message': '감지를 중지했습니다.'})


        # 웹캠 캡처 객체 해제
        cap.release()

        return render(request, 'run_yolov5_webcam.html')

    return JsonResponse({'message': '잘못된 요청입니다.'})


from .CCTV import VideoCamera, gen, video_feed

#def webcam_feed(request):
#    return StreamingHttpResponse(gen(IPWebCam()), content_type='multipart/x-mixed-replace; boundary=frame')


# --------------- 기능 테스트 진행중 -------------------- #
@gzip.gzip_page
def livefe(request):
    try:
        cam = VideoCamera()
        
        return StreamingHttpResponse(gen(cam))
    except:  # This is bad!
        pass

def find_camera(id):
    cameras = ['rtsp://210.99.70.120:1935/live/cctv001.stream',
    'rtsp://210.99.70.120:1935/live/cctv002.stream',
    'rtsp://210.99.70.120:1935/live/cctv003.stream',
    'rtsp://210.99.70.120:1935/live/cctv004.stream',
    'rtsp://210.99.70.120:1935/live/cctv005.stream',
    'rtsp://210.99.70.120:1935/live/cctv006.stream',
    'rtsp://210.99.70.120:1935/live/cctv007.stream',
    'rtsp://210.99.70.120:1935/live/cctv008.stream']
    return cameras[int(id)]


def gen_frames(camera_id):
     
    cam = find_camera(camera_id)
    cap=  cv2.VideoCapture(cam)
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



def video_feed(request, id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingHttpResponse(gen_frames(id),
                    content_type='multipart/x-mixed-replace; boundary=frame')


def index():
    return render('main.html')


