
from django.shortcuts import render
import cv2
import numpy as np
from django.http import StreamingHttpResponse
import threading

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('rtsp://210.99.70.120:1935/live/cctv001.stream')
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
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



def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingHttpResponse(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def index():
    return render('main.html')

