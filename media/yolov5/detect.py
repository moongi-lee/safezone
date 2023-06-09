# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import requests
import hashlib
import hmac
import base64
import time, json
from collections import deque

import torch

# 문자 알람 관련 정보
SMS_timestamp = int(time.time() * 1000)
SMS_timestamp = str(SMS_timestamp)
SMS_access_key = "oDgbh0ZJsFZJU4B02NSP"
SMS_url = "https://sens.apigw.ntruss.com"
SMS_uri = "/sms/v2/services/ncp:sms:kr:308300814920:safezone/messages"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import datetime
import uuid
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# 문자 알람 관련 signature 만들기
def make_signature():
    secret_key = "kcdWYdxcH1Y0FwwzIl980FLz6ZDTV2DH15fivP2Z"
    secret_key = bytes(secret_key, 'UTF-8')
    method = "POST"
    message = method + " " + SMS_uri + "\n" + SMS_timestamp + "\n" + SMS_access_key
    message = bytes(message, 'UTF-8')
    signinkey_val = hmac.new(secret_key, message, digestmod=hashlib.sha256).digest()
    return base64.b64encode(signinkey_val).decode('UTF-8')

# 문자 알람 관련 헤더
header = {
    'Content-Type' : 'application/json; charset=utf-8',
    'x-ncp-apigw-timestamp' : SMS_timestamp,
    'x-ncp-iam-access-key' : SMS_access_key,
    'x-ncp-apigw-signature-v2' : make_signature()
}

# 문자 알람 관련 데이터
SMS_data = {
    "type" : "SMS",
    "from" : "01097069798",
    "subject" : "발신번호테스트",
    "content" : "문자내용테스트",
    "messages" :[{"to" : "01083345690"}]    
}

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        alarm='', # alarm sms/email/discord
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    count = {'0' : 0, '1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0} # detecting count     
    frame = 1 # 프레임별 클래스 검색하기 위한 초기화
    result_frame_string = ''
    detect_deque = deque(maxlen=300)
    prev_frames = deque(maxlen=300)
    for path, im, im0s, vid_cap, s in dataset:
               
        frame_string = str(frame) + ":" # 프레임별 클래스 검색하기 위한 문자열 초기화
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
              
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'log' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results                                
                for c in det[:, 5].unique():                                       
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string                    
                    # count[str(int(c))] += 1 # 검출된 클래스 개수 늘리기
                    frame_string += (str(int(c)) + " ") # 검출된 클래스 추가하기 

                log_file = str(save_dir / 'labels' / p.stem) + 'log_file.txt'
                # Write results
                object_counts = []
                for *xyxy, conf, cls in reversed(det):
                    object_counts.append(cls)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(log_file, 'a') as f:  # 프레임 로그를 파일에 추가
                            for frame_num, cls in enumerate(object_counts, start=1):
                                f.write(f'{frame_num}: {int(cls)}\n')
                    #with open(f'{txt_path}.txt', 'a') as f:
                    #    f.write(('%g ' * len(line)).rstrip() % line + '\n')    
                    
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
           
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            
            prev_frames.append(im0)  # im0를 prev_frames에 추가
                        
            #Save results (image with detections)
            if alarm == '':
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'x264'), fps, (w, h))
                        vid_writer[i].write(im0) 
                
            
            # deque 만들기  
            if alarm == 'SMS':         
                if len(detect_deque) < 300: # deque 사이즈가 300이 안되면 
                    detect_deque.append(list(map(int, list(det[:, 5].unique())))) # 무조건 append            
                else: # deque 사이즈가 70이 넘으면                 
                    detect_deque.popleft() # 선입선출
                    detect_deque.append(list(map(int, list(det[:, 5].unique())))) # 그다음 append
            
        # video upload 시 프레임별 클래스 저장
        #     frame += 1 
        # save_path_split = save_path.split('.mp4')        
        # with open(f'{save_path_split[0]}.txt', 'a') as f:
        #     f.write(f'{frame_string}' + '\n')
        
        # frame_string_split = frame_string.split(':')[1].split(' ')
        # for count_index in count:                
        #     if count_index in frame_string_split:
        #         count[count_index] += 1
        #     # else: # 연속해서 검출이 안될경우 카운트 초기화
        #     #     count[count_index] = 0 
        
        if alarm == 'SMS':
            if len(detect_deque) == 300: # deque 사이즈가 300이면   
                for detect_deque_val in detect_deque: # deque 탐색 시작
                    if detect_deque_val: # deque에 값이 있을때
                        for val in detect_deque_val: # deque의 값이 리스트이기때문에 for문 추가
                            count[str(val)] += 1 # deque의 값에 해당하는 클래스를 count +1 
        
            target_class_ids = [1, 3, 5]
            for count_index in count: # dictionary 탐색 시작        
                if count[count_index] >= 240:  # dictionary 값중 240이 넘는 값이 있다면
                    i += 1
                    # sms 발송하기
                    print("SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기SMS 발송하기")   
                    # SMS_data['content'] = count_index + "번 클래스가 70프레임 발견되었습니다."              
                    # res = requests.post(SMS_url+SMS_uri,headers=header,data=json.dumps(SMS_data))
                    # print(res.json()) 
                    
                    if int(c) in target_class_ids:          # target_class_ids : 1, 3, 5 shoes_X, helmet_X, belt_X 세가지 검출
                        class_name = names[int(c)]          # class_name에 이름 넣어서 표시
                        class_count = (det[:, 5] == c).sum()# 검출을 했을시 동작 시킬 파트
                        for j in range(class_count):
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # 현재 시간 타임 스탬프
                            unique_id = str(uuid.uuid4().hex[:6])                           # 파일 고유 번호 
                            filename = f"{save_dir}/detected_{class_name}_{i}_{timestamp}_{unique_id}.jpg" # 파일 네임 폼
                            print(f"detected_{class_name}_{i}_{timestamp}_{unique_id}.jpg 생성")
                            cv2.imwrite(filename, im0)                                      # EX : detected_safety_helmet_X_1_현재날짜_현재시각_고유번호.jpg
                    
                    # 비디오 저장 부분 진행중
                    videoname = None  # videoname 변수 초기화
                    videoname = f"{save_dir}/detected_{class_name}_{i}_{timestamp}_{unique_id}.mp4" #exp folder 하위에 ex :detected_safety_helmet_X_1_현재날짜_현재시각_고유번호.mp4 파일 생성
                    
                    # 이전 비디오 writer 해제
                    for writer in vid_writer:   #이전에 사용했던 vid_writer를 해제 하고 release()를 새로 함
                        if writer is not None:
                            writer.release()
                    
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    
                    # 비디오 writer 초기화
                    vid_writer.append(cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)))
                    
                    # 이전 프레임들을 비디오로 저장
                    for prev_frame in prev_frames:
                        frame = prev_frame.copy()
                        if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                            vid_writer[-1].write(frame)
                        else:
                            print(f"Invalid frame format: {type(frame)}, skipping...")
                    
                    print(f"detected_{class_name}_{i}_{timestamp}_{unique_id} 생성")
                    
                    log_text = '' # log_text 초기화
                    logs = f"{save_dir}/detect_log.txt"
                    if int(c) == 1:
                        log_text = f'{timestamp} | Detected Class : Not worn shoes \n'
                    elif int(c) == 3:
                        log_text = f'{timestamp} | Detected Class : Not worn Belt \n'
                    elif int(c) == 5:
                        log_text = f'{timestamp} | Detected Class : Not worn Helmet \n'
                        
                    with open(logs, 'a', encoding='utf-8') as f:
                        f.write(log_text)
                    print(f"detected_log 생성 및 쓰기")
                    
                    
                    ## 비디오 저장 부분 진행중
                    #videoname = None  # videoname 변수 초기화
                    #videoname = f"{save_dir}/detected_{class_name}_{i}_{timestamp}_{unique_id}.mp4"
                    ## vid_writer 리스트 초기화
                    #vid_writer = [None] * len(vid_path)
                    #print(vid_writer, videoname)
                    ## videoname이 비어 있지 않을 때만 경로 설정
                    #if videoname is not None:
                    #    save_path = str(Path(videoname).with_suffix('.mp4'))
                    #    print(save_path)
                    #if len(vid_path) <= i:
                    #    vid_path = save_path  # Add new video path to the list
                    #    vid_writer.append(cv2.VideoWriter())  # Add a new element to vid_writer list
                    #    print(vid_writer, vid_path, save_path)
                    ## 새로운 비디오를 위한 파일 이름 생성
                    #
                    #
            #
    #
                    #if vid_writer is not None:
                    #    vid_writer.release()  # release previous video writer
                    #    print('release previous video writer')
    #
                    #if vid_cap:  # video
                    #    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #else:  # stream
                    #    fps, w, h = 30, im0.shape[1], im0.shape[0]
    #
                    #
                    #vid_writer = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    ## 이전 프레임들을 비디오로 저장
                    #for prev_frame in prev_frames:
                    #    frame = prev_frame.copy()
                    #    vid_writer.write(frame)
                    #print(f"detected_{class_name}_{i}_{timestamp}_{unique_id} 생성")


                    # 현재 프레임을 비디오로 저장
                    #vid_writer[i].write(im0)
                    #print(f"detected_{class_name}_{i}_{timestamp}_{unique_id} 생성")
                        
                    for remove_deque in detect_deque:
                        if remove_deque:  
                            remove_deque.clear()
                count[count_index] = 0 # dictionary 초기화

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    if alarm == '':
            save_path_split = save_path.split('.mp4')        
            with open(f'{save_path_split[0]}.txt', 'w') as f:
                f.write(f'{result_frame_string}' + '\n')
   
    # detecting count check
    # save_path_split = save_path.split('.mp4')
    # for count_check in count:    #     
    #     with open(f'{save_path_split[0]}.txt', 'a') as f:
    #         f.write(f'{count_check} {count[count_check]}' + '\n')

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--alarm', default='', help='sms/discord')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
