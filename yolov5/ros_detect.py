#! /usr/bin/env python3


import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from rosa_msgs.msg import Detections, Detection
IMAGE_WIDTH=1241
IMAGE_HEIGHT=376
from std_msgs.msg import Int8
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')




import os
import time
import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier
from utils.plots import colors, plot_one_box
from matplotlib import pyplot as plt

ros_image=0
grab_toggle=0

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def loadimg(img):  # 接受opencv图片
    img_size=640
    cap=None
    path=None
    img0 = img
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return path, img, img0, cap
# src=cv2.imread('biye.jpg')
def detect(img):

    time1 = time.time()

    global ros_image
    cudnn.benchmark = True
    dataset = loadimg(img)
    # print(dataset[3])
    #plt.imshow(dataset[2][:, :, ::-1])
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    #colors=[[0,255,0]]
    augment = 'store_true'
    conf_thres = 0.4
    iou_thres = 0.25
    classes = None #(0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25)
    agnostic_nms = 'store_true'
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    path = dataset[0]
    img = dataset[1]
    im0s = dataset[2]
    vid_cap = dataset[3]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    time2 = time.time()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    view_img = 1
    save_txt = 1
    save_conf = 'store_true'
    time3 = time.time()


     # Process predictions
    for i, det in enumerate(pred):  # detections per image

        detection_list = []
 
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy()   # for save_crop
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -2].unique():
                n = (det[:, -2] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    

            # Write results
            for *xyxy, conf, cls , angle  in reversed(det):
                view_img = True
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    hide_labels = False
                    hide_conf = False
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, angle, im0, label=label, color=colors(c, True), line_thickness=2)
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                    det_msg = Detection()
                    det_msg.letter = str(names[c])
                    det_msg.conf = conf
                    det_msg.pose.x = int(c1[0] + (c2[0]-c1[0])*0.5)
                    det_msg.pose.y = int(c1[1] + (c2[1]-c1[1])*0.5)
                    det_msg.pose.theta = 0
                    detection_list.append(det_msg)


                    global grab_toggle
                    if(c2[1] > 700 and grab_toggle == 0):
                        #counter = counter + 1
                        #if(counter > 20):
                        grab_toggle = 1
                        publish_grab(1)
                    else:
                        grab_toggle = 0

        detections_msg = Detections()
        detections_msg.header.frame_id = ""
        detections_msg.header.stamp = rospy.get_rostime()
        detections_msg.detections = detection_list
        detect_pub.publish(detections_msg)

            

        # cv2.circle(im0, (960, 700), radius=1, color=(0, 0, 255), thickness=-1)
        # Stream results
        if view_img:
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
            cv2.resizeWindow(str(p), 1920, 1080)
            im_out = im0[:,:,[2,1,0]]
            cv2.imshow(str(p), im_out)
            publish_image(im_out)
            cv2.waitKey(1)  # 1 millisecond



def image_callback_1(image):
    global ros_image
    ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    with torch.no_grad():
        detect(ros_image)

def publish_image(imgdata):
    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1241*3
    image_pub.publish(image_temp)


def publish_grab(x):
    Int8 = 0
    if(x == True):
        x = 1
    grab_pub.publish(x)
    


if __name__ == '__main__':
    set_logging()
    device = ''
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = os.getcwd() + '/src/cube_detector/' + 'yolov5/m640_rotated.pt'
    imgsz = 640
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    rospy.init_node('cube_detector')
    image_topic_1 = "/usb_cam/image_raw"
    rospy.Subscriber(image_topic_1, Image, image_callback_1, queue_size=1, buff_size=52428800)
    image_pub = rospy.Publisher('/WS1/image', Image, queue_size=1)
    grab_pub = rospy.Publisher('/WS1/cube_hand', Int8, queue_size=1)
    detect_pub = rospy.Publisher('/WS1/detections', Detections, queue_size=1)
    #rospy.init_node("yolo_result_out_node", anonymous=True)
    

    rospy.spin()
