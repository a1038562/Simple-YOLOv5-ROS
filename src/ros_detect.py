#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from yolov5_ROS.msg import Yolo_Objects, Objects

import os
import sys
from pathlib import Path
import numpy as np

import torch

# yolov5 submodule을 path에 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT를 PATH에 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 관련 path

from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

interval = 1 # frame

class YoloV5_ROS():
    def __init__(self):
        rospy.Subscriber(source, CompressedImage, self.Callback)
        self.pub = rospy.Publisher("yolov5_pub", data_class=Yolo_Objects, queue_size=10)
        self.weights = rospy.get_param("~weights")  # model path
        self.data = rospy.get_param("~data")  # dataset.yaml path
        self.device = torch.device(rospy.get_param("~device"))  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz = rospy.get_param("~imgsz")

        # Load model
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = (imgsz, imgsz)

        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 10  # maximum detections per image
        self.classes = None  # filter by class
        self.agnostic_nms = False  # class-agnostic NMS
        self.line_thickness = 3  # bounding box thickness (pixels)

        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
    
    def Callback(self, data): 
        global interval
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        msg = Yolo_Objects()

        cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        interval += 1

        if interval % 1 == 0:
            im0s = img

            # Run inference
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup

            im = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0s, line_width=self.line_thickness, example=str(self.names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # box coordinate
                    x1, y1, x2, y2 = map(int, xyxy)
                    msg.yolo_objects.append(Objects(c, x1, x2, y1, y2))

            cv2.imshow('result', im0s)
            cv2.waitKey(1)  # 1 millisecond

        self.pub.publish(msg) # msg publish
            
def run():
    global source
    rospy.init_node("yolov5_ROS")
    source = rospy.get_param("~source")
    detect = YoloV5_ROS()
    rospy.spin()

if __name__ == '__main__':
    run()
