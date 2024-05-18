import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

yolo_model = YOLO("D:/Workspace/Research/YoloV8/runs/detect/train/weights/best.pt")

def detect():
    video_path = "pothole_video.mp4"
    cap = cv2.VideoCapture(video_path)
            
    while cap.isOpened(): 
        ret, frame = cap.read()
        # cv2.imshow('Input', frame)
        results = yolo_model.predict(frame)
                # or
        # results = yolo_model.predict(frame,show=True)    # To enable default interpretation

        # print(results)
        for r in results:
                
                annotator = Annotator(frame)
                
                boxes = r.boxes
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, yolo_model.names[int(c)], color=(0, 255, 0))
                
        img = annotator.result()  
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

cv2.destroyAllWindows()

detect()
