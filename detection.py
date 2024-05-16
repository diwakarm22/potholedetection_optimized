import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO


#yolo_model = torch.hub.load('ultralytics', 'custom', path='runs/detect/train9/weights/last.pt', force_reload=True)

yolo_model = YOLO("runs/detect/train9/weights/best.pt")


def detect():
    video_path = "pothole_video.mp4"
    cap = cv2.VideoCapture(video_path)
            
    while cap.isOpened(): 
        ret, frame = cap.read()
        
        results = yolo_model.predict(frame)
        #print(results)

        '''
        cv2.imshow('YOLO', np.squeeze(results.render()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        '''
        
        # Iterate over each result
        for idx, result in enumerate(results):
            # Get the image and detections for the current result
            image = frame.copy()
            
            # Ensure that there are detections in the result
            if len(result) > 0:
                # Filter out detections with confidence less than 0.3
                det = result[result[:, 4] > 0.3]
                
                # Visualize the detections on the image
                for *xyxy, conf, cls in det:
                    cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                    cv2.putText(image, f'{cls}: {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
            cv2.imshow('YOLO', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        
detect()

'''
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path

# Load the YOLOv8 model
model = YOLO("runs/detect/train9/weights/best.pt")

# Path to Video
video_path = "pothole_video.mp4"

if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path {video_path} does not exist.")

names = model.model.names
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()
        annotator = Annotator(frame, line_width=2, example=str(names))

        # Iterate through the results
        for box, cls, conf in zip(boxes, classes, confidences):
            annotator.box_label(box, names[int(cls)], (255, 42, 4))

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
'''