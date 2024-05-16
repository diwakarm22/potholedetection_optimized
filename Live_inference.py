from ultralytics import YOLO
from PIL import Image

model = YOLO("D:/Workspace/Research/YoloV8/runs/detect/train/weights/best.pt")

result = model.predict(source="0",show=True)