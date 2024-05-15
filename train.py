from ultralytics import YOLO
import torch

#model load
model = YOLO("yolov8n.yaml")

# Start training
model.train(data="config.yaml", epochs=1)

# Save trained model
model.save("Path_hole.pt")