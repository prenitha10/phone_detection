# -*- coding: utf-8 -*-
"""Opencv phone detection

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CuwT4wVgYDFbsGNjh-BawxrJctWrdiQp
"""

!nvidia-smi

# Pip install method (recommended)

!pip install ultralytics==8.0.196

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="x4fOWwCz3shoWEbvyp6V")
project = rf.workspace("yurals-pro").project("phone-using-detection-h0hzo")
version = project.version(7)
dataset = version.download("yolov8")

!pip install albumentations==1.3.0

from ultralytics import YOLO
# Load YOLOv8 model
model = YOLO("yolov8n.pt")
# Train the model on your dataset
model.train(data="/content/Phone-using-detection-7/data.yaml", epochs=50, imgsz=640,device='0')

import shutil
# Path to the best weights from the training session
best_weights_path = "/content/runs/detect/train/weights/best.pt"
# Define your target path to save the model
save_path = "/content/best_yolo_model.pt"
# Copy the model to your desired location
shutil.copy(best_weights_path, save_path)
print(f"Model saved to {save_path}")

from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("/content/best_yolo_model.pt")  # Path to your best weights

# Path to your validation dataset configuration (YAML file)
data_yaml_path = "/content/Phone-using-detection-7/data.yaml"

# Perform validation on the dataset
results = model.val(data=data_yaml_path)

# Print metrics from validation results (like mAP, precision, recall)
print(results)

from ultralytics import YOLO
# Load your custom trained YOLO model from the given path
model = YOLO("/content/best_yolo_model.pt")  # Change to your custom model weights

# Perform detection on a source directory (similar to detect.py)
source_path = '/content/runs/detect/val2'  # Path to your validation images or video
save_path = '/content/runs/detect/yolov8'  # Output directory
img_size = 640  # Image size for detection
device = 'cpu'  # Run on CPU

# Run detection
results = model.predict(source=source_path, imgsz=img_size, device=device, save=True, save_txt=True, name="yolov8")

print(f"Detection completed. Results saved to {save_path}")