import torch
torch.cuda.set_device(0)
from ultralytics import YOLO

# Load a model
model = YOLO(r'D:\ML_Projects\YOLOv8_Custom_Dataset_Pothole_Detection\yolov8m.pt')

# train the model
if __name__ == '__main__':
    results = model.train(data="data.yaml", epochs=100)
