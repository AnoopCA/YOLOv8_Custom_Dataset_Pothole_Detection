import torch
torch.cuda.set_device(0)
from ultralytics import YOLO

# Load a model
model = YOLO(r'D:\ML_Projects\YOLOV8_Custom_Dataset\Object_Detection_Pothole\yolov8x.pt') # build a new model from scratch

# train the model
if __name__ == '__main__':
    results = model.train(data="data.yaml", epochs=10)
