import os
import torch
from ultralytics import YOLO
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir = r'D:\ML_Projects\YOLOv8_Custom_Dataset_Pothole_Detection\test\images'
model_path = r'D:\ML_Projects\YOLOv8_Custom_Dataset_Pothole_Detection\runs\detect\train\weights\last.pt'

# Load a model
model = YOLO(model_path).to(device)  # load a custom model
threshold = 0.5

# Iterate through all images in the folder
for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)
    # Read the image
    image = cv2.imread(image_path)
    # Perform inference on the image
    results = model(image)[0]

    # Draw bounding boxes and labels for detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, results.names[int(class_id)].capitalize(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save the modified image with bounding boxes
    base_name, extension = os.path.splitext(image_filename)
    output_image_path = os.path.join(image_dir, base_name +'_output' + extension)
    cv2.imwrite(output_image_path, image)
    # Display the inferred image
    cv2.imshow('Pothole Detection', image)
    cv2.waitKey(1)  # Wait for 1 millisecond to prevent overloading the window
