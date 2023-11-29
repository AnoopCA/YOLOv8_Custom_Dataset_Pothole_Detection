import os
import torch
from ultralytics import YOLO
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
video_path = r'D:\ML_Projects\YOLOv8_Custom_Dataset_Pothole_Detection\Pothole.mp4'
video_path_out = '{}_out.mp4'.format(video_path[:-4])

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
# Load a model
model = YOLO(model_path).to(device)
threshold = 0.5

while ret:
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, results.names[int(class_id)].capitalize(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Annotated Frame', frame)
    cv2.waitKey(1)
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
