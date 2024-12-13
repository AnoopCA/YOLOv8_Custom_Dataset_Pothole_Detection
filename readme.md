# YOLOv8 Custom Dataset Pothole Detection

## Description

This project implements a custom pothole detection system using YOLOv8, a state-of-the-art object detection algorithm. The system is trained on a custom dataset of potholes and can detect potholes in images and videos. The project includes scripts for training the model, predicting potholes in images, and running inference on video files. The primary purpose of this project is to automate the detection of potholes from visual data, which could be useful for road maintenance and safety monitoring.

## Features

- **Custom Model Training**: Train a YOLOv8 model on a custom pothole detection dataset.
- **Pothole Detection in Images**: Perform detection on individual images and highlight potholes with bounding boxes.
- **Pothole Detection in Videos**: Process videos frame by frame, detect potholes, and output a video with marked potholes.
- **Real-time Inference**: The model runs inference on images and videos in real-time using GPU acceleration.
- **Model Export**: Save and load the trained YOLOv8 model for further use.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**:
  - `ultralytics` (YOLOv8)
  - `torch` (PyTorch)
  - `opencv-python` (OpenCV)
- **Hardware**: GPU (CUDA-enabled device for training and inference)

## Data

This project uses a custom pothole detection dataset, which can be used to train the YOLOv8 model. The dataset consists of images labeled with bounding boxes indicating potholes.

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AnoopCA/YOLOv8_Custom_Dataset_Pothole_Detection.git
    cd YOLOv8-Pothole-Detection
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install YOLOv8 and other dependencies**:
    ```bash
    pip install ultralytics opencv-python torch
    ```

## Usage

### Training the Model

To train the YOLOv8 model on your custom pothole detection dataset, use the `train.py` script. You need to modify the `data.yaml` file to point to your dataset configuration. Run the following command to start training:

```bash
python train.py
