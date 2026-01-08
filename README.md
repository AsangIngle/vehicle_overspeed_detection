# Vehicle Overspeed Detection System

This project is a **Vehicle Overspeed Detection and Number Plate Recognition System** built using **YOLOv8, SORT Tracker, EasyOCR**. It detects vehicles in video footage, calculates their speed, identifies overspeeding vehicles, and recognizes license plates. Alerts are sent via **Twilio SMS**.  

---

## Features
1. Detects vehicles (cars, buses, trucks, etc.) in video streams using **YOLOv8**.
2. Tracks vehicles across frames using **SORT Tracker**.
3. Calculates speed of vehicles between two lines drawn in the video.
4. Captures overspeeding vehicles and saves images automatically.
5. Performs **license plate recognition** using **EasyOCR** and a trained **EMNIST CNN model**.
6. Sends **SMS alerts** for overspeeding vehicles via **Twilio**.
7. Displays vehicle count and bounding boxes with speed info in real-time.

---

## Architecture

1. **Video Input:** Load video or camera feed.
2. **Preprocessing:** Resize frames, apply region masks.
3. **Detection:** YOLOv8 detects vehicles in each frame.
4. **Tracking:** SORT tracker maintains unique IDs for vehicles.
5. **Speed Calculation:** Time between two virtual lines determines speed.
6. **Overspeed Capture:** Capture frame, segment characters, and predict plate numbers.
7. **SMS Alerts:** Twilio sends alerts for overspeeding vehicles.

---

## Requirements

- Python 3.10+
- Libraries:
  - `opencv-python`
  - `cvzone`
  - `numpy`
  - `ultralytics`
  - `sort`
  - `easyocr`
  - `keras`
  - `twilio`
  - `matplotlib`
  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/vehicle-overspeed-detection.git
cd vehicle-overspeed-detection

# Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


Setup

1.Video & Mask

  Place your video in the project folder or update the path in the script.

  Provide a mask image to focus detection region (mask.png).

2.YOLOv8 Model

  Download yolov8n.pt weights from Ultralytics YOLO

3.Twilio API

  Set your Twilio Account SID and Auth Token in the script.

4.EMNIST Model
   Used for license plate character recognition.

Usage
python Final_vehicle_tracking_code_with_EasyOCR.py

1.The video will be processed frame by frame.
2.Vehicles crossing the virtual lines will be tracked.
3.Speed of vehicles will be calculated in km/h.
4.Overspeeding vehicles will be captured, plates recognized, and SMS alerts sent.
5.Real-time vehicle count and bounding boxes will be displayed.

Parameters

speed_limit: Set the maximum allowed speed in km/h.

scale_factor: Resize video frames for faster processing.

distance: Distance between two virtual lines in meters.

path_of_folder: Folder to save images of overspeeding vehicles.

File Structure
vehicle_overspeed_detection/
├── Final_vehicle_tracking_code_with_EasyOCR.py
├── Deep_Learning_for_Traffic_Speed_Monitoring.ipynb
├── mask.png
└── README.md

Future Improvements

Integrate real-time deployment with live camera feeds.

Add automatic number plate formatting for multiple regions.

Deploy MLOps pipeline for continuous training and model updates.

Add notification dashboard instead of only SMS alerts.
