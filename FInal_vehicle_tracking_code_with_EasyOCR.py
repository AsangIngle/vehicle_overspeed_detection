import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import time
from datetime import datetime
import os
from easyocr import Reader
from twilio.rest import Client

# Load video
cap = cv2.VideoCapture("C:/Users/HP/Downloads/2103099-uhd_3840_2160_30fps.mp4")

# YOLOv8 model
model = YOLO("yolov8n.pt")

# Load and prepare mask
mask = cv2.imread('C:/Users/HP/Downloads/r.png')

# Tracker setup
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Twilio setup
account_sid = 'AC03746e4007f15693e95fbb2f78bd0fc3'
auth_token = '5dc18919b65651eb677ce61c170e4f0e'
client = Client(account_sid, auth_token)

# Output folder for saved images
path_of_folder = "C:/Users/HP/Desktop/vscode_project"
os.makedirs(path_of_folder, exist_ok=True)

# Speed limit (set low for testing)
speed_limit = 5  # km/h

# OCR reader
reader = Reader(['en'])

# Scale factor (adjust for speed/efficiency)
scale_factor = 0.5

# Get video frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame_width", frame_width)
print("frame_height", frame_height)

# Resize mask to match scaled frame
mask = cv2.resize(mask, (int(frame_width * scale_factor), int(frame_height * scale_factor)))

# Vehicle tracking dictionary
vehicle_times = {}

while True:
    success, img = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)
    img_resized = cv2.resize(img, (new_width, new_height))

    # Apply resized mask
    imgRegion = cv2.bitwise_and(img_resized, mask)

    # Object detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    # Draw detection lines
    line_y1 = int(new_height / 2 - 40)
    line_y2 = int(new_height / 2 + 35)
    cv2.line(img_resized, (0, line_y1), (new_width, line_y1), (200, 100, 255), 5)
    cv2.line(img_resized, (0, line_y2), (new_width, line_y2), (0, 0, 255), 5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.3 and cls in [2, 3, 5, 7]:  # Cars, trucks, buses
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                cvzone.cornerRect(img_resized, (x1, y1, x2 - x1, y2 - y1), l=9, rt=4, colorR=(255, 0, 255))
                cvzone.putTextRect(img_resized, f'{conf} {cls}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Update tracker
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cvzone.cornerRect(img_resized, (x1, y1, x2 - x1, y2 - y1), l=9, rt=4, colorR=(0, 255, 0))
        cvzone.putTextRect(img_resized, f'ID: {int(id)}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
        cv2.circle(img_resized, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Track first line
        if line_y1 - 20 < cy < line_y1 + 20:
            if id not in vehicle_times:
                vehicle_times[id] = {'start_time': time.time(), 'end_time': None}
            cv2.line(img_resized, (0, line_y1), (new_width, line_y1), (0, 255, 0), 5)

        # Track second line
        if line_y2 - 20 < cy < line_y2 + 20 and vehicle_times.get(id, {}).get('start_time') is not None:
            vehicle_times[id]['end_time'] = time.time()

            start_time = vehicle_times[id]['start_time']
            end_time = vehicle_times[id]['end_time']
            if start_time and end_time:
                elapsed_time = end_time - start_time
                distance = 10  # meters between lines
                speed = distance / elapsed_time * 3.6  # m/s to km/h

                print(f"Vehicle ID {id} Speed: {speed:.2f} km/h")
                if speed > speed_limit:
                    photo_name = f"overspeed_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(path_of_folder, photo_name), img_resized)
                    print(f"Vehicle ID {id} exceeded speed limit! Speed: {speed:.2f} km/h")

                    plate_img = img_resized[y1:y2, x1:x2]
                    ocr_result = reader.readtext(plate_img)
                    plate_number = ''.join([text[1] for text in ocr_result])

                    print(f"Detected Plate Number: {plate_number}")

                    try:
                        message = client.messages.create(
                            body=f"Alert! Vehicle with plate {plate_number} overspeeding at {speed:.2f} km/h.",
                            from_='+12314488432',
                            to='+919359645981'
                        )
                        print(f"SMS sent: {message.sid}")
                    except Exception as e:
                        print(f"Failed to send SMS: {e}")
                else:
                    print(f"Vehicle ID {id} is within speed limit.")

    # Display vehicle count
    cvzone.putTextRect(img_resized, f'Count: {len(vehicle_times)}', (50, 50), scale=1, thickness=2)

    cv2.imshow("Video", img_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
