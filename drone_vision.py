import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Set video source (webcam)
cap = cv2.VideoCapture(0)

# Define the class you want to detect
classes = ['Drone']

while True:
    # Read frame from video source
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for YOLOv5
    img = Image.fromarray(frame[..., ::-1])

    # Run inference
    results = model(img, size=640)

    # Process detections
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and classes[int(cls)] in classes:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display confidence score
            text_conf = "{:.2f}%".format(conf * 100)
            cv2.putText(frame, text_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Compute and display centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Centroid: ({cx}, {cy})", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show output
    cv2.imshow('Drone Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
