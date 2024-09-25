# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import cv2
from ultralytics import YOLO


# Load the trained YOLOv8 model
model = YOLO("best_yolo_model_phone.pt")  # Replace with the path to your saved model

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or 1, 2, etc. for external cameras

# Check if the webcam is opened correctly

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process frames from the webcam
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO model prediction on the frame
    results = model(frame)

    # Extract predictions and render them on the frame
    annotated_frame = results[0].plot()

    # Display the frame with predictions
    cv2.imshow("YOLOv8 Webcam Prediction", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
