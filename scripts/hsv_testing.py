import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
FPS_count = "Hold..."

# Check if camera opened
if not cap.isOpened():
    print("Failure opening camera")
    exit()
else:
    print("Camera opening")

# Set the capture proportions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup HSV bounds for cyan colour
lower_bound = np.array([85, 130, 130])
upper_bound = np.array([100, 255, 255])

# Setup FPS counter
window_start_time = time.perf_counter()
frame_counter = 0
target_FPS = 30

while True:
    # Frame capture
    ok, frame = cap.read()
    if not ok:
        break

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)  # The white / black mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)  # Extract contours from mask

    # Find the largest contour by area
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)  # Extract the rectangle contour
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    blended_mask = cv2.bitwise_and(frame, frame, mask=mask)

    # FPS counter setup
    curr_frame = time.perf_counter()
    frame_counter += 1
    if frame_counter % 30 == 0:
        FPS_count = round(frame_counter / (curr_frame - window_start_time))
        frame_counter = 0
        window_start_time = curr_frame

    # Display the window UI thats recorded
    display_position = frame_width - 80, frame_height - 40
    cv2.putText(frame, str(FPS_count), display_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0))

    # Add remaining UI that wont be recorded
    cv2.putText(frame, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100))
    cv2.imshow("BGR", frame)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Mask", mask)
    cv2.imshow("Blend", blended_mask)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close program
cap.release()
cv2.destroyAllWindows()
