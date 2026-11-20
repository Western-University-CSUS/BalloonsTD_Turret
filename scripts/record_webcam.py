import os
import time
from datetime import datetime

import cv2

cap = cv2.VideoCapture(0)
FPS_count = "Hold..."

# Check if camera opened
if not cap.isOpened():
    print("Failure opening camera")
    exit()
else:
    print("Camera opening")

# Set the capture proportions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup FPS counter
window_start_time = time.perf_counter()
frame_counter = 0
target_FPS = 30

# Setup recording
os.makedirs("footage", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out = cv2.VideoWriter(f'footage/testing_{stamp}.mp4', 
    fourcc, target_FPS, (frame_width, frame_height))

# Check that the writer opened
if not out.isOpened():
    print("Failure opening writer")
    exit()
else:
    print("Writer opening")

while True:
    # Frame capture
    ok, frame = cap.read()
    if not ok:
        break

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
    
    # Write the frame to the recording
    out.write(frame)

    # Add remaining UI that wont be recorded
    cv2.putText(frame, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100))
    cv2.imshow("Test", frame)
    
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close program
cap.release()
out.release()
cv2.destroyAllWindows()
