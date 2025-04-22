import cv2
import time
import os
from eye_movement import process_eye_movement
from head_pose import process_head_pose

# Set the path to your video file
video_path = "input_video.mp4"  # Replace with your actual path

# Open video file
cap = cv2.VideoCapture(video_path)

# Create a log directory for screenshots
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

fps = cap.get(cv2.CAP_PROP_FPS)

# Timers and tracking
misalignment_start_time = None
saved_screenshots = []
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    current_time = frame_number / fps

    # Process directions
    frame, gaze_direction = process_eye_movement(frame)
    frame, head_direction = process_head_pose(frame, None)

    # Display on frame
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Detect combined misalignment (both not looking at screen)
    if head_direction != "Looking at Screen" and gaze_direction != "Looking at Screen":
        if misalignment_start_time is None:
            misalignment_start_time = current_time
        elif current_time - misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"misaligned_{int(current_time)}.png")
            cv2.imwrite(filename, frame)
            saved_screenshots.append(filename)
            print(f"Screenshot saved: {filename}")
            misalignment_start_time = None  # Reset after saving
    else:
        misalignment_start_time = None  # Reset if alignment is restored

    # Show video frame
    cv2.imshow("Cheating Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Display screenshots and print inference
for screenshot in saved_screenshots:
    img = cv2.imread(screenshot)
    cv2.imshow("Detected Misalignment", img)
    print(f"Showing: {screenshot}")
    cv2.waitKey(2000)  # Show each for 2 seconds
cv2.destroyAllWindows()

if saved_screenshots:
    print("Inference: Potential cheating behavior detected.")
else:
    print("Inference: No cheating behavior detected.")
