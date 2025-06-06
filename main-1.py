import cv2
import time
import os
from eye_movement import process_eye_movement
from head_pose import process_head_pose

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a log directory for screenshots
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Calibration for head pose
calibrated_angles = (0, 0, 0)
start_time = time.time()

# Timers
head_misalignment_start_time = None
eye_misalignment_start_time = None

# Default states
head_direction = "Looking at Screen"
gaze_direction = "Looking at Screen"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process eye movement
    frame, gaze_direction = process_eye_movement(frame)
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calibration
    if time.time() - start_time <= 5:
        cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        _, temp_angles = process_head_pose(frame, None)
        if isinstance(temp_angles, tuple) and len(temp_angles) == 3:
            calibrated_angles = temp_angles
    else:
        frame, head_direction = process_head_pose(frame, calibrated_angles)

        # Clamp head_direction to valid states
        valid_head_directions = {"Looking at Screen", "Looking Left", "Looking Right"}
        if head_direction not in valid_head_directions:
            head_direction = "Looking at Screen"

        cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Head misalignment screenshot logic
    if head_direction in ["Looking Left", "Looking Right"]:
        if head_misalignment_start_time is None:
            head_misalignment_start_time = time.time()
        elif time.time() - head_misalignment_start_time >= 3:
            print(f"[DEBUG] Head Direction at Save Time: {head_direction}")
            filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            head_misalignment_start_time = None
    else:
        head_misalignment_start_time = None

    # Eye misalignment logic
    if gaze_direction != "Looking at Screen":
        if eye_misalignment_start_time is None:
            eye_misalignment_start_time = time.time()
        elif time.time() - eye_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            eye_misalignment_start_time = None
    else:
        eye_misalignment_start_time = None

    # Display frame
    cv2.imshow("Cheating Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
