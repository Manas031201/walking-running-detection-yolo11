# Installing Ultralytics for YOLO11 and the stable legacy version of MediaPipe
!pip install ultralytics mediapipe==0.10.14

import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import IPython.display as display

# ==========================================
# 1. MODEL INITIALIZATION & SETUP
# ==========================================
# We load the pre-trained YOLO11 Pose Estimation model ('n' stands for nano, which is optimized for speed).
# This model uses the COCO dataset standard, identifying 17 specific human body joints (keypoints) 
# out of the box without requiring any custom training.
model = YOLO("yolo11n-pose.pt")

# Provide the path to the uploaded testing video
video_path = '/content/Running.mp4'
cap = cv2.VideoCapture(video_path)

# Variables to store the joint coordinates from the previous frame.
# This allows us to track distance/displacement over time.
prev_keypoints = None

print("Starting Real-Time Tracking Window...")

# ==========================================
# 2. FRAME-BY-FRAME PROCESSING LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the video ends or fails to load

    # Pass the current frame through the YOLO11 model to detect people and their poses.
    # verbose=False prevents the console from getting cluttered with text logs for every frame.
    results = model(frame, verbose=False)

    # Default fallback state for each frame
    status = "Standing"

    # Iterate through all detected objects/people in the frame
    for result in results:
        # Verify that both a bounding box and pose keypoints exist for the detected person
        if result.boxes is not None and result.keypoints is not None:
            
            # Convert bounding boxes and keypoint tensors to NumPy arrays for easier mathematical manipulation
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            # Process each individual person found in the frame
            for i, box in enumerate(boxes):
                # Extract coordinates (top-left x1,y1 and bottom-right x2,y2) to draw a tracking box
                x1, y1, x2, y2 = map(int, box[:4])
                # Draw a distinct red rectangle around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Ensure the model successfully extracted enough keypoints to analyze the lower body
                if len(keypoints) > i and len(keypoints[i]) > 16:
                    
                    # -------------------------------------------------------------
                    # WHY THESE KEYPOINTS WERE CHOSEN:
                    # In the standard COCO pose architecture:
                    # Index 15 = Left Ankle | Index 16 = Right Ankle
                    # Ankles are chosen because they undergo the highest displacement 
                    # (velocity) relative to the ground during human locomotion. 
                    # Torso or head points remain relatively stable, whereas ankles 
                    # dynamically change position sharply when shifting from walking to running.
                    # -------------------------------------------------------------
                    left_ankle = keypoints[i][15]
                    right_ankle = keypoints[i][16]

                    # Only compute velocity if we have a valid previous frame and the current coordinates are non-zero
                    if prev_keypoints is not None and left_ankle[0] > 0 and right_ankle[0] > 0:
                        prev_left = prev_keypoints[0]
                        prev_right = prev_keypoints[1]

                        # Calculate the distance (pixel movement) traveled by each ankle between frames
                        left_dist = np.linalg.norm(left_ankle - prev_left)
                        right_dist = np.linalg.norm(right_ankle - prev_right)
                        
                        # We use the maximum velocity of either ankle to ensure we capture the active swing phase of walking/running
                        movement_speed = max(left_dist, right_dist)

                        # -------------------------------------------------------------
                        # THRESHOLD LOGIC (LOCOMOTION CLASSIFICATION):
                        # These numerical values represent pixel displacement boundaries per frame.
                        # - < 2.5 pixels: Micro-movements or swaying while stationary -> Standing
                        # - 2.5 to 20 pixels: Steady, rhythmic stride advancement -> Walking
                        # - > 20 pixels: High-velocity explosive leg displacement -> Running
                        # The values can be changed to get more accuracy
                        # -------------------------------------------------------------
                        if movement_speed < 2.5:
                            status = "Standing"
                        elif 2.5 <= movement_speed < 20:
                            status = "Walking"
                        else:
                            status = "Running"

                    # Update history: save current ankle coordinates to serve as the baseline for the next frame
                    prev_keypoints = (left_ankle, right_ankle)

                # Render the classification text dynamically right above the individual's bounding box
                cv2.putText(frame, f"Status: {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # ==========================================
    # 3. DISPLAY RESIZING & COLAB RENDERING
    # ==========================================
    # I did this because the video was getting cropped.
    # Get original image dimensions to preserve the aspect ratio and prevent squishing
    target_width = 720
    h, w, _ = frame.shape
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)

    # Downscale the frame so it fits comfortably within the browser window without spilling off-screen
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Force Google Colab to clear the previous output and render the fresh frame in-place,
    # simulating a live, real-time video playback interface.
    display.clear_output(wait=True)
    cv2_imshow(resized_frame)

# Release the video file pointer once complete
cap.release()
print("Video processing complete.")
