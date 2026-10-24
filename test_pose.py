import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_path = 'pose_landmarker_lite.task'

    # Configure Pose Landmarker for live video stream
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    print("[INFO] Modern MediaPipe Tasks API active. Press 'q' to exit.")

    prev_time = time.time()
    start_time_ms = int(time.time() * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000) - start_time_ms

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        h, w, _ = frame.shape

        # Render landmarks if detected
        if detection_result.pose_landmarks:
            for landmark_list in detection_result.pose_landmarks:
                for lm in landmark_list:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Draw green landmark circles on joints
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        cv2.putText(
            frame, 
            f"Tasks API FPS: {fps:.1f}", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )

        cv2.imshow("Kinematics - MediaPipe Tasks API Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()