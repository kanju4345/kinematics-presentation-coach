import cv2
import json
import math
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_calibration(profile_path="calibration_profile.json"):
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"[ERROR] '{profile_path}' not found. Run calibrate.py first.")
    with open(profile_path, "r") as f:
        return json.load(f)

def compute_cervical_angle(p_ear_mid, p_shoulder_mid):
    vx = p_ear_mid[0] - p_shoulder_mid[0]
    vy = p_ear_mid[1] - p_shoulder_mid[1]
    norm_v = math.sqrt(vx**2 + vy**2)
    if norm_v == 0:
        return 0.0
    cos_theta = (-vy) / norm_v
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

def main():
    profile = load_calibration()
    base_angle = profile["baseline_cervical_angle_deg"]
    base_sh_width = profile["baseline_shoulder_width"]

    # Thresholds (relative deviations)
    SLOUCH_ANGLE_THRESHOLD = 8.0  # degrees deviation from baseline
    SHOULDER_TILT_THRESHOLD = 0.05  # normalized ratio threshold

    model_path = 'pose_landmarker_lite.task'
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

    start_time_ms = int(time.time() * 1000)

    print("\n[INFO] Running Kinematic Inference Engine. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000) - start_time_ms

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        h, w, _ = frame.shape

        status_messages = []
        is_slouching = False
        is_tilted = False

        if detection_result.pose_landmarks:
            lms = detection_result.pose_landmarks[0]

            ear_l = (lms[7].x, lms[7].y)
            ear_r = (lms[8].x, lms[8].y)
            sh_l = (lms[11].x, lms[11].y)
            sh_r = (lms[12].x, lms[12].y)

            ear_mid = ((ear_l[0] + ear_r[0]) / 2.0, (ear_l[1] + ear_r[1]) / 2.0)
            sh_mid = ((sh_l[0] + sh_r[0]) / 2.0, (sh_l[1] + sh_r[1]) / 2.0)

            # Metric 1: Cervical Angle Deviation
            current_angle = compute_cervical_angle(ear_mid, sh_mid)
            angle_delta = current_angle - base_angle

            if angle_delta > SLOUCH_ANGLE_THRESHOLD:
                is_slouching = True
                status_messages.append(f"SLOUCH DETECTED (+{angle_delta:.1f} deg)")

            # Metric 2: Shoulder Tilt Deviation
            sh_y_diff = abs(sh_l[1] - sh_r[1])
            normalized_tilt = sh_y_diff / base_sh_width

            if normalized_tilt > SHOULDER_TILT_THRESHOLD:
                is_tilted = True
                status_messages.append("SHOULDER ASYMMETRY DETECTED")

            # Visual overlay colors: Red if flagged, Green if aligned
            spine_color = (0, 0, 255) if is_slouching else (0, 255, 0)
            sh_color = (0, 0, 255) if is_tilted else (0, 255, 0)

            # Draw cervical alignment line
            cv2.line(frame, 
                     (int(ear_mid[0] * w), int(ear_mid[1] * h)), 
                     (int(sh_mid[0] * w), int(sh_mid[1] * h)), 
                     spine_color, 3)

            # Draw biacromial shoulder line
            cv2.line(frame, 
                     (int(sh_l[0] * w), int(sh_l[1] * h)), 
                     (int(sh_r[0] * w), int(sh_r[1] * h)), 
                     sh_color, 3)

            # Draw key points
            for pt in [ear_l, ear_r, sh_l, sh_r]:
                cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 5, (255, 255, 255), -1)

            # Display real-time telemetry metrics
            cv2.putText(frame, f"Baseline Angle: {base_angle:.1f} deg", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Current Angle:  {current_angle:.1f} deg", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Tilt Ratio:     {normalized_tilt:.3f}", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Render alert banners
            if not status_messages:
                cv2.putText(frame, "POSTURE OPTIMAL", (30, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                y_pos = 150
                for msg in status_messages:
                    cv2.putText(frame, msg, (30, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    y_pos += 40

        cv2.imshow("Kinematics - Live Inference Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()