import cv2
import json
import math
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def compute_angle_degrees(p_ear_mid, p_shoulder_mid):
    """
    Computes angle (in degrees) of the cervical vector relative to vertical plumb line (0, -1).
    """
    vx = p_ear_mid[0] - p_shoulder_mid[0]
    vy = p_ear_mid[1] - p_shoulder_mid[1]
    
    # Plumb line vector pointing upward in screen space: (0, -1)
    # vy is negative when ear is above shoulder
    norm_v = math.sqrt(vx**2 + vy**2)
    if norm_v == 0:
        return 0.0
    
    # Dot product of (vx, vy) with (0, -1) is -vy
    cos_theta = (-vy) / norm_v
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

def main():
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

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    calibrating = False
    calibrated = False
    calib_duration = 5.0
    calib_start_time = 0.0

    samples = {
        "ear_mid_x": [],
        "ear_mid_y": [],
        "shoulder_mid_x": [],
        "shoulder_mid_y": [],
        "shoulder_width": [],
        "cervical_angle": []
    }

    start_time_ms = int(time.time() * 1000)

    print("\n[INFO] Starting Calibration Engine.")
    print("Sit comfortably in your upright presentation posture.")
    print("Press 'c' to trigger the 5-second baseline capture.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000) - start_time_ms

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        h, w, _ = frame.shape

        if detection_result.pose_landmarks:
            lms = detection_result.pose_landmarks[0]

            # Landmark coordinates (normalized 0.0 - 1.0)
            ear_l = (lms[7].x, lms[7].y)
            ear_r = (lms[8].x, lms[8].y)
            sh_l = (lms[11].x, lms[11].y)
            sh_r = (lms[12].x, lms[12].y)

            # Midpoints
            ear_mid = ((ear_l[0] + ear_r[0]) / 2.0, (ear_l[1] + ear_r[1]) / 2.0)
            sh_mid = ((sh_l[0] + sh_r[0]) / 2.0, (sh_l[1] + sh_r[1]) / 2.0)

            # Biacromial scale normalization distance
            sh_width = math.sqrt((sh_r[0] - sh_l[0])**2 + (sh_r[1] - sh_l[1])**2)
            cerv_angle = compute_angle_degrees(ear_mid, sh_mid)

            # Render key skeleton tracking markers
            for pt in [ear_l, ear_r, sh_l, sh_r]:
                cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 5, (0, 255, 0), -1)

            cv2.line(frame, 
                     (int(ear_mid[0] * w), int(ear_mid[1] * h)), 
                     (int(sh_mid[0] * w), int(sh_mid[1] * h)), 
                     (255, 200, 0), 2)

            # Sample collection during calibration window
            if calibrating:
                elapsed = time.time() - calib_start_time
                remaining = max(0.0, calib_duration - elapsed)

                samples["ear_mid_x"].append(ear_mid[0])
                samples["ear_mid_y"].append(ear_mid[1])
                samples["shoulder_mid_x"].append(sh_mid[0])
                samples["shoulder_mid_y"].append(sh_mid[1])
                samples["shoulder_width"].append(sh_width)
                samples["cervical_angle"].append(cerv_angle)

                cv2.putText(frame, f"CALIBRATING: Keep posture steady ({remaining:.1f}s)", 
                            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if elapsed >= calib_duration:
                    calibrating = False
                    calibrated = True

                    # Compute statistical baseline means
                    profile = {
                        "ear_mid_x": float(np.mean(samples["ear_mid_x"])),
                        "ear_mid_y": float(np.mean(samples["ear_mid_y"])),
                        "shoulder_mid_x": float(np.mean(samples["shoulder_mid_x"])),
                        "shoulder_mid_y": float(np.mean(samples["shoulder_mid_y"])),
                        "baseline_shoulder_width": float(np.mean(samples["shoulder_width"])),
                        "baseline_cervical_angle_deg": float(np.mean(samples["cervical_angle"])),
                        "samples_count": len(samples["cervical_angle"])
                    }

                    with open("calibration_profile.json", "w") as f:
                        json.dump(profile, f, indent=4)

                    print("\n[SUCCESS] Calibration complete and saved to 'calibration_profile.json'.")
                    print(f"Profile: {profile}\n")

        # UI instruction overlays
        if not calibrating and not calibrated:
            cv2.putText(frame, "Sit upright and press 'c' to start calibration", 
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif calibrated:
            cv2.putText(frame, "Calibration Profile Saved! Press 'q' to proceed.", 
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Kinematics - Baseline Calibration Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not calibrating:
            calibrating = True
            calibrated = False
            calib_start_time = time.time()
            for k in samples:
                samples[k] = []
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()