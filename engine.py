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

def apply_ema(current_val, prev_val, alpha=0.25):
    if prev_val is None:
        return current_val
    return (alpha * current_val) + ((1.0 - alpha) * prev_val)

def main():
    profile = load_calibration()
    base_angle = profile["baseline_cervical_angle_deg"]
    base_sh_width = profile["baseline_shoulder_width"]

    # Kinematic Thresholds
    SLOUCH_ANGLE_THRESHOLD = 8.0     # degrees deviation from baseline
    SHOULDER_TILT_THRESHOLD = 0.05   # normalized ratio threshold
    DEBOUNCE_DURATION = 1.8          # seconds required before triggering persistent alert

    # State variables for EMA smoothing
    smooth_angle = None
    smooth_tilt = None

    # State variables for Temporal Debouncing
    slouch_start_time = None
    tilt_start_time = None
    slouch_alert_active = False
    tilt_alert_active = False

    # Violation Counters
    slouch_violations_count = 0
    tilt_violations_count = 0

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

    print("\n[INFO] Kinematics Engine running with EMA smoothing & Temporal Debouncing.")
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
        now = time.time()

        if detection_result.pose_landmarks:
            lms = detection_result.pose_landmarks[0]

            ear_l = (lms[7].x, lms[7].y)
            ear_r = (lms[8].x, lms[8].y)
            sh_l = (lms[11].x, lms[11].y)
            sh_r = (lms[12].x, lms[12].y)

            ear_mid = ((ear_l[0] + ear_r[0]) / 2.0, (ear_l[1] + ear_r[1]) / 2.0)
            sh_mid = ((sh_l[0] + sh_r[0]) / 2.0, (sh_l[1] + sh_r[1]) / 2.0)

            # Raw values
            raw_angle = compute_cervical_angle(ear_mid, sh_mid)
            raw_tilt = abs(sh_l[1] - sh_r[1]) / base_sh_width

            # Applied EMA Filtering
            smooth_angle = apply_ema(raw_angle, smooth_angle, alpha=0.25)
            smooth_tilt = apply_ema(raw_tilt, smooth_tilt, alpha=0.25)

            angle_delta = smooth_angle - base_angle

            # ---------------------------------------------------------
            # Slouch Debouncing Logic
            # ---------------------------------------------------------
            if angle_delta > SLOUCH_ANGLE_THRESHOLD:
                if slouch_start_time is None:
                    slouch_start_time = now
                elif (now - slouch_start_time) >= DEBOUNCE_DURATION:
                    if not slouch_alert_active:
                        slouch_violations_count += 1
                        slouch_alert_active = True
            else:
                slouch_start_time = None
                slouch_alert_active = False

            # ---------------------------------------------------------
            # Shoulder Tilt Debouncing Logic
            # ---------------------------------------------------------
            if smooth_tilt > SHOULDER_TILT_THRESHOLD:
                if tilt_start_time is None:
                    tilt_start_time = now
                elif (now - tilt_start_time) >= DEBOUNCE_DURATION:
                    if not tilt_alert_active:
                        tilt_violations_count += 1
                        tilt_alert_active = True
            else:
                tilt_start_time = None
                tilt_alert_active = False

            # ---------------------------------------------------------
            # HUD & Visualization
            # ---------------------------------------------------------
            spine_color = (0, 0, 255) if slouch_alert_active else (0, 255, 0)
            sh_color = (0, 0, 255) if tilt_alert_active else (0, 255, 0)

            # Draw skeleton segments
            cv2.line(frame, 
                     (int(ear_mid[0] * w), int(ear_mid[1] * h)), 
                     (int(sh_mid[0] * w), int(sh_mid[1] * h)), 
                     spine_color, 3)

            cv2.line(frame, 
                     (int(sh_l[0] * w), int(sh_l[1] * h)), 
                     (int(sh_r[0] * w), int(sh_r[1] * h)), 
                     sh_color, 3)

            for pt in [ear_l, ear_r, sh_l, sh_r]:
                cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 5, (255, 255, 255), -1)

            # Telemetry Metrics HUD
            cv2.putText(frame, f"Baseline Angle: {base_angle:.1f} deg", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(frame, f"Smoothed Angle: {smooth_angle:.1f} deg (Delta: {angle_delta:+.1f})", 
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(frame, f"Smoothed Tilt:  {smooth_tilt:.3f}", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

            # Persistent Violation Counters
            cv2.putText(frame, f"Slouch Faults: {slouch_violations_count}", (30, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"Tilt Faults:   {tilt_violations_count}", (30, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Active Alert Banners
            y_banner = 220
            if slouch_alert_active:
                cv2.putText(frame, f"[ALERT] Sustained Slouching (+{angle_delta:.1f} deg)", 
                            (30, y_banner), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                y_banner += 35
            
            if tilt_alert_active:
                cv2.putText(frame, "[ALERT] Sustained Shoulder Tilt", 
                            (30, y_banner), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                y_banner += 35

            if not slouch_alert_active and not tilt_alert_active:
                cv2.putText(frame, "STATUS: OPTIMAL", (30, 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        cv2.imshow("Kinematics - Live Inference Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()