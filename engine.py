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

    # Metrics Tracking
    slouch_violations_count = 0
    tilt_violations_count = 0
    total_slouch_seconds = 0.0
    total_tilt_seconds = 0.0

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

    session_start_time = time.time()
    last_frame_time = session_start_time
    start_time_ms = int(session_start_time * 1000)

    print("\n[INFO] Kinematics Engine running with Session Analytics.")
    print("Press 'q' to end session and generate your Posture Performance Report.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(now * 1000) - start_time_ms

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        h, w, _ = frame.shape

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

            # Slouch Debouncing Logic
            if angle_delta > SLOUCH_ANGLE_THRESHOLD:
                if slouch_start_time is None:
                    slouch_start_time = now
                elif (now - slouch_start_time) >= DEBOUNCE_DURATION:
                    if not slouch_alert_active:
                        slouch_violations_count += 1
                        slouch_alert_active = True
                    total_slouch_seconds += dt
            else:
                slouch_start_time = None
                slouch_alert_active = False

            # Shoulder Tilt Debouncing Logic
            if smooth_tilt > SHOULDER_TILT_THRESHOLD:
                if tilt_start_time is None:
                    tilt_start_time = now
                elif (now - tilt_start_time) >= DEBOUNCE_DURATION:
                    if not tilt_alert_active:
                        tilt_violations_count += 1
                        tilt_alert_active = True
                    total_tilt_seconds += dt
            else:
                tilt_start_time = None
                tilt_alert_active = False

            # HUD & Visualization
            spine_color = (0, 0, 255) if slouch_alert_active else (0, 255, 0)
            sh_color = (0, 0, 255) if tilt_alert_active else (0, 255, 0)

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

            # Elapsed Session Time
            elapsed_session = now - session_start_time
            mins, secs = divmod(int(elapsed_session), 60)

            # Telemetry Metrics HUD
            cv2.putText(frame, f"Session Time:  {mins:02d}:{secs:02d}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Baseline Angle:{base_angle:.1f} deg", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Smoothed Angle:{smooth_angle:.1f} deg (Delta: {angle_delta:+.1f})", 
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Smoothed Tilt: {smooth_tilt:.3f}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Fault Counters
            cv2.putText(frame, f"Slouch Faults: {slouch_violations_count} ({total_slouch_seconds:.1f}s)", 
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.putText(frame, f"Tilt Faults:   {tilt_violations_count} ({total_tilt_seconds:.1f}s)", 
                        (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            # Live Posture Alerts
            y_banner = 250
            if slouch_alert_active:
                cv2.putText(frame, f"[ALERT] Sustained Slouch (+{angle_delta:.1f} deg)", 
                            (30, y_banner), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_banner += 35
            
            if tilt_alert_active:
                cv2.putText(frame, "[ALERT] Sustained Shoulder Tilt", 
                            (30, y_banner), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_banner += 35

            if not slouch_alert_active and not tilt_alert_active:
                cv2.putText(frame, "STATUS: OPTIMAL", (30, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Kinematics - Live Inference Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Session Summary Report Generation
    total_session_duration = max(1.0, time.time() - session_start_time)
    fault_time = min(total_session_duration, total_slouch_seconds + total_tilt_seconds)
    optimal_time = max(0.0, total_session_duration - fault_time)
    posture_score = round((optimal_time / total_session_duration) * 100, 1)

    summary_data = {
        "session_duration_seconds": round(total_session_duration, 2),
        "optimal_posture_seconds": round(optimal_time, 2),
        "slouch_fault_count": slouch_violations_count,
        "slouch_duration_seconds": round(total_slouch_seconds, 2),
        "tilt_fault_count": tilt_violations_count,
        "tilt_duration_seconds": round(total_tilt_seconds, 2),
        "posture_score_percentage": posture_score
    }

    with open("session_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    print("\n" + "="*50)
    print("          PRESENTATION COACH PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Session Duration : {summary_data['session_duration_seconds']}s")
    print(f"Optimal Posture Time   : {summary_data['optimal_posture_seconds']}s")
    print(f"Slouch Violations      : {summary_data['slouch_fault_count']} ({summary_data['slouch_duration_seconds']}s)")
    print(f"Shoulder Tilt Faults   : {summary_data['tilt_fault_count']} ({summary_data['tilt_duration_seconds']}s)")
    print("-" * 50)
    print(f"Final Posture Score    : {summary_data['posture_score_percentage']} / 100")
    print("="*50)
    print("[INFO] Summary saved to 'session_summary.json'.\n")

if __name__ == "__main__":
    main()