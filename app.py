# app.py
import streamlit as st
import tempfile
import subprocess
import cv2
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------ Constants ------------------
FPS_GRAVITY = 9.81
HIP_LEFT, HIP_RIGHT = 23, 24
KNEE_LEFT, KNEE_RIGHT = 25, 26
ANKLE_LEFT, ANKLE_RIGHT = 27, 28
HEEL_LEFT, HEEL_RIGHT = 29, 30
TOE_LEFT, TOE_RIGHT = 31, 32

LOWER_BODY_CONNECTIONS = [
    (HIP_LEFT, HIP_RIGHT),
    (HIP_LEFT, KNEE_LEFT), (KNEE_LEFT, ANKLE_LEFT),
    (HIP_RIGHT, KNEE_RIGHT), (KNEE_RIGHT, ANKLE_RIGHT)
]

POSE_CONNECTIONS = [
    (11, 12),  # Left shoulder to right shoulder
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    (11, 23),  # Left shoulder to left hip
    (12, 24),  # Right shoulder to right hip
    (23, 24),  # Left hip to right hip
    (23, 25),  # Left hip to left knee
    (25, 27),  # Left knee to left ankle
    (24, 26),  # Right hip to right knee
    (26, 28)   # Right knee to right ankle
]

KEYPOINT_COLOR = (0, 100, 0)  # Dark green in BGR
KEYPOINT_RADIUS = 8
LINE_COLOR = (0, 0, 0)  # Black
VALGUS_COLOR = (180, 105, 255)  # Bright pink in BGR.
VARUS_COLOR  = (128, 0, 128)  # Bright purple in BGR.
LINE_THICKNESS = 2
MAX_FRAMES = 5000
MAX_FRAME_WIDTH = 1080

# ------------------ Global Variables ------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

lite_model_path = 'models/pose_landmarker_lite.task'

normative_data = {
    "knee_delta_mean": 0.0,
    "knee_delta_std": 3.0,
    "knee_flexion_angle_mean": 60.0,
    "knee_flexion_angle_std": 5.0,
    "hip_drop_mean": 5.0,
    "hip_drop_std": 1.0,
    "jump_height_mean": 0.50,
    "jump_height_std": 0.10,
    "asymmetry_threshold": 3.0,
    "ground_contact_time_mean": 0.20,
    "ground_contact_time_std": 0.05,
    "knee_alignment_variability_mean": 1.0,
    "knee_alignment_variability_std": 0.5
}

# ------------------ Utility Functions ------------------
def convert_mov_to_mp4_cv2(input_path, output_path):
    """
    Converts a .mov video file to MP4 using OpenCV.
    Reads the input video, rotates each frame 90° clockwise,
    and writes out an MP4 using the 'mp4v' codec.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback if FPS is not available.
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width, new_height = orig_height, orig_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if orig_width > orig_height:
            out.write(rotated_frame)
        else:
            out.write(frame)
    cap.release()
    out.release()
    return output_path

def convert_video_to_h264(input_path, output_path):
    """
    Converts the input video file to H264-encoded MP4 using FFmpeg.
    This ensures compatibility with HTML5 video players in Streamlit.
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', '-2',  # sometimes needed for AAC
        '-y',  # overwrite output file if exists
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion failed:", e)
        return None

def compute_angle(p1, p2, p3):
    """
    Compute the angle (in degrees) at point p2 given three points p1, p2, p3.
    Each point is a tuple (x, y).
    """
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a) + 1e-8
    mag_b = np.linalg.norm(b) + 1e-8
    if mag_a * mag_b == 0:
        return 0
    angle_rad = np.arccos(dot / (mag_a * mag_b))
    return np.degrees(angle_rad)

# ------------------ Dummy Pose Landmarker ------------------
def create_pose_landmarker(model_asset_path, running_mode=VisionRunningMode.VIDEO):
    """
    Function to create a pose landmarker from a given model asset file.
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=running_mode
    )
    landmarker = PoseLandmarker.create_from_options(options)
    st.write(f"Initialized Lite PoseLandmarker from {model_asset_path}")
    return landmarker

# ------------------ Video Processing Function ------------------
def process_front_video_lite(video_path, pose_landmarker):
    """
    Processes a front-view video ensuring correct (portrait) orientation.
    Returns:
      processed_frames: list of frames with overlays.
      toe_y_list: list of average toe Y values per frame.
      hip_y_list: list of average hip Y values per frame.
      knee_y_list: list of average knee Y values per frame.
      frame_indices: list of frame indices.
      fps: frames per second of the video.
      landmarks_list: list of detected landmarks (dummy values here).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    processed_frames = []
    processing_times = []
    keypoints_counts = []
    toe_y_list = []
    knee_y_list = []
    hip_y_list = []
    frame_indices = []
    landmarks_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < MAX_FRAMES:
        try:
            ret, frame = cap.read()
        except Exception as e:
            st.write("Error reading frame:", e)
            break
        if not ret:
            break

        # If the frame is landscape, rotate it to portrait.
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Resize if necessary.
        if frame.shape[1] > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / frame.shape[1]
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert frame to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_count * 1000 / fps)
        
        # Run pose detection on the frame
        start_time = time.time()
        detection_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        end_time = time.time()
        processing_times.append(end_time - start_time)
        
        # If landmarks detected, extract toe and hip keypoints.
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]  # assume one person
            landmarks_list.append(landmarks)
            keypoints = []
            toe_points = []
            knee_points = []
            hip_points = []

            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                keypoints.append((x, y))

                if (idx == TOE_LEFT or idx == TOE_RIGHT):
                    toe_points.append(y)
                if (idx == KNEE_LEFT or idx == KNEE_RIGHT):
                    knee_points.append(y)
                if (idx == HIP_LEFT or idx == HIP_RIGHT):
                    hip_points.append(y)

                cv2.circle(frame, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

            avg_toe_y = np.mean(toe_points) if toe_points else np.nan
            avg_knee_y = np.mean(knee_points) if knee_points else np.nan
            avg_hip_y = np.mean(hip_points) if hip_points else np.nan
            toe_y_list.append(avg_toe_y)
            knee_y_list.append(avg_knee_y)
            hip_y_list.append(avg_hip_y)

            for connection in POSE_CONNECTIONS:
                idx1, idx2 = connection
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    pt1, pt2 = keypoints[idx1], keypoints[idx2]
                    cv2.line(frame, pt1, pt2, LINE_COLOR, LINE_THICKNESS)
            
            keypoints_counts.append(len(landmarks))
        else:
            cv2.putText(frame, "No pose detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            keypoints_counts.append(0)
            # If no pose detected, use previous values or NaN.
            toe_y_list.append(np.nan)
            knee_y_list.append(np.nan)
            hip_y_list.append(np.nan)
        
        processed_frames.append(frame)
        frame_indices.append(frame_count)
        frame_count += 1

    cap.release()
    total_time = sum(processing_times)
    avg_time = np.mean(processing_times) if processing_times else 0
    avg_keypoints = np.mean(keypoints_counts) if keypoints_counts else 0

    st.write(f"Processed {frame_count} frames from {video_path}.")
    st.write(f"Total processing time: {total_time:.2f} sec.")
    return processed_frames, avg_time, avg_keypoints, toe_y_list, knee_y_list, hip_y_list, landmarks_list, frame_indices, fps

# ------------------ Analysis Functions ------------------
def analyze_jump_metrics(pose_landmarker, processed_frames, toe_y_list, knee_y_list, hip_y_list, frame_indices, fps):
    """
    Analyze jump metrics from the processed frames and pose landmarks.
    """
    # Convert lists to numpy arrays for easier processing.
    toe_y_arr = np.array(toe_y_list)
    knee_y_arr = np.array(knee_y_list)
    hip_y_arr = np.array(hip_y_list)
    
    # Assume baseline toe height is the average of the first 10 valid frames.
    valid_toe = toe_y_arr[~np.isnan(toe_y_arr)]
    if len(valid_toe) >= 10:
        baseline_toe = np.mean(valid_toe[:10])
    else:
        baseline_toe = np.mean(valid_toe)
    st.write(f"Baseline toe height (pixels): {baseline_toe:.2f}")

    # Assume baseline hip height is the average of the first 10 valid frames.
    valid_hip = hip_y_arr[~np.isnan(hip_y_arr)]
    if len(valid_hip) >= 10:
        baseline_hip = np.mean(valid_hip[:10])
    else:
        baseline_hip = np.mean(valid_hip)
    st.write(f"Baseline hip height (pixels): {baseline_hip:.2f}")
    
    # Define a threshold (in pixels) for detecting departure/landing.
    threshold = 20  # adjust this threshold as needed
    
    # Find the takeoff frame: first frame where toe y is significantly lower (i.e., smaller) than baseline.
    takeoff_frame = None
    for i, val in enumerate(toe_y_arr):
        if not np.isnan(val) and val < (baseline_toe - threshold):
            takeoff_frame = i
            break
    
    # Find the apex frame: frame with minimum toe y (i.e., highest jump) after takeoff.
    if takeoff_frame is not None:
        apex_frame = takeoff_frame + np.argmin(toe_y_arr[takeoff_frame:])
    else:
        apex_frame = None

    # Landing: find first frame after apex where toe_y becomes stable.
    # Define stability: over a window of 5 frames, if the range is below a small threshold.
    stability_window = 20  # frames
    stability_threshold = 20  # pixels.
    landing_frame = None
    if apex_frame is not None:
        for i in range(apex_frame, len(toe_y_arr) - stability_window):
            window = toe_y_arr[i:i+stability_window]
            if np.nanmax(window) - np.nanmin(window) < stability_threshold:
                landing_frame = i
                break

    # Calculate time of flight and estimated jump height (using time of flight method).
    if takeoff_frame is not None and landing_frame is not None:
        time_of_flight = (landing_frame - takeoff_frame) / fps
        # Estimated jump height (meters): h = g * (t/2)^2
        estimated_jump_height = 0.5 * FPS_GRAVITY * (time_of_flight / 2)**2
    else:
        time_of_flight = np.nan
        estimated_jump_height = np.nan

    st.write(f"Takeoff frame: {takeoff_frame}, Apex frame: {apex_frame}, Landing frame: {landing_frame}")
    st.write(f"Time of flight: {time_of_flight:.3f} s, Estimated jump height: {estimated_jump_height:.3f} m")

    # Post-landing analysis: From landing_frame onward, find the frame where the hips are at their lowest (largest y value).
    lowest_hip_frame = None
    if landing_frame is not None:
        post_landing_hip = hip_y_arr[landing_frame:]
        if len(post_landing_hip) > 0:
            relative_index = np.argmax(post_landing_hip)  # highest y value means lowest physical hip position.
            lowest_hip_frame = landing_frame + relative_index

    st.write(f"Lowest hip frame (post-landing): {lowest_hip_frame}")

    # Hip drop: difference between landing hip value and lowest hip value.
    hip_drop = hip_y_arr[landing_frame] - hip_y_arr[lowest_hip_frame]
    st.write(f"Hip Drop (pixels): {hip_drop:.2f}")

    # Ground contact time: time between landing and lowest hip point.
    ground_contact_time = (lowest_hip_frame - landing_frame) / fps
    st.write(f"Ground Contact Time: {ground_contact_time:.3f} s")

    # Hip-Knee Crossing: between landing_frame and lowest_hip_frame,
    # find the first frame where hip_y >= knee_y.
    hip_knee_cross_frame = None
    if landing_frame is not None and lowest_hip_frame is not None:
        for i in range(landing_frame, lowest_hip_frame + 1):
            if not np.isnan(hip_y_arr[i]) and not np.isnan(knee_y_arr[i]) and hip_y_arr[i] >= knee_y_arr[i]:
                hip_knee_cross_frame = i
                break

    if hip_knee_cross_frame is not None:
        st.write(f"Hip-Knee crossing frame: {hip_knee_cross_frame}")
    else:
        st.write("Hip never crossed below knee during landing phase.")

    # If we have both landing and lowest_hip frames, compute knee angles for each leg at the lowest hip frame.
    knee_angles = {}
    if lowest_hip_frame is not None and lowest_hip_frame < len(processed_frames):
        # We need to re-run detection on that frame to get the detailed landmarks.
        # For simplicity, assume we stored the detection result per frame;
        # Here, we re-read that frame from processed_frames and run pose detection again.
        frame = processed_frames[lowest_hip_frame]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Use the timestamp corresponding to lowest_hip_frame.
        timestamp_ms = int(lowest_hip_frame * 1000 / fps)
        detection_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            # Left leg knee angle: using hip (23), knee (25), ankle (27)
            pt_hip_left = (int(landmarks[HIP_LEFT].x * frame.shape[1]), int(landmarks[HIP_LEFT].y * frame.shape[0]))
            pt_knee_left = (int(landmarks[KNEE_LEFT].x * frame.shape[1]), int(landmarks[KNEE_LEFT].y * frame.shape[0]))
            pt_ankle_left = (int(landmarks[ANKLE_LEFT].x * frame.shape[1]), int(landmarks[ANKLE_LEFT].y * frame.shape[0]))
            angle_left = compute_angle(pt_hip_left, pt_knee_left, pt_ankle_left)
            # Right leg knee angle: using hip (24), knee (26), ankle (28)
            pt_hip_right = (int(landmarks[HIP_RIGHT].x * frame.shape[1]), int(landmarks[HIP_RIGHT].y * frame.shape[0]))
            pt_knee_right = (int(landmarks[KNEE_RIGHT].x * frame.shape[1]), int(landmarks[KNEE_RIGHT].y * frame.shape[0]))
            pt_ankle_right = (int(landmarks[ANKLE_RIGHT].x * frame.shape[1]), int(landmarks[ANKLE_RIGHT].y * frame.shape[0]))
            angle_right = compute_angle(pt_hip_right, pt_knee_right, pt_ankle_right)
            knee_angles = {'left_knee_angle': angle_left, 'right_knee_angle': angle_right}
            avg_knee_flexion = (angle_left + angle_right) / 2
            st.write(f"Left Knee Angle: {angle_left:.2f} deg, Right Knee Angle: {angle_right:.2f} deg")
            st.write(f"Average Knee Flexion: {avg_knee_flexion:.2f} deg")
        else:
            st.write("No landmarks detected at lowest hip frame for knee angle analysis.")

    # Return a dictionary of metrics and frames
    results = {
        'processed_frames': processed_frames,
        'toe_y_list': toe_y_arr,
        'hip_y_list': hip_y_arr,
        'frame_indices': frame_indices,
        'baseline_toe': baseline_toe,
        'baseline_hip': baseline_hip,
        'hip_drop': hip_drop,
        'takeoff_frame': takeoff_frame,
        'apex_frame': apex_frame,
        'landing_frame': landing_frame,
        'time_of_flight': time_of_flight,
        'ground_contact_time': ground_contact_time,
        'estimated_jump_height': estimated_jump_height,
        'lowest_hip_frame': lowest_hip_frame,
        'hip_knee_cross_frame': hip_knee_cross_frame,
        'knee_angles': knee_angles,
        'knee_flexion_angle': avg_knee_flexion
    }
    return results

def analyze_knee_valgus_varus(processed_frames, landmarks_list, landing_frame, lowest_hip_frame, align_threshold=15):
    """
    For each frame in the landing phase (from landing_frame to lowest_hip_frame),
    determine knee alignment by comparing knee and ankle x values.
    """
    left_deltas = []
    right_deltas = []
    
    # Iterate through landing phase frames.
    for i in range(landing_frame, lowest_hip_frame + 1):
        landmarks = landmarks_list[i]
        if landmarks is not None:
            frame = processed_frames[i]
            h, w = frame.shape[:2]
            # Left leg: indices for left knee and ankle.
            left_knee_x = int(landmarks[KNEE_LEFT].x * w)
            left_ankle_x = int(landmarks[ANKLE_LEFT].x * w)
            delta_left = left_ankle_x - left_knee_x  # positive => knee more medial → valgus; negative => varus.
            left_deltas.append(delta_left)
            
            # Right leg: use mirrored logic.
            right_knee_x = int(landmarks[KNEE_RIGHT].x * w)
            right_ankle_x = int(landmarks[ANKLE_RIGHT].x * w)
            # For right leg, if knee is medially deviated, knee_x will be lower than ankle_x.
            delta_right = right_knee_x - right_ankle_x  # positive => valgus; negative => varus.
            right_deltas.append(delta_right)
            
            # Determine condition and color for left leg.
            if delta_left > align_threshold:
                condition_left = "valgus"
                color_left = VALGUS_COLOR
            elif delta_left < -align_threshold:
                condition_left = "varus"
                color_left = VARUS_COLOR
            else:
                condition_left = "neutral"
                color_left = (0, 255, 0)
            
            # Determine condition for right leg.
            if delta_right > align_threshold:
                condition_right = "valgus"
                color_right = VALGUS_COLOR
            elif delta_right < -align_threshold:
                condition_right = "varus"
                color_right = VARUS_COLOR
            else:
                condition_right = "neutral"
                color_right = (0, 255, 0)
            
            # Overlay line segments: draw a thick line from knee to ankle.
            left_knee_y = int(landmarks[KNEE_LEFT].y * h)
            left_ankle_y = int(landmarks[ANKLE_LEFT].y * h)
            cv2.line(frame, (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), color_left, thickness=5)
            right_knee_y = int(landmarks[KNEE_RIGHT].y * h)
            right_ankle_y = int(landmarks[ANKLE_RIGHT].y * h)
            cv2.line(frame, (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), color_right, thickness=5)
            
            # Overlay text labels.
            cv2.putText(frame, f"L:{condition_left} ({delta_left:+.1f})", (left_knee_x, left_knee_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_left, 2)
            cv2.putText(frame, f"R:{condition_right} ({delta_right:+.1f})", (right_knee_x, right_knee_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_right, 2)
        else:
            left_deltas.append(np.nan)
            right_deltas.append(np.nan)
    
    # Compute average delta values.
    avg_left_delta = np.nanmean(left_deltas) if left_deltas else np.nan
    avg_right_delta = np.nanmean(right_deltas) if right_deltas else np.nan
    asymmetry = abs(avg_left_delta - avg_right_delta)
    knee_alignment_variability = np.nanmean([np.nanstd(left_deltas), np.nanstd(right_deltas)])
    
    # Risk scoring per leg.
    def risk_category(avg_delta):
        if np.isnan(avg_delta):
            return "unknown"
        if abs(avg_delta) < align_threshold:
            return "safe"
        elif abs(avg_delta) < 2 * align_threshold:
            return "caution"
        else:
            return "high risk"
    
    left_risk = risk_category(avg_left_delta)
    right_risk = risk_category(avg_right_delta)
    overall_risk = "high risk" if "high risk" in [left_risk, right_risk] else ("caution" if "caution" in [left_risk, right_risk] else "safe")
    
    st.write(f"Left average delta: {avg_left_delta:.2f} ({left_risk}), Right average delta: {avg_right_delta:.2f} ({right_risk}).")
    st.write(f"Overall risk: {overall_risk}")
    
    return {
        'left_avg_delta': avg_left_delta,
        'right_avg_delta': avg_right_delta,
        'left_risk': left_risk,
        'right_risk': right_risk,
        'overall_risk': overall_risk,
        'left_deltas': left_deltas,
        'right_deltas': right_deltas,
        'asymmetry': asymmetry,
        'knee_alignment_variability': knee_alignment_variability
    }

# ------------------ Risk Scoring and Recommendations ------------------
def compute_composite_risk_score(metrics, normative):
    """
    Computes a composite risk score for lower-extremity injury risk (e.g., ACL injury) 
    and overall neuromuscular performance based on several biomechanical metrics.
    """
    # 1. Knee Alignment Score.
    left_z = abs(metrics["left_knee_delta"] - normative["knee_delta_mean"]) / normative["knee_delta_std"]
    right_z = abs(metrics["right_knee_delta"] - normative["knee_delta_mean"]) / normative["knee_delta_std"]
    knee_alignment_score = max(left_z, right_z)
    
    # 2. Knee Flexion Score.
    # Lower-than-normative flexion (stiffer landing) is unfavorable.
    flexion_deficit = max(0, normative["knee_flexion_angle_mean"] - metrics["knee_flexion_angle"])
    knee_flexion_score = flexion_deficit / normative["knee_flexion_angle_std"]
    
    # 3. Hip Drop Score.
    hip_drop_excess = max(0, metrics["hip_drop"] - normative["hip_drop_mean"])
    hip_drop_score = hip_drop_excess / normative["hip_drop_std"]
    
    # 4. Jump Performance Score.
    jump_deficit = max(0, normative["jump_height_mean"] - metrics["jump_height"])
    jump_score = jump_deficit / normative["jump_height_std"]
    
    # 5. Asymmetry Score.
    asymmetry_score = metrics["asymmetry"] / normative["asymmetry_threshold"]
    
    # 6. Ground Contact Time Score.
    gct_excess = max(0, metrics["ground_contact_time"] - normative["ground_contact_time_mean"])
    ground_contact_score = gct_excess / normative["ground_contact_time_std"]
    
    # 7. Knee Alignment Variability Score.
    variability_excess = max(0, metrics["knee_alignment_variability"] - normative["knee_alignment_variability_mean"])
    variability_score = variability_excess / normative["knee_alignment_variability_std"]
    
    # Weight the individual components based on clinical significance.
    # For instance, excessive knee alignment deviation (valgus/varus) is highly linked to ACL injuries.
    weights = {
        "knee_alignment": 0.30,
        "knee_flexion": 0.25,
        "hip_drop": 0.15,
        "jump_performance": 0.10,
        "asymmetry": 0.10,
        "ground_contact": 0.05,
        "variability": 0.05
    }
    
    composite_score = (weights["knee_alignment"] * knee_alignment_score +
                       weights["knee_flexion"] * knee_flexion_score +
                       weights["hip_drop"] * hip_drop_score +
                       weights["jump_performance"] * jump_score +
                       weights["asymmetry"] * asymmetry_score +
                       weights["ground_contact"] * ground_contact_score +
                       weights["variability"] * variability_score)
    
    # Define broader risk categories.
    if composite_score < 0.8:
        risk_category = "Optimal"
    elif composite_score < 1.5:
        risk_category = "Good"
    elif composite_score < 2.2:
        risk_category = "Moderate Risk"
    elif composite_score < 3.0:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"
    
    component_scores = {
        "knee_alignment_score": knee_alignment_score,
        "knee_flexion_score": knee_flexion_score,
        "hip_drop_score": hip_drop_score,
        "jump_score": jump_score,
        "asymmetry_score": asymmetry_score,
        "ground_contact_score": ground_contact_score,
        "variability_score": variability_score
    }
    
    return composite_score, risk_category, component_scores

def generate_recommendations(composite_score, risk_category, component_scores, measured_metrics):
    """
    Generates detailed, tailored textual feedback and recommendations based on the composite risk score,
    individual component scores, and raw measured metrics.
    """
    recommendations = []
    recommendations.append("=== Overall Analysis ===\n")
    recommendations.append(f"Composite Risk Score: {composite_score:.2f}\n")
    recommendations.append(f"Overall Risk Category: {risk_category}\n")
    recommendations.append("")
    
    # Knee Alignment
    if component_scores["knee_alignment_score"] > 1.5:
        recommendations.append("Knee Alignment: Your knee alignment deviates significantly from neutral. "
                               "Excessive dynamic knee valgus/varus is strongly linked to ACL injury risk. "
                               "It is recommended that you engage in neuromuscular training (e.g., plyometrics, "
                               "landing mechanics drills, and dynamic balance exercises) to improve alignment.\n")
    else:
        recommendations.append("Knee Alignment: Your knee alignment is within an acceptable range.\n")
    
    # Knee Flexion
    if component_scores["knee_flexion_score"] > 1.0:
        recommendations.append("Knee Flexion: Your landing knee flexion is lower than the optimal value. "
                               "A stiffer landing increases impact forces on the knee. Consider incorporating "
                               "eccentric strength training and landing technique coaching to enhance shock absorption.\n")
    else:
        recommendations.append("Knee Flexion: Your landing technique shows adequate knee flexion.\n")
    
    # Hip Drop
    if component_scores["hip_drop_score"] > 1.0:
        recommendations.append("Hip Drop: Excessive hip drop indicates poor core and gluteal control. "
                               "Strengthening exercises focusing on the glutes, hip abductors, and core stability, "
                               "such as lateral band walks, single-leg squats, and planks, may help reduce this risk.\n")
    else:
        recommendations.append("Hip Drop: Your hip drop is within a normal range.\n")
    
    # Jump Performance
    if component_scores["jump_score"] > 1.0:
        recommendations.append("Jump Performance: Your estimated jump height is below normative values, "
                               "which may reflect deficits in lower-body power. Plyometric exercises and "
                               "explosive strength training (e.g., box jumps, squat jumps) could improve performance.\n")
    else:
        recommendations.append("Jump Performance: Your jump height is comparable to normative data.\n")
    
    # Asymmetry
    if component_scores["asymmetry_score"] > 1.0:
        recommendations.append("Asymmetry: There is a noticeable imbalance between your left and right limb mechanics. "
                               "Unilateral strength and stability exercises can help address this imbalance and reduce injury risk.\n")
    else:
        recommendations.append("Asymmetry: Your bilateral movement symmetry is good.\n")
    
    # Ground Contact Time
    if component_scores["ground_contact_score"] > 1.0:
        recommendations.append("Ground Contact Time: Prolonged ground contact time may indicate inefficient landing mechanics. "
                               "Consider drills that emphasize quick, explosive reactivity to improve shock absorption and reduce injury risk.\n")
    else:
        recommendations.append("Ground Contact Time: Your ground contact time is within an optimal range.\n")
    
    # Knee Alignment Variability
    if component_scores["variability_score"] > 1.0:
        recommendations.append("Knee Alignment Variability: High variability in knee alignment suggests inconsistent neuromuscular control. "
                               "Balance and stabilization exercises can help promote more consistent movement patterns during landing.\n")
    else:
        recommendations.append("Knee Alignment Variability: Your knee alignment remains consistently controlled during landing.\n")
    
    recommendations.append("")
    recommendations.append("=== Summary and Next Steps ===\n")
    if risk_category in ["High Risk", "Very High Risk"]:
        recommendations.append("Your overall risk profile is concerning. It is highly recommended that you consult "
                               "with a sports physiotherapist or strength and conditioning specialist to design "
                               "a targeted intervention program focusing on neuromuscular control, strength, and proper landing technique.\n")
    elif risk_category == "Moderate Risk":
        recommendations.append("Your metrics indicate a moderate risk. With targeted training focusing on the specific areas identified above, "
                               "you can likely reduce your injury risk. Consider integrating corrective exercises and monitoring progress over time.\n")
    else:
        recommendations.append("Your performance metrics are within optimal or good ranges. Continue with your current training, "
                               "but consider incorporating advanced drills to further enhance performance and maintain injury resilience.\n")
    
    # Optional: Provide numeric summaries of each component.
    recommendations.append("")
    recommendations.append("Detailed Component Scores:")
    for comp, score in component_scores.items():
        recommendations.append(f" - {comp.replace('_', ' ').title()}: {score:.2f}")
    
    # Optional: Provide measured values for further context.
    recommendations.append("")
    recommendations.append("Measured Metrics:")
    for key, value in measured_metrics.items():
        recommendations.append(f" - {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(recommendations)

def export_video(frames, temp_output_path, fps):
    """
    Exports the given list of frames to a temporary MP4 video file using OpenCV.
    Returns the path of the exported video file.
    """
    if not frames:
        st.error("No frames to export.")
        return None
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_output_path

def export_variable_speed_video(processed_frames, output_path, output_fps=120,
                                takeoff_frame=None, landing_frame=None, lowest_hip_frame=None):
    """
    Exports a video with variable playback speeds by duplicating frames in certain segments.
    
    Segmentation:
      - Pre-takeoff (frame 0 to takeoff_frame-1): 120 fps (normal)
      - Takeoff-to-landing (takeoff_frame to landing_frame-1): Each frame is repeated 2 times
          so that playback appears at 60 fps (0.5x speed).
      - Landing-to-lowest-hip (landing_frame to lowest_hip_frame-1): Each frame is repeated 4 times
          so that playback appears at 30 fps (0.25x speed).
      - Post-lowest-hip (lowest_hip_frame onward): 120 fps (normal).
    
    The final output video is written at output_fps (default 120 fps).
    """
    if not processed_frames:
        st.error("No frames to export.")
        return
    
    total_frames = len(processed_frames)
    new_frames = []
    
    # Segment 1: Pre-takeoff (normal speed).
    pre_takeoff_end = takeoff_frame if takeoff_frame is not None else 0
    for i in range(0, pre_takeoff_end):
        new_frames.append(processed_frames[i])
    
    # Segment 2: Takeoff-to-landing (0.5x speed: each frame repeated 2 times).
    if takeoff_frame is not None and landing_frame is not None:
        for i in range(takeoff_frame, landing_frame):
            # Append 2 copies of each frame.
            new_frames.extend([processed_frames[i]] * 2)
    else:
        st.write("Warning: takeoff or landing frame not defined; skipping slow-motion segment 2.")
    
    # Segment 3: Landing-to-lowest-hip (0.25x speed: each frame repeated 4 times).
    if landing_frame is not None and lowest_hip_frame is not None:
        for i in range(landing_frame, lowest_hip_frame):
            new_frames.extend([processed_frames[i]] * 4)
    else:
        st.write("Warning: landing or lowest hip frame not defined; skipping slow-motion segment 3.")
    
    # Segment 4: Post-lowest-hip (normal speed).
    if lowest_hip_frame is not None:
        for i in range(lowest_hip_frame, total_frames):
            new_frames.append(processed_frames[i])
    else:
        # If lowest_hip_frame is undefined, add the rest normally.
        for i in range(landing_frame if landing_frame is not None else 0, total_frames):
            new_frames.append(processed_frames[i])
    
    # Use the dimensions from the first frame.
    height, width = new_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    st.write(f"Exporting {len(new_frames)} frames to {output_path} at {output_fps} fps...")
    for frame in new_frames:
        out.write(frame)
    out.release()
    st.write(f"Export completed: {output_path}")
    return output_path

# ------------------ Streamlit UI Code ------------------

# Configure the page (the dark theme is configured in .streamlit/config.toml)
st.set_page_config(page_title="Jump Analysis & Injury Risk Tool", layout="wide")
st.title("Jump Analysis & Injury Risk Tool")
st.markdown("## Upload Your Video")

uploaded_file = st.file_uploader("Choose a video file (mp4, mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file.
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    tfile.write(uploaded_file.read())
    st.success("Video uploaded successfully!")
    video_path = tfile.name
    tfile.close()
    input_video_path = video_path
    st.write(f"Video path: {video_path}")

    if file_ext == ".mov":
        st.info("Converting MOV file to MP4 using OpenCV...")
        temp_mp4_path = os.path.join(tempfile.gettempdir(), "converted_from_mov.mp4")
        input_video_path = convert_mov_to_mp4_cv2(input_video_path, temp_mp4_path)

    output_video_path = os.path.join(tempfile.gettempdir(), "converted_upload_video.mp4")
    st.info("Converting video... Please wait.")
    converted_path = convert_video_to_h264(input_video_path, output_video_path)
    if converted_path and os.path.exists(converted_path):
        st.success("Video conversion successful!")
        with open(converted_path, "rb") as video_file:
            video_bytes = video_file.read()
        col_left, col_mid, col_right = st.columns([1, 1, 1])
        with col_mid:
            st.header("Your Uploaded Video")
            st.video(video_bytes)
    else:
        st.error("Video conversion failed.")
    # st.video(video_path)

    # Create pose landmarker (dummy instance).
    pose_landmarker = create_pose_landmarker(lite_model_path)
    
    st.info("Processing video... Please wait.")
    processed_frames, avg_time_lite_front, avg_kp_lite_front, toe_y_list, knee_y_list, hip_y_list, landmarks_list, frame_indices, fps = process_front_video_lite(video_path, pose_landmarker)
    
    if processed_frames:
        pose_landmarker = create_pose_landmarker(lite_model_path)
        jump_metrics = analyze_jump_metrics(pose_landmarker, processed_frames, toe_y_list, knee_y_list, hip_y_list, frame_indices, fps)
        knee_analysis = analyze_knee_valgus_varus(processed_frames, landmarks_list, jump_metrics['landing_frame'], jump_metrics['lowest_hip_frame'])
        
        measured_metrics = {
            "left_knee_delta": knee_analysis['left_avg_delta'],
            "right_knee_delta": knee_analysis['right_avg_delta'],
            "knee_flexion_angle": jump_metrics['knee_flexion_angle'],
            "hip_drop": jump_metrics['hip_drop'],
            "jump_height": jump_metrics['estimated_jump_height'],
            "asymmetry": knee_analysis['asymmetry'],
            "ground_contact_time": jump_metrics['ground_contact_time'],
            "knee_alignment_variability": knee_analysis['knee_alignment_variability']
        }
        
        composite_score, risk_category, component_scores = compute_composite_risk_score(measured_metrics, normative_data)
        recommendations = generate_recommendations(composite_score, risk_category, component_scores, measured_metrics)
        
        # output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        # saved_video_path = export_video(processed_frames, output_video_path, fps)
        # processed_video_bytes = export_video(processed_frames, fps)
        temp_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        if fps > 60:
            exported_path = export_variable_speed_video(processed_frames, temp_video_path, output_fps=120,
                                                    takeoff_frame=jump_metrics['takeoff_frame'],
                                                    landing_frame=jump_metrics['landing_frame'],
                                                    lowest_hip_frame=jump_metrics['lowest_hip_frame'])
        else:
            exported_path = export_video(processed_frames, temp_video_path, fps)
        converted_video_path = os.path.join(tempfile.gettempdir(), "processed_video_converted.mp4")
        final_video_path = convert_video_to_h264(exported_path, converted_video_path)
        
        # Layout: left column for interactive plots and textual feedback; right column for video.
        col_left, col_right = st.columns([1.33, 1])
        
        with col_left:
            st.header("Interactive Plots")
            # Plot Toe Y over time with markers
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, jump_metrics['toe_y_list'], label='Average Toe Y')
            ax.axhline(jump_metrics['baseline_toe'], color='r', linestyle='--', label='Baseline Toe')
            if jump_metrics['takeoff_frame'] is not None:
                ax.axvline(jump_metrics['takeoff_frame'], color='g', linestyle='--', label='Takeoff')
            if jump_metrics['apex_frame'] is not None:
                ax.axvline(jump_metrics['apex_frame'], color='m', linestyle='--', label='Apex')
                ax.plot(jump_metrics['apex_frame'], jump_metrics['toe_y_list'][jump_metrics['apex_frame']], 'r*', markersize=15, label='Apex (peak)')
            if jump_metrics['landing_frame'] is not None:
                ax.axvline(jump_metrics['landing_frame'], color='b', linestyle='--', label='Landing')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Toe Y (pixels)')
            ax.set_title('Toe Y Position Over Time')
            ax.legend()
            st.pyplot(fig)

            # Plot Hip Y over time with markers
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, jump_metrics['hip_y_list'], label='Average Hip Y', color='purple')
            if jump_metrics['lowest_hip_frame'] is not None:
                ax.axvline(jump_metrics['lowest_hip_frame'], color='orange', linestyle='--', label='Lowest Hip')
            if jump_metrics['hip_knee_cross_frame'] is not None:
                ax.axvline(jump_metrics['hip_knee_cross_frame'], color='cyan', linestyle='--', label='Hip-Knee Crossing')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Hip Y (pixels)')
            ax.set_title('Hip Y Position Over Time')
            ax.legend()
            st.pyplot(fig)

            # Plot the delta values.
            landing_indices = list(range(jump_metrics['landing_frame'], jump_metrics['lowest_hip_frame'] + 1))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(landing_indices, knee_analysis['left_deltas'], label='Left Delta', color='blue')
            ax.plot(landing_indices, knee_analysis['right_deltas'], label='Right Delta', color='red')
            ax.axhline(y=15, color='black', linestyle='--', label='Threshold')
            ax.axhline(y=-15, color='black', linestyle='--')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Delta X (pixels)')
            ax.set_title('Knee Alignment Delta Values During Landing Phase')
            ax.legend()
            st.pyplot(fig)
        with col_right:
            st.header("Processed Video")
            if final_video_path and os.path.exists(final_video_path):
                with open(final_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.error("The exported video could not be displayed.")

        st.header("Textual Feedback")
        st.markdown(recommendations)
    else:
        st.error("Video processing failed or produced no frames.")
