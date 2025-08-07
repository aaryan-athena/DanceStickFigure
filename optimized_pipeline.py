import sys
sys.stdout.reconfigure(line_buffering=True)

import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import io
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_optimized_pipeline(input_file="input_videos/dance2.mov"):
    """
    Optimized pipeline that processes video directly without intermediate files
    """
    input_path = os.path.join(BASE_DIR, input_file)
    output_videos_dir = os.path.join(BASE_DIR, "output_videos")
    os.makedirs(output_videos_dir, exist_ok=True)

    # Initialize MediaPipe with optimized settings
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Use lighter model
        enable_segmentation=False,  # Disable unnecessary segmentation
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}", flush=True)
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = 1280, 720  # Fixed output size
    
    print(f"VIDEO_FPS:{fps}", flush=True)
    print(f"TOTAL_FRAMES:{total_frames}", flush=True)

    # Prepare output video writer
    output_path = os.path.join(output_videos_dir, 'stick_figure_tutorial.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pose connection pairs (optimized set)
    POSE_PAIRS = [
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # Right arm
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # Left arm  
        (23, 25), (25, 27), (27, 29), (29, 31),            # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32),            # Left leg
        (11, 12), (23, 24), (11, 23), (12, 24),            # Torso
        (0, 11), (0, 12)                                   # Neck
    ]

    frame_idx = 0
    prev_points = None
    SMOOTHING_ALPHA = 0.4

    # Process frames directly
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"FRAME_IDX:{frame_idx}", flush=True)

        # Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Extract and smooth keypoints
        points = extract_keypoints(results, width, height)
        if prev_points and len(prev_points) == len(points):
            points = smooth_keypoints(points, prev_points, SMOOTHING_ALPHA)
        prev_points = points.copy()

        # Generate stick figure frame directly
        stick_frame = create_stick_figure_frame(points, POSE_PAIRS, width, height)
        
        # Write frame to output video
        out.write(stick_frame)
        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()
    pose.close()

    print(f"✅ Stick figure video created: {output_path}", flush=True)

    # Add audio using FFmpeg (optimized command)
    output_with_audio = os.path.join(output_videos_dir, 'stick_figure_with_audio.mp4')
    add_audio_to_video(input_path, output_path, output_with_audio)

def extract_keypoints(results, width, height):
    """Extract and normalize keypoints from MediaPipe results"""
    points = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            visibility = landmark.visibility
            if visibility > 0.1:
                points.append((x, y))
            else:
                points.append(None)
    return points

def smooth_keypoints(current_points, prev_points, alpha):
    """Apply temporal smoothing to keypoints"""
    smoothed = []
    for curr, prev in zip(current_points, prev_points):
        if curr and prev:
            x = alpha * curr[0] + (1 - alpha) * prev[0]
            y = alpha * curr[1] + (1 - alpha) * prev[1]
            smoothed.append((x, y))
        else:
            smoothed.append(curr)
    return smoothed

def create_stick_figure_frame(points, pose_pairs, width, height):
    """Create stick figure frame using matplotlib (optimized)"""
    # Create figure with fixed size and DPI for consistency
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_facecolor('black')

    if len(points) > 24:
        # Draw torso (optimized check)
        torso_points = [points[11], points[12], points[24], points[23]]  # shoulders and hips
        if all(torso_points):
            torso = Polygon(torso_points, closed=True, color='orange', alpha=0.5)
            ax.add_patch(torso)

            # Add chest and stomach markers
            mid_shoulder = ((torso_points[0][0] + torso_points[1][0]) / 2,
                           (torso_points[0][1] + torso_points[1][1]) / 2)
            mid_hip = ((torso_points[2][0] + torso_points[3][0]) / 2,
                      (torso_points[2][1] + torso_points[3][1]) / 2)
            
            ax.add_patch(Circle(mid_shoulder, radius=10, color='red'))
            ax.add_patch(Circle(mid_hip, radius=10, color='purple'))

            # Draw neck
            if points[0]:
                neck = Line2D([points[0][0], mid_shoulder[0]], 
                             [points[0][1], mid_shoulder[1]], 
                             linewidth=4, color='brown')
                ax.add_line(neck)

    # Draw limbs and joints
    for pair in pose_pairs:
        a, b = pair
        if a < len(points) and b < len(points) and points[a] and points[b]:
            # Draw limb
            limb = Line2D([points[a][0], points[b][0]], 
                         [points[a][1], points[b][1]], 
                         linewidth=6, color='red')
            ax.add_line(limb)
            
            # Draw joints
            ax.add_patch(Circle(points[a], radius=5, color='black'))
            ax.add_patch(Circle(points[b], radius=5, color='black'))

    # Draw head
    if points and points[0]:
        head_radius = height * 0.04
        head_circle = Circle(points[0], radius=head_radius, 
                           fill=False, color='blue', linewidth=3)
        ax.add_patch(head_circle)
        
        # Eyes
        eye_offset_x = head_radius * 0.3
        eye_offset_y = head_radius * 0.3
        ax.plot(points[0][0] - eye_offset_x, points[0][1] - eye_offset_y, 'bo', markersize=4)
        ax.plot(points[0][0] + eye_offset_x, points[0][1] - eye_offset_y, 'bo', markersize=4)

    # Convert matplotlib figure to OpenCV format
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black', bbox_inches='tight', 
                pad_inches=0, dpi=100)
    buf.seek(0)
    
    # Convert to OpenCV image
    img = Image.open(buf)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to exact dimensions if needed
    img_bgr = cv2.resize(img_bgr, (width, height))
    
    plt.close(fig)
    buf.close()
    
    return img_bgr

def add_audio_to_video(input_path, video_path, output_path):
    """Add audio from input video to stick figure video"""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",  # Faster encoding
        "-crf", "28",           # Slightly lower quality for speed
        "-c:a", "aac",
        "-b:a", "128k",         # Lower audio bitrate
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"✅ Video with audio created: {output_path}", flush=True)
    except subprocess.CalledProcessError:
        # Fallback: add silent audio
        print("⚠️ Adding silent audio track...", flush=True)
        ffmpeg_silent = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(ffmpeg_silent, check=True, capture_output=True)
        print(f"✅ Video with silent audio created: {output_path}", flush=True)

if __name__ == "__main__":
    run_optimized_pipeline()
