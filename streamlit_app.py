import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from PIL import Image
import io
import tempfile
import os
import time
from typing import List, Tuple, Optional

# Check deployment environment
def check_deployment_environment():
    """Check if running in a deployment environment and show info"""
    if os.path.exists('/home/adminuser') or os.path.exists('/app'):
        st.info("🌐 Running in deployment environment - optimized for cloud hosting")
        return True
    return False

def check_mediapipe_version():
    """Check MediaPipe version compatibility"""
    try:
        mp_version = mp.__version__
        st.sidebar.info(f"📦 MediaPipe version: {mp_version}")
        
        # Check for known problematic versions
        if mp_version.startswith('0.10'):
            st.sidebar.warning("⚠️ MediaPipe 0.10.x may have deployment issues. Consider updating.")
    except:
        pass

IS_DEPLOYED = check_deployment_environment()
check_mediapipe_version()

# Optimized configuration
st.set_page_config(page_title="Show Me The Moves 💃🕺", layout="centered")
st.title("💃🕺 Show Me The Moves")
st.write("Turn your dance moves into a custom stick figure animation!")
st.markdown("**⚡ Lightweight & Fast** - Upload a video and get your stick figure animation!")

# Initialize MediaPipe with optimized settings
@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe with optimized settings for performance and deployment compatibility"""
    try:
        mp_pose = mp.solutions.pose
        
        # Different settings for deployment vs local
        if IS_DEPLOYED:
            # Deployment-optimized settings
            pose = mp_pose.Pose(
                static_image_mode=True,  # Static mode works better in deployment
                model_complexity=1,  # Middle complexity for stability
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            st.success("✅ MediaPipe initialized for deployment environment")
        else:
            # Local development settings
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Lightest model for speed
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.success("✅ MediaPipe initialized for local environment")
        
        return pose
        
    except Exception as e:
        st.warning(f"⚠️ MediaPipe initialization issue: {str(e)}")
        st.info("🔄 Trying fallback MediaPipe configuration...")
        
        try:
            # Fallback configuration
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()  # Use default settings
            st.info("✅ MediaPipe initialized with default settings")
            return pose
            
        except Exception as e2:
            st.error(f"❌ Failed to initialize MediaPipe: {str(e2)}")
            st.error("💡 This might be a deployment environment issue. Try using a different hosting platform or run locally.")
            return None

# Pose connection pairs for stick figure
POSE_PAIRS = [
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # Right arm
    (23, 25), (25, 27), (27, 29), (29, 31),            # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32),            # Right leg
    (11, 12), (23, 24), (11, 23), (12, 24),            # Torso
    (0, 11), (0, 12)                                    # Head to shoulders
]

def create_stick_figure(pose_landmarks, width: int = 1280, height: int = 720) -> Image.Image:
    """Create stick figure from pose landmarks - optimized for speed"""
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_facecolor('black')
    
    if not pose_landmarks:
        # Return empty figure if no pose detected
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    
    # Extract landmark points
    points = []
    for lm in pose_landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
    
    # Draw torso (optimized)
    if len(points) > 24:
        l_shoulder, r_shoulder = points[11], points[12]
        l_hip, r_hip = points[23], points[24]
        
        # Torso polygon
        torso = Polygon([l_shoulder, r_shoulder, r_hip, l_hip],
                       closed=True, color='orange', alpha=0.5)
        ax.add_patch(torso)
        
        # Body centers
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        
        ax.add_patch(Circle(mid_shoulder, radius=8, color='red'))
        ax.add_patch(Circle(mid_hip, radius=8, color='purple'))
        
        # Neck (simplified)
        head = points[0]
        neck_end = (head[0] + (mid_shoulder[0] - head[0]) * 0.5,
                   head[1] + (mid_shoulder[1] - head[1]) * 0.5)
        ax.add_line(Line2D([head[0], neck_end[0]], [head[1], neck_end[1]], 
                          linewidth=4, color='brown'))
    
    # Draw pose connections
    for pair in POSE_PAIRS:
        a, b = pair
        if a < len(points) and b < len(points):
            ax.add_line(Line2D([points[a][0], points[b][0]], [points[a][1], points[b][1]],
                              linewidth=6, color='red'))
            ax.add_patch(Circle(points[a], radius=4, color='black'))
            ax.add_patch(Circle(points[b], radius=4, color='black'))
    
    # Draw head
    if len(points) > 0:
        head = points[0]
        head_radius = height * 0.04
        head_circle = Circle(head, radius=head_radius, fill=False, color='blue', linewidth=3)
        ax.add_patch(head_circle)
        
        # Eyes
        eye_offset_x = head_radius * 0.3
        eye_offset_y = head_radius * 0.3
        ax.plot(head[0] - eye_offset_x, head[1] - eye_offset_y, 'bo', markersize=4)
        ax.plot(head[0] + eye_offset_x, head[1] - eye_offset_y, 'bo', markersize=4)
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def process_video_optimized(video_path: str) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """Process video with optimized pipeline - no intermediate files"""
    pose = init_mediapipe()
    
    # Check if MediaPipe initialization failed
    if pose is None:
        st.error("❌ MediaPipe initialization failed. Cannot process video.")
        return [], []
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file")
        return [], []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"📹 Processing {total_frames} frames at {fps:.1f} fps")
    
    original_frames = []
    stick_figures = []
    
    # Process with optimized frame skipping
    frame_skip = max(1, int(fps // 10))  # Process ~10 frames per second for speed
    progress_bar = st.progress(0)
    
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for optimization
        if frame_count % frame_skip != 0:
            continue
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Process frame with error handling
        try:
            frame = cv2.flip(frame, 1)  # Mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe pose detection with error handling
            results = pose.process(frame_rgb)
            
            # Create stick figure
            stick_figure = create_stick_figure(results.pose_landmarks)
            
            # Store results
            original_frames.append(frame_rgb)
            stick_figures.append(stick_figure)
            processed_count += 1
            
        except Exception as frame_error:
            st.warning(f"⚠️ Error processing frame {frame_count}: {str(frame_error)}")
            # Skip this frame and continue
            continue
    
    cap.release()
    progress_bar.progress(1.0)
    
    st.success(f"✅ Processed {processed_count} frames successfully!")
    return original_frames, stick_figures

def create_video_from_frames(frames: List[Image.Image], output_path: str, fps: float = 10):
    """Create video from stick figure frames"""
    if not frames:
        return False
    
    # Get frame dimensions
    first_frame = frames[0]
    width, height = first_frame.size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert PIL to OpenCV format
        frame_array = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True

# Main UI
uploaded_file = st.file_uploader(
    "📹 **Upload your dance video** (MP4, MOV, AVI)",
    type=['mp4', 'mov', 'avi'],
    help="Upload a video file to convert to stick figure animation"
)

if uploaded_file is not None:
    # Show file info
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
    st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size:.1f} MB)")
    
    if file_size > 100:  # MB
        st.warning("⚠️ Large file detected. Processing may take longer. Consider using a shorter video for faster results.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name
    
    # Process button
    if st.button("🎭 Create Stick Figure Animation", type="primary"):
        with st.spinner("Processing video... This may take a moment"):
            try:
                # Check MediaPipe availability first
                test_pose = init_mediapipe()
                if test_pose is None:
                    st.error("❌ MediaPipe is not available in this environment")
                    st.info("💡 **Deployment Issue**: This appears to be a hosting platform limitation.")
                    st.info("🔧 **Solutions:**")
                    st.info("- Try a different hosting platform (Heroku, Railway, etc.)")
                    st.info("- Run the app locally with: `streamlit run streamlit_app.py`")
                    st.info("- Use the optimized_app.py version which may be more compatible")
                else:
                    # Process video
                    original_frames, stick_figures = process_video_optimized(temp_video_path)
                    
                    if original_frames and stick_figures:
                        st.success("✅ Video processed successfully!")
                        
                        # Store in session state for display
                        st.session_state.original_frames = original_frames
                        st.session_state.stick_figures = stick_figures
                        st.session_state.current_frame = 0
                        
                        # Create downloadable video
                        output_path = tempfile.mktemp(suffix='.mp4')
                        if create_video_from_frames(stick_figures, output_path):
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Stick Figure Video",
                                    data=f.read(),
                                    file_name=f"stick_figure_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                            os.unlink(output_path)
                    else:
                        st.error("❌ Failed to process video")
                    
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_video_path)
                except:
                    pass

# Display results if available
if 'original_frames' in st.session_state and 'stick_figures' in st.session_state:
    st.markdown("---")
    st.markdown("### 🎬 Animation Preview")
    
    original_frames = st.session_state.original_frames
    stick_figures = st.session_state.stick_figures
    
    if original_frames:
        total_frames = len(original_frames)
        current_frame = st.session_state.get('current_frame', 0)
        
        # Frame slider
        selected_frame = st.slider(
            f"Frame ({total_frames} total)",
            0, total_frames - 1, 
            current_frame,
            help="Navigate through the animation frames"
        )
        st.session_state.current_frame = selected_frame
        
        # Display frames side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Original")
            st.image(original_frames[selected_frame], use_container_width=True)
        
        with col2:
            st.markdown("#### 🎭 Stick Figure")
            st.image(stick_figures[selected_frame], use_container_width=True)
        
        # Navigation buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⏮️ First") and selected_frame > 0:
                st.session_state.current_frame = 0
                st.rerun()
        
        with col2:
            if st.button("⬅️ Previous") and selected_frame > 0:
                st.session_state.current_frame = selected_frame - 1
                st.rerun()
        
        with col3:
            if st.button("Next ➡️") and selected_frame < total_frames - 1:
                st.session_state.current_frame = selected_frame + 1
                st.rerun()
        
        with col4:
            if st.button("Last ⏭️") and selected_frame < total_frames - 1:
                st.session_state.current_frame = total_frames - 1
                st.rerun()
        
        # Auto-play option
        if st.checkbox("🔄 Auto-play animation"):
            if st.button("▶️ Start Auto-play"):
                placeholder = st.empty()
                for i in range(total_frames):
                    with placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_frames[i], caption=f"Original Frame {i+1}")
                        with col2:
                            st.image(stick_figures[i], caption=f"Stick Figure {i+1}")
                    st.session_state.current_frame = i
                    time.sleep(0.2)  # Adjust speed
        
        # Clear results
        if st.button("🗑️ Clear Results"):
            for key in ['original_frames', 'stick_figures', 'current_frame']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Performance info
if 'original_frames' not in st.session_state:
    st.markdown("---")
    st.markdown("### ⚡ Optimized Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Performance:**
        - Lightweight MediaPipe model
        - Optimized frame processing
        - Memory efficient pipeline
        - Fast stick figure generation
        """)
    
    with col2:
        st.markdown("""
        **🎯 Features:**
        - Real-time pose detection
        - Smooth animations
        - Download capability
        - Frame navigation
        """)
    
    with col3:
        st.markdown("""
        **📱 Compatibility:**
        - Python 3.10 optimized
        - Cross-platform support
        - Minimal dependencies
        - Web deployment ready
        """)

st.markdown("---")
st.markdown("**💡 Tip:** For best results, use videos with clear body movements and good lighting!")
