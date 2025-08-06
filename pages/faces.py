import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Face + Hands Detail", layout="wide")
st.title("🧠 Advanced Face + Hands")
st.markdown("Detailed hand and face landmark rendering. No body stick figure.")
st.info("🌐 This now works in deployed versions using WebRTC!")

# Initialize MediaPipe Holistic with face refinement
mp_holistic = mp.solutions.holistic

OUTPUT_W, OUTPUT_H = 640, 480  # Reduced resolution for better performance

class FaceHandsProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Reduced from 1 to 0 for better performance
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror the image
        
        # Resize input for faster processing
        height, width = img.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Holistic
        results = self.holistic.process(img_rgb)
        
        # Create face and hands visualization
        face_hands_img = self.draw_face_and_hands(results, img.shape)
        
        return av.VideoFrame.from_ndarray(face_hands_img, format="bgr24")

    def draw_face_and_hands(self, results, img_shape):
        # Use smaller figure size for better performance
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=80)  # Reduced size and DPI
        ax.set_xlim(0, OUTPUT_W)
        ax.set_ylim(0, OUTPUT_H)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_facecolor('black')

        def extract_points(landmark_list, width, height):
            if not landmark_list:
                return []
            return [(int(lm.x * width), int(lm.y * height)) for lm in landmark_list]

        # Extract landmarks
        face = extract_points(results.face_landmarks.landmark if results.face_landmarks else [], OUTPUT_W, OUTPUT_H)
        left_hand = extract_points(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], OUTPUT_W, OUTPUT_H)
        right_hand = extract_points(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], OUTPUT_W, OUTPUT_H)

        def draw_lines(points, pairs, color="white", linewidth=2):
            for a, b in pairs:
                if a < len(points) and b < len(points):
                    ax.add_line(Line2D([points[a][0], points[b][0]], [points[a][1], points[b][1]], 
                                     linewidth=linewidth, color=color))

        # Simplified hand connections for better performance
        HAND_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                      (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                      (0, 9), (9, 10), (10, 11), (11, 12)]  # Middle only (removed ring and pinky for performance)

        # Draw hands with different colors
        draw_lines(left_hand, HAND_PAIRS, color="lime", linewidth=2)  # Reduced linewidth
        draw_lines(right_hand, HAND_PAIRS, color="cyan", linewidth=2)

        # Draw hand landmarks (smaller circles)
        for pt in left_hand:
            ax.add_patch(Circle(pt, radius=3, color='lime'))  # Reduced radius
        for pt in right_hand:
            ax.add_patch(Circle(pt, radius=3, color='cyan'))

        # Draw face landmarks (smaller for performance)
        for i, pt in enumerate(face):
            if i % 3 == 0:  # Draw every 3rd point for performance
                ax.add_patch(Circle(pt, radius=1, color='orange'))  # Smaller radius

        # Enhanced eyeball tracking using iris
        if len(face) > 475:
            left_iris = face[468] if 468 < len(face) else None
            right_iris = face[473] if 473 < len(face) else None
            
            if left_iris:
                ax.add_patch(Circle(left_iris, radius=4, color='red'))  # Smaller radius
            if right_iris:
                ax.add_patch(Circle(right_iris, radius=4, color='red'))

        # Simplified face outline (fewer points for performance)
        if len(face) > 10:
            # Draw simplified face contour with fewer points
            face_oval_indices = [10, 151, 9, 8, 168, 6, 148, 176, 149, 150]  # Reduced points
            face_contour = [face[i] for i in face_oval_indices if i < len(face)]
            if len(face_contour) > 2:
                draw_lines(face_contour, [(i, i+1) for i in range(len(face_contour)-1)], color="yellow", linewidth=1)

        # Convert matplotlib to image (optimized)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black', dpi=80)  # Reduced DPI
        buf.seek(0)
        img_pil = Image.open(buf)
        img_array = np.array(img_pil)
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr

# WebRTC Configuration with multiple STUN/TURN servers for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ],
    "iceCandidatePoolSize": 10,
})

# Optimized media constraints for better performance
MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"min": 320, "ideal": 640, "max": 1280},
        "height": {"min": 240, "ideal": 480, "max": 720},
        "frameRate": {"min": 10, "ideal": 15, "max": 30},
    },
    "audio": False
}

st.markdown("### 🎥 Live Camera Feed")
st.markdown("📹 **Your live webcam feed:**")
st.warning("⚠️ **Network Tips**: If connection fails, try refreshing the page or check your internet connection.")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📷 Camera Input")
    try:
        camera_stream = webrtc_streamer(
            key="face-hands-camera",
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            async_processing=False,
        )
    except Exception as e:
        st.error(f"Camera connection failed: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser (Chrome/Firefox recommended)")

with col2:
    st.markdown("#### 🎭 Face & Hands Analysis")
    try:
        analysis_stream = webrtc_streamer(
            key="face-hands-analysis",
            video_processor_factory=FaceHandsProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            async_processing=True,
        )
    except Exception as e:
        st.error(f"Face/hands processing failed: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser (Chrome/Firefox recommended)")

st.markdown("---")
st.markdown("### 📊 Features:")
st.markdown("""
- 🟢 **Left Hand**: Lime green tracking (thumb, index, middle fingers)
- 🔵 **Right Hand**: Cyan blue tracking (thumb, index, middle fingers)  
- 🟠 **Face Landmarks**: Orange dots (optimized sampling)
- 🔴 **Eye Iris**: Red circles for precise eye tracking
- 🟡 **Face Contour**: Yellow outline (simplified)
""")

st.info("💡 **Tip**: Make sure your hands and face are well-lit for better detection!")

# Troubleshooting section
with st.expander("🔧 **Troubleshooting Connection Issues**"):
    st.markdown("""
    **If you see "Connection is taking longer than expected":**
    
    1. **Refresh the page** - Often fixes temporary connection issues
    2. **Use Chrome or Firefox** - Better WebRTC support than Safari/Edge
    3. **Check your internet** - Stable connection required for video streaming
    4. **Allow camera permissions** - Browser must have camera access
    5. **Disable VPN/Firewall** - May block WebRTC connections
    6. **Try incognito mode** - Bypasses browser extensions that might interfere
    
    **Network Requirements:**
    - Stable internet connection (minimum 1 Mbps upload)
    - Unrestricted UDP traffic
    - Browser with WebRTC support
    
    **Performance Notes:**
    - Optimized for face and hand detection
    - Reduced resolution for better performance
    - Simplified hand tracking (thumb, index, middle fingers)
    """)

st.markdown("---")
st.markdown("**🌐 Optimized for deployed versions** - Uses multiple relay servers for better connectivity")

st.markdown("---")
st.markdown("### 📊 Features:")
st.markdown("""
- 🟢 **Right Hand**: Lime green tracking
- 🔵 **Left Hand**: Cyan blue tracking  
- 🟠 **Face Landmarks**: Orange dots
- 🔴 **Eye Iris**: Red circles for precise eye tracking
- 🟡 **Face Contour**: Yellow outline
""")

st.info("� **Tip**: Make sure your hands and face are well-lit for better detection!")









