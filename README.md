# Dance Stick Figure Generator 🕺💃

An optimized Streamlit application that transforms dance videos into stick figure animations using MediaPipe pose estimation.

## 🚀 Features

- **Video Upload**: Upload dance videos in various formats (MP4, AVI, MOV)
- **Live Recording**: Record videos directly in the browser using WebRTC
- **Pose Detection**: High-performance MediaPipe pose estimation
- **Stick Figure Animation**: Clean stick figure visualization with smooth animations
- **Real-time Processing**: Optimized pipeline for fast processing
- **Multiple Modes**: 
  - Standard stick figures
  - Hand tracking
  - Face landmarks

## 📦 Installation

### Requirements
- Python 3.10+ (recommended for optimal performance)
- Webcam (for live recording features)

### Setup
```bash
# Clone or download the project
cd DanceStickFigure

# Install dependencies
pip install -r requirements.txt

# Run the optimized app
streamlit run optimized_app.py
```

## 🎯 Usage

### Main App (optimized_app.py)
The lightweight main application with optimized processing:
```bash
streamlit run optimized_app.py
```

### Additional Features
Access advanced features through the sidebar:
- **Live Version**: Real-time video recording and processing
- **Face Analysis**: Detailed face and hand landmark detection

## 🏗️ Project Structure

```
DanceStickFigure/
├── optimized_app.py          # Main streamlit app
├── requirements.txt          # Minimal dependencies for Python 3.10
├── README.md                 # This file
├── cleanup.py                # Project optimization script
├── input_videos/             # Upload directory
├── output_videos/            # Generated animations
└── pages/                    # Additional features
    ├── live_version.py       # Live video recording
    └── faces.py              # Face analysis
```

## ⚡ Performance Optimizations

The project has been optimized for:
- **Faster Processing**: In-memory operations, reduced file I/O
- **Lower Memory Usage**: Optimized MediaPipe settings
- **Better Compatibility**: Python 3.10 version constraints
- **Cleaner Codebase**: Removed unnecessary files and dependencies

### Optimization Features:
- Direct frame processing without intermediate files
- Optimized MediaPipe model settings
- Streamlined dependency list
- Efficient video encoding
- Real-time visualization

## 🔧 Technical Details

### Dependencies
- **streamlit**: Web interface (≥1.28.0, <2.0.0)
- **mediapipe**: Pose estimation (<0.11.0)
- **opencv-python**: Video processing (<5.0.0)
- **streamlit-webrtc**: Live video recording
- **matplotlib**: Visualization (<4.0.0)

### MediaPipe Configuration
```python
# Optimized settings for performance
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # Balanced accuracy/speed
    enable_segmentation=False, # Disabled for speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## 🎨 Stick Figure Visualization

The application creates smooth stick figure animations by:
1. Detecting pose landmarks using MediaPipe
2. Connecting key body points with lines
3. Creating frame-by-frame animations
4. Outputting as MP4 videos

### Supported Poses
- Full body pose (33 landmarks)
- Hand tracking (21 landmarks per hand)
- Face landmarks (468 points)

## 🌐 Deployment

### Local Development
```bash
streamlit run optimized_app.py
```

### Production Deployment
For production deployment, consider:
- HTTPS requirement for WebRTC features
- Server resources for video processing
- Python 3.10 environment setup

## 📋 Troubleshooting

### Common Issues

1. **WebRTC not working**: Ensure HTTPS and proper server configuration
2. **Memory issues**: Reduce video resolution or length
3. **Import errors**: Check Python version compatibility (3.10+ recommended)

### Performance Tips
- Use shorter videos (< 30 seconds) for faster processing
- Lower video resolution for real-time performance

## 🤝 Contributing

To contribute to this project:
1. Use the optimized codebase (`optimized_app.py`)
2. Follow Python 3.10 compatibility guidelines
3. Test with various video formats
4. Maintain the lightweight structure

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🎯 Next Steps

- [ ] Add more stick figure styles
- [ ] Implement batch processing
- [ ] Add export options (GIF, different formats)
- [ ] Improve mobile compatibility
- [ ] Add pose comparison features

---

**Note**: This is an optimized version focused on performance and simplicity. The original files are preserved for reference.
