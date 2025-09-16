# Lift Monitoring System

A computer vision-based system that monitors lift (elevator) usage, ensuring proper etiquette and capacity limits are followed.

## Features

- Real-time person detection using YOLOv8
- Zone detection for proper standing position (left/right side)
- People counting with capacity limit enforcement
- Visual and audio alerts for violations
- Configurable settings via YAML file
- Real-time FPS counter for performance monitoring

## Prerequisites

- Python 3.8 or higher
- Webcam or IP camera
- NVIDIA GPU (recommended for better performance)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd lift-monitoring-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit the `config.yaml` file to adjust the following settings:

- Camera source (0 for default webcam or video file path)
- Zone boundaries (left/right side of the lift)
- Maximum capacity limit
- Alert preferences (visual/audio)
- Display settings

## Usage

1. Ensure your camera is connected and working
2. Run the application:
   ```bash
   python lift_monitor.py
   ```
3. The system will display the camera feed with detections and alerts
4. Press 'q' to quit the application

## Keyboard Controls

- `q`: Quit the application

## Customization

### Changing Alert Sounds

1. Place your custom sound file (WAV format) in the project directory
2. Update the `warning_sound` path in `config.yaml`

### Adjusting Detection Parameters

Modify the following in `config.yaml`:
- `detection/confidence_threshold`: Adjust sensitivity of person detection (0-1)
- `detection/nms_threshold`: Adjust non-maximum suppression threshold (0-1)
- `zones/left_zone` and `zones/right_zone`: Update coordinates to match your lift's layout

## Performance Tips

- For better performance, use a smaller YOLO model (e.g., yolov8n.pt)
- Reduce the camera resolution if experiencing lag
- Ensure proper lighting conditions for better detection accuracy

## Troubleshooting

- **No camera feed**: Check if the camera is properly connected and not being used by another application
- **Low FPS**: Try reducing the camera resolution or using a smaller YOLO model
- **Incorrect detections**: Adjust the confidence threshold in the config file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV for computer vision processing
- Pygame for audio alerts
