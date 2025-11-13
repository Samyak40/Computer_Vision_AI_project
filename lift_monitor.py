import cv2
import yaml
import numpy as np
import time
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional

class LiftMonitor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Lift Monitoring System.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize camera
        self.cap = self._initialize_camera()
        
        # Load YOLO model (must be done before setting threads)
        self.model = YOLO(self.config['detection']['model'])
        
        # Configure multithreading (after model is loaded)
        self._setup_multithreading(initial_setup=True)
        
        # Track people
        self.people_in_lift = []
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds
        
        # Frame skipping for better performance
        self.frame_skip = 2  # Process every other frame by default
        self.frame_count = 0
        self.last_detections = []
        self.last_boxes = []  # Store boxes for smooth display
    
    def _setup_multithreading(self, initial_setup: bool = False) -> None:
        """Configure multithreading for OpenCV and PyTorch.
        
        Args:
            initial_setup: If True, will also set PyTorch interop threads (can only be done once)
        """
        # Use 75% of available cores for better system responsiveness
        num_threads = max(1, int(os.cpu_count() * 0.75))
        
        print(f"Configuring multithreading with {num_threads} threads...")
        
        # Configure OpenCV threading
        cv2.setNumThreads(num_threads)
        print(f"OpenCV threads set to: {cv2.getNumThreads()}")
        
        # Configure PyTorch threading
        try:
            torch.set_num_threads(num_threads)
            
            # Interop threads can only be set once, before any parallel work starts
            if initial_setup or not hasattr(self, '_pytorch_initialized'):
                torch.set_num_interop_threads(num_threads)
                self._pytorch_initialized = True
                
            print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
            print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
        except RuntimeError as e:
            print(f"Warning: Could not set PyTorch threads: {e}")
            print("Continuing with current thread configuration...")
        
        # Enable OpenCV optimizations
        cv2.setUseOptimized(True)
        print(f"OpenCV optimizations enabled: {cv2.useOptimized()}")
        
        # Store thread count for reference
        self.num_threads = num_threads
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_camera(self):
        """Initialize the camera with configured settings."""
        cap = cv2.VideoCapture(self.config['camera']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        return cap
    
    def _draw_zone_lines(self, frame: np.ndarray) -> None:
        """Draw zone lines and labels on the frame."""
        if not self.config['display']['show_zone_lines']:
            return
            
        left_zone = self.config['zones']['left_zone']
        right_zone = self.config['zones']['right_zone']
        color = self.config['display']['zone_line_color']
        thickness = self.config['display']['zone_line_thickness']
        
        # Draw vertical line separating left and right zones
        cv2.line(frame, 
                (left_zone['x2'], left_zone['y1']), 
                (left_zone['x2'], left_zone['y2']), 
                color, thickness)
        
        # Add zone labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        # Left zone label (Incorrect)
        left_text = "INCORRECT ZONE"
        left_text_size = cv2.getTextSize(left_text, font, font_scale, font_thickness)[0]
        left_text_x = (left_zone['x1'] + left_zone['x2'] - left_text_size[0]) // 2
        left_text_y = 50  # 50 pixels from top
        
        # Calculate background rectangle dimensions for left zone
        bg_padding = 10
        bg_x1 = left_zone['x1']
        bg_y1 = left_text_y - left_text_size[1] - bg_padding
        bg_x2 = left_zone['x2']
        bg_y2 = left_text_y + bg_padding
        
        # Draw semi-transparent background for left zone label
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (bg_x1, bg_y1), 
                     (bg_x2, bg_y2), 
                     (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the left zone text
        cv2.putText(frame, left_text, 
                   (left_text_x, left_text_y), 
                   font, font_scale, (0, 0, 255), font_thickness)
        
        # Right zone label (Correct)
        right_text = "CORRECT ZONE"
        right_text_size = cv2.getTextSize(right_text, font, font_scale, font_thickness)[0]
        right_text_x = (right_zone['x1'] + right_zone['x2'] - right_text_size[0]) // 2
        
        # Calculate background rectangle dimensions for right zone
        bg_x1 = right_zone['x1']
        bg_y1 = left_text_y - right_text_size[1] - bg_padding
        bg_x2 = right_zone['x2']
        bg_y2 = left_text_y + bg_padding
        
        # Draw semi-transparent background for right zone label
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (bg_x1, bg_y1), 
                     (bg_x2, bg_y2), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the right zone text
        cv2.putText(frame, right_text, 
                   (right_text_x, left_text_y), 
                   font, font_scale, (0, 255, 0), font_thickness)
    
    def _process_detections(self, frame: np.ndarray, results) -> Tuple[int, int]:
        """Process detection results and update people tracking."""
        correct_side = 0
        wrong_side = 0
        
        # Get detection boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Reset people in lift and boxes
        self.people_in_lift = []
        self.last_boxes = []  # Clear previous boxes
        
        for box, conf in zip(boxes, confidences):
            if conf < self.config['detection']['confidence_threshold']:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check which zone the person is in
            if (self.config['zones']['right_zone']['x1'] <= center_x <= self.config['zones']['right_zone']['x2'] and
                self.config['zones']['right_zone']['y1'] <= center_y <= self.config['zones']['right_zone']['y2']):
                # Person is on the correct side (right)
                label = "Correct"
                correct_side += 1
            else:
                # Person is on the wrong side (left)
                label = "Wrong Side"
                wrong_side += 1
            
            # Calculate square coordinates
            size = max(x2 - x1, y2 - y1)  # Use the larger dimension
            half_size = size // 2
            
            x1_sq = max(0, center_x - half_size)
            y1_sq = max(0, center_y - half_size)
            x2_sq = min(frame.shape[1] - 1, center_x + half_size)
            y2_sq = min(frame.shape[0] - 1, center_y + half_size)
            
            # Store box info for drawing later
            self.last_boxes.append({
                'x1_sq': int(x1_sq),
                'y1_sq': int(y1_sq),
                'x2_sq': int(x2_sq),
                'y2_sq': int(y2_sq),
                'x1': x1,
                'y1': y1,
                'label': label
            })
            
            # Add to people in lift
            self.people_in_lift.append({
                'bbox': (x1, y1, x2, y2),
                'position': (center_x, center_y),
                'side': 'right' if label == "Correct" else 'left'
            })
        
        return correct_side, wrong_side
    
    
    def _draw_boxes(self, frame: np.ndarray) -> None:
        """Draw bounding boxes on the frame using stored box information."""
        for box_info in self.last_boxes:
            # Draw the green square
            cv2.rectangle(frame, 
                         (box_info['x1_sq'], box_info['y1_sq']), 
                         (box_info['x2_sq'], box_info['y2_sq']), 
                         (0, 255, 0),  # Green
                         3)  # Thickness
            
            # Add label with background for better visibility
            label_text = f"PERSON - {box_info['label'].upper()}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text (above the square)
            cv2.rectangle(frame, 
                         (box_info['x1'], max(0, box_info['y1'] - text_height - 15)), 
                         (box_info['x1'] + text_width + 10, max(0, box_info['y1'] - 5)), 
                         (0, 255, 0), -1)
            
            # Draw the text
            cv2.putText(frame, label_text,
                       (box_info['x1'] + 5, max(0, box_info['y1'] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _check_capacity(self, frame: np.ndarray) -> None:
        """Check if lift capacity is exceeded and trigger alerts if needed."""
        total_people = len(self.people_in_lift)
        max_capacity = self.config['alerts']['max_capacity']
        
        if total_people > max_capacity:
            # Visual alert - centered on screen
            warning_text = "WARNING: CAPACITY EXCEEDED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            
            # Get text size to calculate center position
            (text_width, text_height), _ = cv2.getTextSize(warning_text, font, font_scale, font_thickness)
            
            # Calculate center position
            frame_height, frame_width = frame.shape[:2]
            text_x = (frame_width - text_width) // 2
            text_y = (frame_height + text_height) // 2
            
            # Draw semi-transparent background for better visibility
            overlay = frame.copy()
            padding = 20
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_height - padding
            bg_x2 = text_x + text_width + padding
            bg_y2 = text_y + padding
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw the warning text
            cv2.putText(frame, warning_text, (text_x, text_y),
                       font, font_scale, (0, 0, 255), font_thickness)
            
            # Update last alert time for cooldown
            current_time = time.time()
            self.last_alert_time = current_time
    
    def _process_frame(self, frame):
        """Process a single frame for detection and drawing."""
        display_frame = frame.copy()
        
        # Process only every N-th frame (frame skipping)
        if self.frame_count % self.frame_skip == 0:
            # Process frame with YOLO
            results = self.model(frame, verbose=False)

            # Process detections (stores boxes in self.last_boxes)
            correct_side, wrong_side = self._process_detections(frame, results)
            self.last_detections = (correct_side, wrong_side)
        else:
            # Use detections from the last processed frame
            correct_side, wrong_side = self.last_detections if hasattr(self, 'last_detections') else (0, 0)
        
        # Draw UI elements
        self._draw_zone_lines(display_frame)
        self._draw_boxes(display_frame)
        self._check_capacity(display_frame)
        
        # Increment frame counter
        self.frame_count += 1
        
        return display_frame, correct_side, wrong_side

    def cleanup(self):
        """Release resources and clean up."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")

    def run(self):
        """Run the main application loop."""
        try:
            prev_time = time.time()
            self.prev_time = prev_time  # Store as instance variable for FPS calculation
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process the frame
                display_frame, correct_side, wrong_side = self._process_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                
                # Display FPS and other info
                if self.config['display']['show_fps']:
                    font_scale = 0.7
                    thickness = 2
                    
                    # Prepare text
                    fps_text = f"FPS: {int(fps)}"
                    skip_text = f"Frame Skip: {self.frame_skip}x"
                    thread_text = f"Threads: {torch.get_num_threads()}"
                    
                    # Draw FPS and settings info
                    y_offset = 30
                    # Draw FPS and settings info
                    y_offset = 30
                    cv2.putText(display_frame, fps_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    cv2.putText(display_frame, skip_text, (10, y_offset + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
                    cv2.putText(display_frame, thread_text, (10, y_offset + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 128, 0), thickness)
                
                # Display status
                total_people = correct_side + wrong_side
                status_text = f"People: {total_people} (Correct: {correct_side}, Wrong: {wrong_side})"
                cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow("Lift Monitoring System", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):  # Cycle frame skipping between 1x, 2x, and 3x
                    self.frame_skip = (self.frame_skip % 3) + 1
                    print(f"Frame skip set to: {self.frame_skip}x")
                
                # Check if window is still open
                if cv2.getWindowProperty("Lift Monitoring System", cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

if __name__ == "__main__":
    # You can specify custom number of threads or leave as None for auto-detection
    # Example: monitor = LiftMonitor(num_threads=4)
    monitor = LiftMonitor(num_threads=None)  # Auto-detect optimal thread count
    print("Controls:")
    print("  'm' - Toggle OpenMP multithreading")
    print("  'f'  - Toggle frame skipping (1/2)")
    print("  'q'  - Quit")
    
    # Start the main loop
    monitor.run()
