import cv2
import yaml
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional

class LiftMonitor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Lift Monitoring System."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize camera
        self.cap = self._initialize_camera()
        
        # Load YOLO model
        self.model = YOLO(self.config['detection']['model'])
        
        # Track people
        self.people_in_lift = []
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds
        
        # Frame skipping for better performance
        self.frame_skip = 2  # Process every other frame by default
        self.frame_count = 0
        self.last_detections = []
    
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
        
        # Reset people in lift
        self.people_in_lift = []
        
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
                color = (0, 255, 0)  # Green
                label = "Correct"
                correct_side += 1
            else:
                # Person is on the wrong side (left)
                color = (0, 0, 255)  # Red
                label = "Wrong Side"
                wrong_side += 1
            
            # Draw a square around the detection
            size = max(x2 - x1, y2 - y1)  # Use the larger dimension
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = size // 2
            
            # Calculate square coordinates
            x1_sq = max(0, center_x - half_size)
            y1_sq = max(0, center_y - half_size)
            x2_sq = min(frame.shape[1] - 1, center_x + half_size)
            y2_sq = min(frame.shape[0] - 1, center_y + half_size)
            
            # Draw the square with the appropriate color
            cv2.rectangle(frame, 
                         (int(x1_sq), int(y1_sq)), 
                         (int(x2_sq), int(y2_sq)), 
                         color,  # Green for correct, Red for wrong
                         3)  # Thickness
            
            # Add label with background for better visibility
            label_text = f"PERSON - {label.upper()}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text (above the square)
            cv2.rectangle(frame, 
                         (x1, max(0, y1 - text_height - 15)), 
                         (x1 + text_width + 10, max(0, y1 - 5)), 
                         color, -1)
            
            # Add to people in lift
            self.people_in_lift.append({
                'bbox': (x1, y1, x2, y2),
                'position': (center_x, center_y),
                'side': 'right' if label == "Correct" else 'left'
            })
        
        return correct_side, wrong_side
    
    
    def _check_capacity(self, frame: np.ndarray) -> None:
        """Check if lift capacity is exceeded and trigger alerts if needed."""
        total_people = len(self.people_in_lift)
        max_capacity = self.config['alerts']['max_capacity']
        
        if total_people > max_capacity:
            # Visual alert
            cv2.putText(frame, "WARNING: CAPACITY EXCEEDED!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Update last alert time for cooldown
            current_time = time.time()
            self.last_alert_time = current_time
    
    def run(self):
        """Main loop for the lift monitoring system with frame skipping for better performance."""
        prev_time = 0
        
        print("Starting lift monitoring system...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            
            display_frame = frame.copy()
            
            # Process only every N-th frame (frame skipping)
            if self.frame_count % self.frame_skip == 0:
                # Process frame with YOLO
                results = self.model(frame, verbose=False)
                
                # Process detections
                correct_side, wrong_side = self._process_detections(frame, results)
                self.last_detections = (correct_side, wrong_side)
            else:
                # Use detections from the last processed frame
                correct_side, wrong_side = self.last_detections if hasattr(self, 'last_detections') else (0, 0)
            
            # Always draw UI elements for smooth display
            self._draw_zone_lines(display_frame)
            
            # Check capacity and trigger alerts (using the latest detections)
            self._check_capacity(display_frame)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Display FPS with frame skipping info
            if self.config['display']['show_fps']:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                
                # Show FPS and frame skip in bottom left corner with semi-transparent background
                font_scale = 0.7
                thickness = 2
                margin = 10
                
                # Get text sizes for background
                fps_text = f"FPS: {int(fps)}"
                skip_text = f"Frame Skip: {self.frame_skip}x"
                (fps_width, fps_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                (skip_width, skip_height), _ = cv2.getTextSize(skip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Calculate positions (bottom right)
                text_x = display_frame.shape[1] - max(fps_width, skip_width) - margin
                text_y1 = display_frame.shape[0] - margin - skip_height - 5  # Skip text
                text_y2 = display_frame.shape[0] - margin  # FPS text
                
                # Draw semi-transparent background
                overlay = display_frame.copy()
                bg_x1 = text_x - 5
                bg_y1 = text_y1 - fps_height - 5
                bg_x2 = display_frame.shape[1] - margin + 5
                bg_y2 = text_y2 + 5
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                # Draw FPS and frame skip text
                cv2.putText(display_frame, fps_text, (text_x, text_y2),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                cv2.putText(display_frame, skip_text, (text_x, text_y1),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # Display status
            total_people = correct_side + wrong_side
            status_text = f"People: {total_people} (Correct: {correct_side}, Wrong: {wrong_side})"
            cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Ensure we have a valid frame to display
            if display_frame is not None and display_frame.size > 0:
                # Show the frame in the main window
                cv2.imshow("Lift Monitoring System", display_frame)
                
                # Check if window is still open
                if cv2.getWindowProperty("Lift Monitoring System", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+'):  # Increase frame skip
                    self.frame_skip = min(3, self.frame_skip + 1)
                    print(f"Frame skip set to: {self.frame_skip}")
                elif key == ord('-'):  # Decrease frame skip
                    self.frame_skip = max(1, self.frame_skip - 1)
                    print(f"Frame skip set to: {self.frame_skip}")
            else:
                print("Error: Invalid frame to display")
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    monitor = LiftMonitor()
    monitor.run()
