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
        
        # Draw semi-transparent background for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (left_zone['x1'], 10), 
                     (left_zone['x2'], 10 + left_text_size[1] + 20), 
                     (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the text
        cv2.putText(frame, left_text, 
                   (left_text_x, left_text_y + left_text_size[1]), 
                   font, font_scale, (0, 0, 255), font_thickness)
        
        # Right zone label (Correct)
        right_text = "CORRECT ZONE"
        right_text_size = cv2.getTextSize(right_text, font, font_scale, font_thickness)[0]
        right_text_x = (right_zone['x1'] + right_zone['x2'] - right_text_size[0]) // 2
        
        # Draw semi-transparent background for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (right_zone['x1'], 10), 
                     (right_zone['x2'], 10 + right_text_size[1] + 20), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the text
        cv2.putText(frame, right_text, 
                   (right_text_x, left_text_y + right_text_size[1]), 
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
            
            # Draw bounding box with thicker border and rounded corners
            box_thickness = 3
            corner_radius = 10
            
            # Draw main rectangle with rounded corners
            cv2.rectangle(frame, (x1 + corner_radius, y1), 
                         (x2 - corner_radius, y2), color, box_thickness)
            cv2.rectangle(frame, (x1, y1 + corner_radius), 
                         (x1 + corner_radius, y2 - corner_radius), color, -1)
            cv2.rectangle(frame, (x2 - corner_radius, y1 + corner_radius), 
                         (x2, y2 - corner_radius), color, -1)
            
            # Draw corner circles for rounded effect
            cv2.circle(frame, (x1 + corner_radius, y1 + corner_radius), 
                      corner_radius, color, box_thickness)
            cv2.circle(frame, (x2 - corner_radius, y1 + corner_radius), 
                      corner_radius, color, box_thickness)
            cv2.circle(frame, (x1 + corner_radius, y2 - corner_radius), 
                      corner_radius, color, box_thickness)
            cv2.circle(frame, (x2 - corner_radius, y2 - corner_radius), 
                      corner_radius, color, box_thickness)
            
            # Add label with background for better visibility
            label_text = f"PERSON - {label.upper()}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 15), 
                         (x1 + text_width + 10, y1 - 5), 
                         color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, 
                       (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
                
                # Show FPS and processing info
                cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frame Skip: {self.frame_skip}x", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
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
            elif key == ord('+'):  # Increase frame skip
                self.frame_skip = min(3, self.frame_skip + 1)
                print(f"Frame skip set to: {self.frame_skip}")
            elif key == ord('-'):  # Decrease frame skip
                self.frame_skip = max(1, self.frame_skip - 1)
                print(f"Frame skip set to: {self.frame_skip}")
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        if self.config['alerts']['audio']['enabled']:
            pygame.quit()

if __name__ == "__main__":
    monitor = LiftMonitor()
    monitor.run()
