import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

# Configuration
MODEL_PATH = r"C:\Users\Arwa\webcam script\rps_model.h5"
LABELS_PATH = r"C:\Users\Arwa\webcam script\class_labels.json"
IMG_SIZE = 224

# Load the trained model
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load class labels
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)
    class_labels = {int(k): v for k, v in class_labels.items()}

print(f"Classes: {class_labels}")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Colors for visualization (BGR format)
colors = {
    'rock': (0, 0, 255),
    'paper': (255, 0, 0),
    'scissors': (0, 255, 0)
}

print("\n" + "="*50)
print("WEBCAM STARTED WITH HAND TRACKING!")
print("="*50)
print("Instructions:")
print("- The box will follow your hand automatically")
print("- Use good lighting for best results")
print("- Press 'q' to quit")
print("="*50 + "\n")

# Background subtractor for hand detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Smooth predictions
prediction_history = []
history_size = 5

def detect_hand(frame):
    """Detect hand position using skin color detection"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        if cv2.contourArea(largest_contour) > 5000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand box to be square and larger
            size = max(w, h) + 100
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate box coordinates
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(frame.shape[1], x1 + size)
            y2 = min(frame.shape[0], y1 + size)
            
            return (x1, y1, x2 - x1, y2 - y1)
    
    return None

# Default ROI position
roi_x = 170
roi_y = 90
roi_size = 300

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect hand
    hand_box = detect_hand(frame)
    
    if hand_box:
        # Update ROI to follow hand
        hx, hy, hw, hh = hand_box
        # Smooth transition
        target_x = hx
        target_y = hy
        target_size = hw
        
        roi_x = int(roi_x * 0.7 + target_x * 0.3)
        roi_y = int(roi_y * 0.7 + target_y * 0.3)
        roi_size = int(roi_size * 0.7 + target_size * 0.3)
    
    # Ensure ROI is within frame bounds
    roi_x = max(0, min(roi_x, frame.shape[1] - roi_size))
    roi_y = max(0, min(roi_y, frame.shape[0] - roi_size))
    
    # Extract ROI
    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    
    if roi.size == 0:
        continue
    
    # Preprocess the ROI for prediction
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Smooth predictions using history
    prediction_history.append(predictions)
    if len(prediction_history) > history_size:
        prediction_history.pop(0)
    
    # Average predictions
    avg_predictions = np.mean(prediction_history, axis=0)
    predicted_class = np.argmax(avg_predictions)
    confidence = avg_predictions[predicted_class]
    label = class_labels[predicted_class]
    
    # Draw ROI rectangle
    color = colors.get(label, (255, 255, 255))
    cv2.rectangle(frame, (roi_x, roi_y), 
                  (roi_x+roi_size, roi_y+roi_size), color, 3)
    
    # Draw corner markers for better visibility
    corner_length = 30
    # Top-left
    cv2.line(frame, (roi_x, roi_y), (roi_x + corner_length, roi_y), color, 5)
    cv2.line(frame, (roi_x, roi_y), (roi_x, roi_y + corner_length), color, 5)
    # Top-right
    cv2.line(frame, (roi_x+roi_size, roi_y), (roi_x+roi_size - corner_length, roi_y), color, 5)
    cv2.line(frame, (roi_x+roi_size, roi_y), (roi_x+roi_size, roi_y + corner_length), color, 5)
    # Bottom-left
    cv2.line(frame, (roi_x, roi_y+roi_size), (roi_x + corner_length, roi_y+roi_size), color, 5)
    cv2.line(frame, (roi_x, roi_y+roi_size), (roi_x, roi_y+roi_size - corner_length), color, 5)
    # Bottom-right
    cv2.line(frame, (roi_x+roi_size, roi_y+roi_size), (roi_x+roi_size - corner_length, roi_y+roi_size), color, 5)
    cv2.line(frame, (roi_x+roi_size, roi_y+roi_size), (roi_x+roi_size, roi_y+roi_size - corner_length), color, 5)
    
    # Display main prediction with background
    text = f"{label.upper()}: {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (10, 10), 
                  (text_width + 20, text_height + 20), 
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (15, 15 + text_height), 
                font, font_scale, color, thickness)
    
    # Draw instructions
    cv2.putText(frame, "Press 'q' to quit | Box follows your hand", 
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw all predictions (probability bars)
    y_offset = 80
    for i, (class_idx, class_name) in enumerate(class_labels.items()):
        prob = avg_predictions[class_idx] * 100
        bar_width = int(prob * 2)
        class_color = colors.get(class_name, (255, 255, 255))
        
        # Draw probability bar background
        cv2.rectangle(frame, (10, y_offset + i*40), 
                     (210, y_offset + i*40 + 25), 
                     (50, 50, 50), -1)
        
        # Draw probability bar
        cv2.rectangle(frame, (10, y_offset + i*40), 
                     (10 + bar_width, y_offset + i*40 + 25), 
                     class_color, -1)
        
        # Draw border
        cv2.rectangle(frame, (10, y_offset + i*40), 
                     (210, y_offset + i*40 + 25), 
                     class_color, 2)
        
        # Draw class name and percentage
        cv2.putText(frame, f"{class_name}: {prob:.1f}%", 
                   (220, y_offset + i*40 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Rock Paper Scissors Classifier', frame)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n" + "="*50)
print("Webcam closed. Goodbye!")
print("="*50)