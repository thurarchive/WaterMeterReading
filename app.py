from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import tempfile
import os

app = Flask(__name__)

# Load YOLO model
model = None
try:
    model = YOLO('my_trained_model.pt')
except Exception as e:
    print(f"Error loading model: {str(e)}")

def resize_maintain_aspect(image, target_size=800):
    """
    Resize image to target size while maintaining aspect ratio and padding if necessary
    """
    height, width = image.shape[:2]
    
    # Calculate aspect ratio
    aspect = width / height
    
    if aspect > 1:  # width > height
        new_width = target_size
        new_height = int(target_size / aspect)
    else:  # height >= width
        new_height = target_size
        new_width = int(target_size * aspect)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    # Place resized image in center of square canvas
    square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return square_img, (x_offset, y_offset, new_width, new_height)

def process_image(image):
    """
    Process a single image and return the detected digit sequence and confidence scores
    """
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    processed_image, (x_offset, y_offset, new_width, new_height) = resize_maintain_aspect(image)
    
    # Perform inference
    results = model(processed_image)
    
    # Process the results
    detected_digits = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if confidence > 0.5:  # Confidence threshold
                # Convert coordinates back to original image scale
                scale_x = original_width / new_width
                scale_y = original_height / new_height
                
                # Remove padding offset and scale back to original image size
                x1 = (x1 - x_offset) * scale_x
                x2 = (x2 - x_offset) * scale_x
                y1 = (y1 - y_offset) * scale_y
                y2 = (y2 - y_offset) * scale_y
                
                # Store detection for sorting
                detected_digits.append({
                    'digit': class_id,
                    'confidence': confidence,
                    'bbox': [int(coord) for coord in [x1, y1, x2, y2]]
                })

    # Sort digits by x-coordinate to get the sequence
    detected_digits.sort(key=lambda x: x['bbox'][0])
    
    # Create digit sequence
    digit_sequence = ''.join([str(digit['digit']) for digit in detected_digits])
    
    return {
        'reading': digit_sequence,
        'digits': detected_digits
    }

@app.route("/")
def get_app_name():
    return jsonify({
        "name": "Water Meter Reading API",
        "version": "1.0",
        "status": "running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            "error": "Model not loaded",
            "message": "The YOLO model failed to load. Please check server logs."
        }), 500

    if 'image' not in request.json:
        return jsonify({
            "error": "Missing image data",
            "message": "Please provide base64 encoded image data in the 'image' field"
        }), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.json['image'])
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "error": "Invalid image data",
                "message": "Could not decode the image. Please ensure it's properly base64 encoded."
            }), 400

        # Process the image
        results = process_image(image)
        
        return jsonify({
            "success": True,
            "message": "Image processed successfully",
            "data": {
                "meter_reading": results['reading'],
                "detected_digits": results['digits']
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)