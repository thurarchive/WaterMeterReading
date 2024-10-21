import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import os

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

def process_image(image, model):
    """
    Process a single image and return the detected digit sequence and annotated image
    """
    # Convert PIL Image to OpenCV format
    image = np.array(image)
    if image.shape[-1] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    processed_image, (x_offset, y_offset, new_width, new_height) = resize_maintain_aspect(image)
    
    # Perform inference
    results = model(processed_image)
    
    # Create a copy of original image for drawing
    display_image = image.copy()
    
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
                
                # Convert to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Store detection for sorting
                detected_digits.append((x1, y1, x2, y2, class_id, confidence))
                
                # Draw bounding box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sort digits by x-coordinate to get the sequence
    detected_digits.sort(key=lambda x: x[0])
    
    # Create digit sequence
    digit_sequence = ''.join([str(digit[4]) for digit in detected_digits])
    
    # Convert back to PIL Image for Streamlit
    display_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    
    return digit_sequence, display_image

def app():
    st.set_page_config(page_title="Water Meter Reader", layout="wide")
    
    st.title('Water Meter Reading Web App')
    st.subheader('Powered by YOLOv8')
    
    # Load model
    @st.cache_resource
    def load_model():
        return YOLO('my_trained_model.pt')
    
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader that accepts multiple files
    uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        # Create columns for the results
        cols = st.columns(len(uploaded_files))
        
        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx]:
                try:
                    # Read image
                    image = Image.open(uploaded_file)
                    
                    # Process image
                    digit_sequence, annotated_image = process_image(image, model)
                    
                    # Display results
                    st.image(annotated_image, caption=f"Processed Image {idx+1}", use_column_width=True)
                    st.success(f"Reading: {digit_sequence} m³")
                    
                except Exception as e:
                    st.error(f"Error processing image {idx+1}: {str(e)}")
    
    # Add some helpful information
    with st.expander("Help"):
        st.markdown("""
        ### How to use:
        1. Upload one or more images of water meters
        2. The app will automatically detect and display the readings
        3. Each reading will be shown in cubic meters (m³)
        
        ### Tips:
        - Make sure the images are clear and well-lit
        - The meter display should be clearly visible
        - Supported formats: JPG, JPEG, PNG
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("Created by Fajri Fathur/Mahardika Caraka Indonesia")

if __name__ == "__main__":
    app()