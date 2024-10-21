# Water Meter Reading Application

This repository contains a deep learning-based water meter reading application developed for PT. Mahardika Caraka Indonesia. The system uses YOLOv8 for digit detection and recognition in water meter images.

## Overview

The application provides two interfaces:
1. A web-based dashboard (Streamlit)
2. A REST API (Flask)

Both interfaces use the same underlying YOLOv8 model to detect and read water meter digits.

## Repository Structure

```
water-meter-reader/
├── app.py              # Flask API server
├── dashboard.py        # Streamlit web interface
├── baseconvert.py      # Base64 conversion utility
├── my_trained_model.pt # YOLOv8 trained model
└── requirements.txt    # Python dependencies
```

## Features

- Automatic digit detection and recognition
- Support for multiple image formats (JPG, JPEG, PNG)
- Real-time processing
- High accuracy digit recognition
- REST API support
- User-friendly web interface

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd water-meter-reader
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Dashboard

To run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
The dashboard will be accessible at `http://localhost:8501`

### API Server

To run the Flask API server:
```bash
python app.py
```
The API will be accessible at `http://localhost:5000`

### API Endpoints

- `GET /`: Returns API information
- `POST /predict`: Processes water meter images
  - Accepts: Base64 encoded image
  - Returns: JSON with digit readings and confidence scores

### Example API Request

```python
import requests
import base64

# Convert image to base64
with open("meter_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Make API request
response = requests.post(
    "http://localhost:5000/predict",
    json={"image": encoded_string}
)

# Get results
result = response.json()
print(f"Meter Reading: {result['data']['meter_reading']}")
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- Streamlit
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Model Information

The system uses a custom-trained YOLOv8 model (`my_trained_model.pt`) specifically trained to detect and recognize digits on water meters. The model has been trained on a diverse dataset of water meter images to ensure robust performance across different meter types and lighting conditions.

## Development

Built and maintained by PT. Mahardika Caraka Indonesia. For questions or support, please contact [contact information].


---
© 2024 PT. Mahardika Caraka Indonesia. All rights reserved.
