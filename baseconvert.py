import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

# Replace 'your_image.jpg' with your actual image path
image_path = 'sampleImages\sample.jpg'
base64_string = image_to_base64(image_path)

# Print the base64 string
print(base64_string)