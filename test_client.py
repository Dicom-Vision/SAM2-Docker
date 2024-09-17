import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Set the URL for the server (adjust if the server runs on a different port or IP)
server_url = 'http://localhost:5000/predict_mask'

# Path to the local image file to test
image_path = 'examples/images/truck.jpg'

# Input point and label for the prediction
input_point = [500, 375]  # Example point (adjust as necessary)
input_label = 1  # Positive label for the input point

# Open and display the input image
image = Image.open(image_path)
image_np = np.array(image.convert("RGB"))

# Show input image and point
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.scatter(input_point[0], input_point[1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
plt.title("Input Image with Point", fontsize=18)
plt.axis('on')
plt.show()

# Prepare the request payload
files = {'image': open(image_path, 'rb')}
data = {
    'input_point': f'{input_point[0]},{input_point[1]}',  # Send point as comma-separated values
    'input_label': input_label
}

# Make a request to the server
response = requests.post(server_url, files=files, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Load the returned mask image
    mask_image = Image.open(BytesIO(response.content))

    # Display the output mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(mask_image, alpha=0.6)  # Overlay the mask with transparency
    plt.title("Predicted Mask", fontsize=18)
    plt.axis('off')
    plt.show()
else:
    print(f"Request failed with status code {response.status_code} and message: {response.text}")
