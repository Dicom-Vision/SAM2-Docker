import os
import io
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image
import matplotlib.pyplot as plt
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)

# Load the SAM2 model on server start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = os.path.dirname(os.path.dirname(sam2.__file__))

sam2_checkpoint = f"{root_path}/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

@app.route('/predict_mask', methods=['POST'])
def predict_mask():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    input_point = request.form.get('input_point', None)
    input_label = request.form.get('input_label', 1)

    if input_point is None:
        return jsonify({'error': 'No input point provided'}), 400

    # Convert input_point back to a list of integers
    input_point = [int(coord) for coord in input_point.split(',')]
    input_point = np.array([input_point])  # Convert to the required NumPy array format

    # Process the image
    image = Image.open(image_file)
    image = np.array(image.convert("RGB"))

    # Set image in the predictor
    predictor.set_image(image)

    # Convert the input point into a NumPy array and input label
    input_label = np.array([int(input_label)])

    # Predict masks
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Sort by scores
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    # Generate mask image
    mask_image = masks[0]  # Use the highest score mask
    mask_image = mask_image.astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_image)

    # Return the generated mask as an image
    img_io = io.BytesIO()
    mask_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
