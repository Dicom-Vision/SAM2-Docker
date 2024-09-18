import os
import io
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import uuid
import tempfile
import os
inference_states = {}

app = Flask(__name__)


# Use bfloat16 precision
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TensorFloat-32 (if applicable)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

root_path = os.path.dirname(os.path.dirname(sam2.__file__))
# Load the SAM 2 model
sam2_checkpoint = f"{root_path}/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

from flask import Flask, request, jsonify, send_file
import json


@app.route('/initialize_video', methods=['POST'])
def initialize_video():
    # Get the images from the request
    image_files = request.files.getlist('images')
    if not image_files:
        return jsonify({'error': 'No images provided'}), 400

    # Save images to a temporary directory
    temp_dir = tempfile.mkdtemp()
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(temp_dir, f"{idx}.jpg")
        image = Image.open(image_file).convert("RGB")
        image.save(image_path)

    # Initialize the inference state
    session_id = str(uuid.uuid4())
    inference_state = predictor.init_state(video_path=temp_dir)
    inference_states[session_id] = {
        'inference_state': inference_state,
        'temp_dir': temp_dir
    }

    return jsonify({'session_id': session_id})

@app.route('/add_points', methods=['POST'])
def add_points():
    # Get data from the request JSON
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    session_id = data.get('session_id')
    frame_idx = data.get('frame_idx')
    obj_id = data.get('obj_id')
    points = data.get('points')
    labels = data.get('labels')
    if session_id is None or frame_idx is None or obj_id is None or points is None or labels is None:
        return jsonify({'error': 'All fields are required'}), 400

    # Retrieve the inference state
    state_info = inference_states.get(session_id)
    if state_info is None:
        return jsonify({'error': 'Invalid session_id'}), 400
    inference_state = state_info['inference_state']

    # Convert to numpy arrays
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Add new points to the inference state
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # Optionally return the mask for the current frame
    mask = (out_mask_logits[0] > 0).cpu().numpy()[0]
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))

    # Send the mask image
    img_io = io.BytesIO()
    mask_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.route('/propagate_masks', methods=['POST'])
def propagate_masks():
    # Get data from the request JSON
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    session_id = data.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    # Retrieve the inference state
    state_info = inference_states.get(session_id)
    if state_info is None:
        return jsonify({'error': 'Invalid session_id'}), 400
    inference_state = state_info['inference_state']
    temp_dir = state_info['temp_dir']

    # Propagate masks
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        masks = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0).cpu().numpy()
            masks[out_obj_id] = mask.tolist()
        video_segments[out_frame_idx] = masks

    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    del inference_states[session_id]

    return jsonify({'video_segments': video_segments})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
