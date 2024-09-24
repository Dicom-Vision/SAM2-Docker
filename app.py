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
from io import BytesIO
import nibabel as nib
import json
from tempfile import NamedTemporaryFile

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

inference_states = {}
scheduler = BackgroundScheduler()
scheduler.start()


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


# Helper function to delete session
def delete_session(session_id):
    if session_id in inference_states:
        del inference_states[session_id]
        print(f"Session {session_id} deleted due to timeout.")

# Function to set or reset a session timer
def set_or_reset_timer(session_id, timeout_seconds=300):
    job_id = f"session_cleanup_{session_id}"
    # Remove the existing job for this session if it exists
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    # Schedule a new job to delete the session after 'timeout_seconds'
    scheduler.add_job(
        delete_session,
        trigger=IntervalTrigger(seconds=timeout_seconds),
        id=job_id,
        args=[session_id],
        replace_existing=True
    )



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
        'temp_dir': temp_dir,
        'n_frames': len(image_files)
    }
    set_or_reset_timer(session_id)
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
    mask = np.zeros(out_mask_logits.cpu().numpy()[0].shape)
    for i, obj_id in enumerate(out_obj_ids):
        mask = mask + ((out_mask_logits[i] > 0).cpu().numpy()[0] * (1+obj_id))
    # Convert the mask to a NIfTI file
    # Convert the mask to a NIfTI file
    affine = np.eye(4)  # You can set an appropriate affine if necessary
    nii_img = nib.Nifti1Image(mask.astype(np.float32), affine)

    # Create a temporary file for the NIfTI image
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        temp_file_path = temp_file.name
        nib.save(nii_img, temp_file_path)

    # Return the NIfTI file
    # Read the file content into memory
    with open(temp_file_path, 'rb') as f:
        nii_file_content = f.read()

    set_or_reset_timer(session_id)
    return send_file(BytesIO(nii_file_content),
                     download_name='masks.nii.gz',
                     as_attachment=True,
                     mimetype='application/gzip')


def convert_masks_to_nii(video_segments, n_frames):
    """
    Convert the propagated masks to a NIfTI (.nii.gz) file from video segments.
    :param video_segments: Dictionary with frame indices as keys and masks as values
    :return: Binary content of the .nii.gz file
    """
    # Assuming the shape of the reference image is the same as the masks
    print("Number of frames: ", len(video_segments))
    
    # Extract the shape of the first mask for dimensions
    first_frame_masks = next(iter(video_segments.values()))  # Get masks from the first frame
    ref_img_shape = first_frame_masks[next(iter(first_frame_masks))].shape  # Get the shape of the first mask
    combined_mask = np.zeros((n_frames, *ref_img_shape), dtype=np.uint8)

    # Combine all the mask arrays into one
    for out_frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            # Place the mask values directly into the combined mask
            combined_mask[out_frame_idx] = np.where(mask > 0, obj_id + 1, combined_mask[out_frame_idx])

    # Create a Nifti1Image with the combined mask
    affine = np.eye(4)  # Identity affine matrix, replace with actual affine if available
    nii_image = nib.Nifti1Image(combined_mask, affine)

    # Create a temporary file to save the NIfTI image
    with NamedTemporaryFile(suffix='.nii.gz') as tmp_file:
        nib.save(nii_image, tmp_file.name)  # Save to the temporary file

        # Read the file content into memory
        with open(tmp_file.name, 'rb') as f:
            nii_file_content = f.read()

    return nii_file_content


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
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    start_frame_idx = min(video_segments.keys())
    if start_frame_idx != 0:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_idx, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }


    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    del inference_states[session_id] # TODO set a timer instead

    # Convert masks to .nii and return as binary content
    nii_file_content = convert_masks_to_nii(video_segments, state_info['n_frames'])

    set_or_reset_timer(session_id)
    # Send the .nii file as a binary response
    return send_file(BytesIO(nii_file_content),
                     download_name='masks.nii.gz',
                     as_attachment=True,
                     mimetype='application/gzip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
