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
import zipfile
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pydicom
import glob
inference_states = {}
scheduler = BackgroundScheduler()
scheduler.start()


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB

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
def set_or_reset_timer(session_id, timeout_seconds=3000):
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


@app.route("/get_server_status", methods=["POST"])
def get_server_status():
    meta = request.json
    # if not has_valid_credentials(meta["api_key"]):
    #    return {"message": "invalid access code"}, 401

    return {"status": "happily running"}, 200


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File is too large"}), 413


@app.route('/initialize_video', methods=['POST'])
def initialize_video():
    meta = request.form.to_dict()
    session_id = meta.get('session_id', str(uuid.uuid4()))

    # Get the zip file from the request
    zip_file = request.files.get('data_binary')
    print("received params", meta)
    # get ww/wl values
    ww = meta.get('ww', None)
    wl = meta.get('wl', None)
    if ww:
        print(f"found WW value: ${ww}")
    if wl:
        print(f"found WW value: ${wl}")
    
    if not zip_file:
        return jsonify({'error': 'No zip file provided'}), 400

    print(f"Received file: {zip_file.filename}")
    
    # Save the uploaded file to a temporary location
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_file.filename)
    zip_file.save(temp_zip_path)

    # Check if the file was saved correctly and has a non-zero size
    if os.path.getsize(temp_zip_path) == 0:
        print("File size is 0 after saving!")
        return jsonify({'error': 'File size is 0'}), 400

    # Create a temporary directory for extracted DICOM files and converted JPGs
    temp_dir = tempfile.mkdtemp()
    dcm_dir = os.path.join(temp_dir, 'dicoms')
    jpg_dir = os.path.join(temp_dir, 'jpgs')
    os.makedirs(dcm_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall(dcm_dir)

    # Load DICOM files
    dicom_filenames = glob.glob(os.path.join(dcm_dir, '*.dcm'))

    files = [pydicom.dcmread(fname) for fname in dicom_filenames if fname.endswith('.dcm')]

    print(f"Loaded {len(files)} DICOM files.")

    # Skip files without SliceLocation
    # slices = [f for f in files if hasattr(f, 'SliceLocation')]
    # print(f"Skipped {len(files) - len(slices)} files with no SliceLocation.")

    # Sort slices based on SliceLocation
    # slices = sorted(slices, key=lambda s: s.SliceLocation)

    # Sort filenames in the same order as slices
    # dicom_filenames = sorted(dicom_filenames, key=lambda fname: pydicom.dcmread(fname).SliceLocation)

    # Sort slices using SliceLocation if available, otherwise fallback to InstanceNumber or ImagePositionPatient
    def sort_key(s):
        if hasattr(s, 'SliceLocation'):
            return s.SliceLocation
        elif hasattr(s, 'InstanceNumber'):
            return s.InstanceNumber
        elif hasattr(s, 'ImagePositionPatient'):
            return s.ImagePositionPatient[2]
        else:
            return float('inf')  # Put files with no relevant tag at the end

    slices = sorted(files, key=sort_key)

    # Sort filenames in the same order as slices
    dicom_filenames = sorted(dicom_filenames, key=lambda fname: sort_key(pydicom.dcmread(fname)))


    # Prepare 3D array of pixel data
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # Fill the 3D array with pixel data
    for i, s in enumerate(slices):
        # Normalize pixel data with slope and intercept
        img2d = s.pixel_array.astype(np.float32)
        slope = getattr(s, 'RescaleSlope', 1)
        intercept = getattr(s, 'RescaleIntercept', 0)
        img2d = img2d * slope + intercept

        # Apply window width and level if they are provided
        if ww and wl:
            ww = float(ww)
            wl = float(wl)
            min_val = wl - ww / 2
            max_val = wl + ww / 2
            img2d = np.clip(img2d, min_val, max_val)
            img2d = (img2d - min_val) / (max_val - min_val) * 255  # Scale to 0-255 for JPG

        img3d[:, :, i] = img2d

    # Normalize the 3D array
    # non_zero_values = img3d[img3d != 0]
    # min_val = int(np.min(non_zero_values)) + 100
    # max_val = int(0.67 * np.max(non_zero_values))
    # img3d_normalized = np.clip(img3d, min_val, max_val)
    # img3d_normalized = 255 * (img3d_normalized - min_val) / (max_val - min_val)
    # img3d_normalized = img3d_normalized.astype(np.uint8)

    # Convert slices to JPG and save in jpg_dir
    for idx in range(img3d.shape[2]):
        image_array = img3d[:, :, idx]
        image = Image.fromarray(image_array).convert("L")
        image.save(os.path.join(jpg_dir, f"{idx}.jpg"), quality=100)

    # Initialize the inference state
    inference_state = predictor.init_state(video_path=jpg_dir)
    inference_states[session_id] = {
        'inference_state': inference_state,
        'temp_dir': temp_dir,
        'n_frames': len(os.listdir(jpg_dir))
    }
    set_or_reset_timer(session_id)
    return jsonify({'session_id': session_id})

@app.route('/add_points', methods=['POST'])
def add_points():
    # Get data from the request JSON
    data = request.form.to_dict()
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    session_id = data.get('session_id')
    frame_idx = int(data.get('frame_idx'))
    obj_id = int(data.get('obj_id'))
    points = json.loads(data.get('points'))
    labels = json.loads(data.get('labels'))
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
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        clear_old_points=False,
    )

    # Optionally return the mask for the current frame
    mask = np.zeros(out_mask_logits.cpu().numpy()[0].shape)
    for i, obj_id in enumerate(out_obj_ids):
        mask = mask + ((out_mask_logits[i] > 0).cpu().numpy()[0] * (1+obj_id))
    non_zero_count = np.count_nonzero(mask)

    print(f"Number of non-zero values in the mask: {non_zero_count}")
    print("logits_sum: ", np.sum(out_mask_logits.cpu().numpy()))
    # Convert the mask to a NIfTI file
    # Convert the mask to a NIfTI file
    affine = np.eye(4)  # You can set an appropriate affine if necessary
    nii_img = nib.Nifti1Image(mask.astype(np.float32), affine)

    # Create a temporary file for the NIfTI image
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        temp_file_path = temp_file.name
        nib.save(nii_img, temp_file_path)

    # Create a temporary ZIP file and add the NIfTI file to it
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip_file:
        temp_zip_file_path = temp_zip_file.name
        with zipfile.ZipFile(temp_zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the NIfTI file to the ZIP archive
            zipf.write(temp_file_path, arcname='masks.nii.gz')

    # Read the ZIP file content into memory
    with open(temp_zip_file_path, 'rb') as f:
        zip_file_content = f.read()

    # Clean up the temporary files
    import os
    os.remove(temp_file_path)
    os.remove(temp_zip_file_path)

    set_or_reset_timer(session_id)

    # Return the ZIP file
    return send_file(
        BytesIO(zip_file_content),
        download_name='masks.zip',
        as_attachment=True,
        mimetype='application/octet-stream'
        )
                     


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

    return nii_image


@app.route('/propagate_masks', methods=['POST'])
def propagate_masks():
    # Get data from the request JSON
    data = request.form.to_dict()
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
    print("total number of segments: ", video_segments)

    # Clean up temporary files
    import shutil
    try: 
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        pass
    # Convert masks to .nii and return as binary content
    nii_img = convert_masks_to_nii(video_segments, state_info['n_frames'])

    # Create a temporary file for the NIfTI image
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        temp_file_path = temp_file.name
        nib.save(nii_img, temp_file_path)

     # Create a temporary ZIP file and add the NIfTI file to it
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip_file:
        temp_zip_file_path = temp_zip_file.name
        with zipfile.ZipFile(temp_zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the NIfTI file to the ZIP archive
            zipf.write(temp_file_path, arcname='masks.nii.gz')

    # Read the ZIP file content into memory
    with open(temp_zip_file_path, 'rb') as f:
        zip_file_content = f.read()

    # Clean up the temporary files
    import os
    os.remove(temp_file_path)
    os.remove(temp_zip_file_path)

    set_or_reset_timer(session_id)
    # Send the .nii file as a binary response
    return send_file(BytesIO(zip_file_content),
                     download_name='masks.nii.gz',
                     as_attachment=True,
                     mimetype='application/octet-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
