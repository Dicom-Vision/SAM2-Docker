
import requests
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import glob
import nibabel as nib
import re
# Server URL (adjust if needed)
server_url = 'http://localhost:80' # change port number if needed

# Path to the video images directory
video_images_dir = 'examples/images/video/'
output_images_dir = 'examples/images/output/'  # Directory to save overlaid images
image_files = [os.path.join(video_images_dir, f) for f in sorted([f for f in os.listdir(video_images_dir) if "jpg" in f], key=lambda x: int(x.split('.')[0])) if f.endswith('.jpg')]
output_nii_dir = './output_nii_files'


# Step 1: Test the '/initialize_video' endpoint
def initialize_video():
    files = [('images', open(img, 'rb')) for img in image_files]

    response = requests.post(f'{server_url}/initialize_video', files=files)
    if response.status_code == 200:
        print("Video initialized successfully!")
        session_id = response.json()['session_id']
        return session_id
    else:
        print(f"Failed to initialize video: {response.text}")
        return None

def visualize_overlay_from_nii(nii_file_path, frame_idx):
    """
    Loads and visualizes the overlay mask from a NIfTI file on the input image.
    """
    # Load the NIfTI image
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()

    mask_slice = nii_data[0, :, :]
    # Load the input image
    input_image_path = os.path.join(video_images_dir, f'{frame_idx}.jpg')  # Assuming first frame is '0.jpg'
    input_image = Image.open(input_image_path)
    input_image_np = np.array(input_image)

    # Overlay the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(input_image_np)
    plt.imshow(mask_slice, alpha=0.5, cmap='jet')  # Overlay with transparency
    plt.title("Input Image with Overlayed Mask from NIfTI")
    plt.axis('off')
    #plt.show()


# Step 2: Test the '/add_points' endpoint (positive and negative clicks)
def add_points(session_id, input_points, labels, object_id=0, frame_idx=0):

    data = {
        'session_id': session_id,
        'points': input_points,  # Format: [[x1, y1], [x2, y2], ...]
        'labels': labels,  # Format: [label1, label2, ...] where 1 = positive, 0 = negative
        'obj_id': object_id,
        'frame_idx': frame_idx
    }
    
    response = requests.post(f'{server_url}/add_points', json=data)
    if response.status_code == 200:
        # Assuming the server saves the NIfTI mask file at a known path
        nii_file_path = os.path.join(output_nii_dir, "mask.nii.gz")
        os.makedirs(os.path.dirname(nii_file_path), exist_ok=True)
        with open(nii_file_path, 'wb') as f:
            f.write(response.content)

        # Visualize the overlay mask from the NIfTI file
        visualize_overlay_from_nii(nii_file_path, frame_idx)

    else:
        print(f"Failed to add points: {response.text}")


# Function to display masks over the frame image
def show_mask(mask, ax, obj_id=None):
    """Display a mask over the current frame."""
    mask = np.array(mask[0])  # Convert mask to numpy array if needed
    ax.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask with some transparency
    if obj_id is not None:
        ax.text(10, 10, f"Object {obj_id}", bbox=dict(facecolor='yellow', alpha=0.5))


def create_overlay_video(session_id):
    nii_file_path = os.path.join(output_nii_dir, f'{session_id}_masks.nii.gz')  # Path to the saved NIfTI file
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()
    print(nii_data.shape)
    # Get the number of slices (assuming 3D)
    num_slices = nii_data.shape[0]

    overlay_images = []

    for i in range(num_slices):
        input_image_path = os.path.join(video_images_dir, f'{i}.jpg')  # Assuming frames are named '0.jpg', '1.jpg', etc.
        input_image = Image.open(input_image_path)
        input_image_np = np.array(input_image)

        # Get the mask for the current slice
        mask_slice = nii_data[i, 0, :, :]

        # Create overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(input_image_np)
        plt.imshow(mask_slice, alpha=0.5, cmap='jet')  # Overlay with transparency
        plt.title(f"Overlay for Frame {i}")
        plt.axis('off')

        # Save the overlaid image
        output_overlay_path = os.path.join(output_images_dir, f'overlay_{i}.png')
        plt.savefig(output_overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        overlay_images.append(output_overlay_path)

    # Create a video from overlay images
    save_overlay_video(overlay_images)

def save_overlay_video(overlay_images):
    images = [imageio.imread(img) for img in overlay_images]

    # Save as video
    video_path = 'output_overlay_video.mp4'
    imageio.mimsave(video_path, images, fps=10)  # Save video with 10 fps
    print(f"Overlay video saved at {video_path}")


# Step 3: Test the '/propagate_masks' endpoint
def propagate_masks(session_id):
    data = {'session_id': session_id}

    # Send the request to the Flask server to propagate masks
    response = requests.post(f'{server_url}/propagate_masks', json=data)

    if response.status_code == 200:
        # Save the received NIfTI file (.nii.gz)
        nii_file_path = os.path.join(output_nii_dir, f'{session_id}_masks.nii.gz')
        os.makedirs(os.path.dirname(nii_file_path), exist_ok=True)
        with open(nii_file_path, 'wb') as f:
            f.write(response.content)

        print(f"Masks propagated successfully and saved as {nii_file_path}")

        # Load and display the .nii.gz file (optional visualization step)
        create_overlay_video(session_id)

    else:
        print(f"Failed to propagate masks: {response.text}")


# Natural sort function using regex to extract numbers from filenames
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# Test flow
if __name__ == "__main__":
    # clear output files
    [os.remove(f) for f in glob.glob(os.path.join(output_nii_dir, '*')) if os.path.isfile(f)]
    if os.path.isfile("output_video.mp4"):
        os.remove("output_video.mp4")

    # Initialize the video
    session_id = initialize_video()

    if session_id:
        # Add points (Example: Adding two points with positive and negative labels)
        input_points = [[100, 200]]
        labels = [1]  # Positive (1) and Negative (0)
        add_points(session_id, input_points, labels, object_id=1, frame_idx=3)
        input_points = [[260, 50]]  
        labels = [1]  # Positive (1) and Negative (0)
        add_points(session_id, input_points, labels,  frame_idx=3)
        #breakpoint()
        # Propagate masks
        propagate_masks(session_id)
        
        # After propagating masks, you can use the saved voerlaid images to create a video using imageio
                
        # Load image files
        image_files = sorted(glob.glob(os.path.join(output_nii_dir, 'overlay_*.png')), key=natural_sort_key)

        # Ensure images exist
        if image_files:
            images = [imageio.imread(filename) for filename in image_files]

            # Save as video
            video_path = 'output_video.mp4'
            imageio.mimsave(video_path, images, fps=10)  # Save video with 10 fps
            print(f"Video saved at {video_path}")
        else:
            print("No images found to create the video.")