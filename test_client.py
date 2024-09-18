
import requests
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import glob
# Server URL (adjust if needed)
server_url = 'http://localhost:5000'

# Path to the video images directory
video_images_dir = 'examples/images/video/'
output_images_dir = 'examples/images/output/'  # Directory to save overlaid images
image_files = [os.path.join(video_images_dir, f) for f in sorted([f for f in os.listdir(video_images_dir) if "jpg" in f], key=lambda x: int(x.split('.')[0])) if f.endswith('.jpg')]
output_files = []
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

# Step 2: Test the '/add_points' endpoint (positive and negative clicks)
def add_points(session_id, input_points, labels):

    data = {
        'session_id': session_id,
        'points': input_points,  # Format: [[x1, y1], [x2, y2], ...]
        'labels': labels,  # Format: [label1, label2, ...] where 1 = positive, 0 = negative
        'obj_id': 0,
        'frame_idx': 0
    }
    
    response = requests.post(f'{server_url}/add_points', json=data)
    if response.status_code == 200:
        input_image_path = os.path.join(video_images_dir, '0.jpg')  # Assuming first frame is '0.jpg'
        input_image = Image.open(input_image_path)
        input_image_np = np.array(input_image)
        print("Points added successfully!")
        # Retrieve the mask image from the response
        mask_image = Image.open(BytesIO(response.content))
        mask_image_np = np.array(mask_image)

        # Display the input image with the mask overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(input_image_np)
        plt.imshow(mask_image_np, alpha=0.5)  # Overlay with transparency
        plt.title("Input Image with Overlayed Mask")
        plt.axis('off')
        plt.show()
    else:
        print(f"Failed to add points: {response.text}")

# Function to display masks over the frame image
def show_mask(mask, ax, obj_id=None):
    """Display a mask over the current frame."""
    mask = np.array(mask[0])  # Convert mask to numpy array if needed
    ax.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask with some transparency
    if obj_id is not None:
        ax.text(10, 10, f"Object {obj_id}", bbox=dict(facecolor='yellow', alpha=0.5))


# Step 3: Test the '/propagate_masks' endpoint
def propagate_masks(session_id):
    data = {'session_id': session_id}
    response = requests.post(f'{server_url}/propagate_masks', json=data)
    
    if response.status_code == 200:
        print("Masks propagated successfully!")
        video_segments = response.json()['video_segments']
        vis_frame_stride = 1  # Display every 15th frame (can be adjusted)
        
        # Iterate over frames and display the masks
        for out_frame_idx in range(0, len(image_files), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"Frame {out_frame_idx}")
            
            # Load the frame image
            frame_path = image_files[out_frame_idx]
            frame_image = Image.open(frame_path)
            # plt.imshow(frame_image)
            
            # Overlay masks on the frame
            for out_obj_id, out_mask in video_segments.get(str(out_frame_idx), {}).items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            
            plt.axis('off')

            # Save the overlaid image
            output_path = os.path.join(output_images_dir, f'{out_frame_idx}_overlaid.jpg')
            output_files.append(output_path)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        print("Overlaid images saved.")
    else:
        print(f"Failed to propagate masks: {response.text}")

# Test flow
if __name__ == "__main__":
    # clear output files
    [os.remove(f) for f in glob.glob(os.path.join(output_images_dir, '*')) if os.path.isfile(f)]
    if os.path.isfile("output_video.mp4"): 
        os.remove("output_video.mp4")

    # Initialize the video
    session_id = initialize_video()
    
    if session_id:
        # Add points (Example: Adding two points with positive and negative labels)
        input_points = [[100, 200], [150, 250]]  # Two example points
        labels = [1, 0]  # Positive (1) and Negative (0)
        add_points(session_id, input_points, labels)
        
        # Propagate masks
        propagate_masks(session_id)
        
        # After propagating masks, you can use the saved overlaid images to create a video using imageio
        images = []
        for frame_idx in output_files:
            if frame_idx.endswith('_overlaid.jpg'):
                images.append(imageio.imread(os.path.join(output_images_dir, frame_idx.split(os.sep)[-1])))

        # Save as video (optional step)
        video_path = 'output_video.mp4'
        imageio.mimsave(video_path, images, fps=10)  # Save video with 10 fps
        print(f"Video saved at {video_path}")