from utils import *

#USER EDITS HERE: these should be variables that can be input from the run_vidpred_seq.sh, not hard coded here
fps = 24 # Replace with your fps
SAM2_start=0 #Replace with the SAM2_start, as provided by your project and used in your GUI annotations

#Import the frames
video_dir = "./GX137102_frames"
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#Import the video_segment results
save_points_dir = "save_points"



# Loop over all video segment batch files
batch_files = sorted(glob.glob(os.path.join(save_points_dir, "video_segments_batch_*.pkl")))

# choose the video output parameters
out_fps = 3
frame_size = (600, 400)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for batch_file in batch_files:
    # Extract the batch index from the filename.
    # For a filename like "video_segments_batch_0.pkl" the index will be 0.
    base = os.path.basename(batch_file)
    batch_idx = int(os.path.splitext(base)[0].split('_')[-1])
    
    # Load the video_segments for this batch
    with open(batch_file, 'rb') as f:
        video_segments = pickle.load(f)
    
    # Set up a video writer for this batch
    output_video_path = f"output_video_batch_{batch_idx}.mp4"
    video_writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, frame_size)
    
    # Determine the frame range for this batch (10 frames per batch)
    start_frame = batch_idx * 10
    end_frame = min((batch_idx + 1) * 10, len(frame_names))
    
    for out_frame_idx in range(start_frame, end_frame):
        # Create a new figure for the frame
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}/{out_frame_idx * (fps/3) + SAM2_start}")
    
        # Load and display the image
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])
        image = Image.open(image_path)
        ax.imshow(image)
    
        # If masks exist for this frame, display them and add text
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
                
                # Calculate the maximum y coordinate for the mask
                mask_coords = np.column_stack(np.where(out_mask))
                if mask_coords.size > 0:  # Ensure mask contains points
                    min_y = mask_coords[:, 0].min()
                    min_x = mask_coords[mask_coords[:, 0] == min_y][:, 1].mean()
                    add_text(ax, f"ID: {out_obj_id}", position=(min_x, min_y - 10))
    
        # Draw the canvas and convert to a numpy array (RGB only)
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        frame = buf[..., :3]  # Drop the alpha channel
    
        # Write the frame to the video (convert RGB to BGR for OpenCV)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
        plt.close(fig)
    
    video_writer.release()
 
# --- Concatenate all batch output videos into one final video ---
final_output_video_path = "final_output_video.mp4"
final_video_writer = cv2.VideoWriter(final_output_video_path, fourcc, out_fps, frame_size)

# Get list of all batch videos sorted by batch index
batch_output_videos = sorted(
    glob.glob("output_video_batch_*.mp4"),
    key=lambda x: int(x.split('_')[-1].split('.')[0])
)
for batch_video in batch_output_videos:
    cap = cv2.VideoCapture(batch_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        final_video_writer.write(frame)
    cap.release()

final_video_writer.release()
print("Final concatenated video saved as:", final_output_video_path)
