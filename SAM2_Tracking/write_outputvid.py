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
vid_segments = "video_segments.pkl"

with open(vid_segments, 'rb') as f:
        video_segments = pickle.load(f)

# choose the video output parameters
output_video_path = "corrected_third_video.mp4"
out_fps = 3
frame_size = (600, 400)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, frame_size)

for out_frame_idx in range(0, len(frame_names)):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"frame {out_frame_idx}/{out_frame_idx * (fps/3) + SAM2_start}")

    image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
    ax.imshow(image)

    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, ax, obj_id=out_obj_id)

            #Calculate the max y coordinate for the mask
            mask_coords = np.column_stack(np.where(out_mask))
            if mask_coords.size>0: #Ensure mask contains points
                min_y = mask_coords[:,0].min()
                min_x = mask_coords[mask_coords[:,0] == min_y][:,1].mean()
                add_text(ax, f"ID: {out_obj_id}", position=(min_x, min_y - 10))
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    plt.close(fig)

video_writer.release()
