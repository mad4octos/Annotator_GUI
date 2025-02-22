import torch 
from sam2.build_sam import build_sam2_video_predictor
import numpy as np 
import os 
import cv2
import utils 
import sys 
import pickle 
import plot_utils
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialize_predictor(sam2_checkpoint, model_cfg, non_overlap_masks):

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, non_overlap_masks=non_overlap_masks)

    return predictor

def predict_masks(predictor, annotations_file, fps, SAM2_start, video_dir, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False):

    annotations_filtered  = utils.filter_annotations(annotations_file, fps, SAM2_start)

    ######### INFERENCE STATE ###########
    # Reference of init: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L63
    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=offload_video_to_cpu, 
                                           offload_state_to_cpu=offload_state_to_cpu, async_loading_frames=async_loading_frames)

    utils.process_annotations_and_predict(annotations_filtered, predictor, inference_state)

    return inference_state

def run_propagation(predictor, inference_state, frame_dir='', save_jpgs=False, save_masks=False,
                    jpg_save_dir='', masks_dict_file='', font_size=75, font_color="red", alpha=0.6):
    """
    Propagates the prompts to get the masklet across the video using the 
    provided predictor and inference state. 

    Parameters
    ----------
    predictor : SAM2VideoPredictor
        SAM2 video predictor used to predict segmentation masks
    inference_state : dict
        The video predictor's inference state,  which is a dictionary 
        of key parameters set by `SAM2VideoPredictor.init_state()`
    frame_dir : str
        The full path to the directory containing JPGs representing 
        the frames we want to perform segmentation on. 
    save_jpgs : bool
        If True, draw segmentation masks on top of frames and
        save the generated JPGs to `jpg_save_dir`
    save_masks : bool
        If True, save the generated masks in a dictionary and save 
        it as a pkl file defined by `masks_dict_file`
    jpg_save_dir : str
        The full path to the directory we want to save frames with 
        segmentation masks drawn on them
    masks_dict_file : str
        The location you want to save the dictionary of masks 
    font_size : int
        Font size for drawn object IDs
    font_color : str
        Color of font for the drawn object IDs
    alpha : float 
        Alpha value for the drawn segmentation masks 
    """

    # Gather all the JPG paths representing the frames 
    frame_paths = utils.get_jpg_paths(frame_dir)

    if save_jpgs:
        # Generate a list of RGB colors for segmentation masks 
        colors = plot_utils.get_spaced_colors(100)

    if save_masks:
        # Initialize dictionary of masks for each frame
        frame_masks = {}

    # Perform prediction of masklets across video frames 
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # TODO: In the future, we may want to modify the start and number of frames for propagation. If so, see:
        # https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L546

        # Create Bool mask and delete unneeded tensor
        bool_masks = out_mask_logits > 0.0
        del out_mask_logits

        # There's an extra dimension (1) to the masks, remove it
        bool_masks = bool_masks.squeeze(1)

        if save_masks: 
            # Convert mask tensor to sparse format and store it
            frame_masks[out_frame_idx] = bool_masks.to_sparse().cpu()

        if save_jpgs: 
            utils.draw_and_save_frame_seg(bool_masks=bool_masks, img_save_dir=jpg_save_dir, frame_paths=frame_paths, 
                                          out_frame_idx=out_frame_idx, out_obj_ids=out_obj_ids, colors=colors, 
                                          font_size=font_size, font_color=font_color, alpha=alpha)

    if save_masks: 
        # Save frame_masks as pkl file 
        with open(masks_dict_file, "wb") as file:
                pickle.dump(frame_masks, file)


def write_output_video(masked_imgs_dir, video_file, video_fps, video_frame_size):
    """
    Compiles the JPGs in `masked_imgs_dir` into an MP4. 

    Parameters
    ----------
    masked_imgs_dir : str 
        The full path to the directory containing the JPGs we 
        will use to create the video 
    video_file : str
        The name of the video file to be created 
    video_fps : int
        The frames per second for the video 
    video_frame_size : list or tuple of ints
        Specifies the frame size for the video, with the first 
        element representing the width and the second corresponding
        to the height
    """

    masked_img_paths = utils.get_jpg_paths(masked_imgs_dir)

    if not masked_img_paths:
        print(f"No images found in the path: {masked_imgs_dir}.")
        return

    # Set the width and height of the video 
    width = video_frame_size[0]
    height = video_frame_size[1]
    
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, video_fps, video_frame_size)

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 1000)  # Adjust based on image size
    font_thickness = 2
    text_color = (255, 255, 255)  # White text

    # Write each image to the video with modifications
    for frame_idx, img_path in tqdm(enumerate(masked_img_paths), total=len(masked_img_paths)):

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        # Get original image dimensions (before resizing)
        orig_height, orig_width = img.shape[:2]
        
        # Resize the image
        img = cv2.resize(img, video_frame_size)

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Display the image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Set title with frame number
        ax.set_title(f"Frame {frame_idx + 1}/{len(masked_img_paths)}", fontsize=16)

        # Set tick marks based on the original image dimensions
        ax.set_xticks(np.linspace(0, width, num=10))  # 10 evenly spaced ticks
        ax.set_xticklabels(np.linspace(0, orig_width, num=10, dtype=int))  # Map to original width
        ax.set_yticks(np.linspace(0, height, num=10))
        ax.set_yticklabels(np.linspace(0, orig_height, num=10, dtype=int))  # Map to original height

        # Set axis labels
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Pixel value")

        # Remove tick labels to keep only marks
        ax.tick_params(axis='both', labelsize=10, color='black')

        # Convert Matplotlib figure to an image
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Write the frame to the video
        video.write(frame)

        # Close the figure to save memory
        plt.close(fig)
    
    # Release the video writer
    video.release()