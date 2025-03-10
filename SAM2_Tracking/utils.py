import yaml 
import glob
import numpy as np  
import os  
import plot_utils
from torchvision.io import decode_image
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 
import pickle
import time 

def read_config_yaml(config_path):
    """
    Reads in configuration YAML file and converts it
    to a dictionary. 

    Parameters
    ----------
    config_path : str
        The full path to the configuration file 

    Returns
    -------
    dict
        Dictionary of configuration parameters 
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # safe_load avoids arbitrary code execution

    return config

def adjust_annotations(annotations_file=None, fps=None, SAM2_start=None, 
                       df_columns=None, frame_col_name=None):
    """
    Reads in a dictionary of annotations and converts it to a Pandas 
    DataFrame. Additionally, adjusts provided annotations so the 
    frame value aligns with SAM2. Adjustment is dictated by the 
    formula: `(frame_number - SAM2_start - 1) / (fps / 3)`, which 
    is then rounded and turned into an integer. 

    Parameters
    ----------
    annotations_file : str
        The full path to the annotation file
    fps : int
        The FPS of the unreduced video that the annotations were
        initially meant for
    SAM2_start : int
        Value the ensures the annotated frame value matches up 
        with the fames that will be ingested by SAM2
    df_columns : list of str
        Keys to extract from dictionary that will become the 
        DataFrame columns
    frame_col_name : str 
        The name in `df_columns` that corresponds to the 
        annotation frame column

    Returns
    -------
    Pandas.DataFrame
        DataFrame with columns `df_columns` with column 
        `frame_col_name` adjusted

    Examples
    --------
    >>> annotations_file = "./my_annotations.npy"
    >>> fps = 24
    >>> SAM2_start = 0
    >>> df_columns = ['Frame', 'ClickType', 'FishLabel', 'Location']
    >>> frame_col_name = 'Frame'
    >>> adjust_annotations(annotations_file, fps, SAM2_start, 
                           df_columns, frame_col_name)
    """

    # TODO: in formula 3 is hardcoded, we might want to change that

    # TODO: check that all inputs are correctly provided 

    # Read in npy file corresponding to dict of annotations
    annotations = np.load(annotations_file, allow_pickle=True)

    # Convert dict to DataFrame 
    df = pd.DataFrame(list(annotations))

    # Drop all columns, except those in df_columns 
    df = df[df_columns]

    # Correct annotation frame value, so it coincides with video frame value
    df[frame_col_name] = (df[frame_col_name] - SAM2_start - 1) / (fps / 3)

    # Round and convert frame value to an integer 
    # TODO: do we need to provide a warning that a rounding was necessary? 
    df[frame_col_name] = round(df[frame_col_name]).astype(int)

    return df  

def get_frame_chunks_df(df=None, obj_name=None, frame_name=None, click_type_name=None):
    """
    Using `click_type_name` column values of 3 and 4, obtains the enter and 
    exit frame values for each `obj_name`. Additionally, returns `df`
    with index `obj_name` and drops `click_type_name` rows with 
    values of 3 and 4. 

    Parameters
    ----------
    df : Pandas.DataFrame 
        The DataFrame representing the adjusted annotations 
        that will be chunked 
    obj_name : str
        A string representing the column of `df` that will 
        become the index of returned DataFrames and corresponds 
        to the object ID
    frame_name : str 
        The name that corresponds to the column that contains 
        frame values
    click_type_name : str
        The name of the column in `df` that corresponds to 
        the click type

    Returns
    -------
    obj_frame_chunks : Pandas.DataFrame
        DataFrame with index `obj_name` and columns 
        `EnterFrame` and `ExitFrame` representing the 
        frame the object enters and exits the scene, respectively 
    df : Pandas.DataFrame
        Input `df` with index `obj_name` and dropped `click_type_name` 
        rows with values of 3 and 4.

    Raises
    ------
    RuntimeError
        If each enter point does not have a corresponding exit point

    Examples
    --------
    >>> obj_name = 'FishLabel'
    >>> frame_name = 'Frame'
    >>> click_type_name = 'ClickType'
    >>> get_frame_chunks_df(df, obj_name, frame_name, click_type_name)
    """

    # TODO: check types of inputs 

    # For each obj_name get frame where the object enters the scene 
    enter_frame = df[df[click_type_name] == 3][[obj_name, frame_name]].astype(int) 
    enter_frame = enter_frame.sort_values(by=[obj_name, frame_name], ascending=True)
    
    # For each obj_name get frame where the object exits the scene 
    exit_frame = df[df[click_type_name] == 4][[obj_name, frame_name]].astype(int) 
    exit_frame = exit_frame.sort_values(by=[obj_name, frame_name], ascending=True)

    # Check that each enter point has a corresponding exit point
    if (enter_frame.shape != exit_frame.shape) or (not np.array_equal(enter_frame[obj_name].values, exit_frame[obj_name].values)):
        raise RuntimeError(f"A {obj_name} does not have both an enter and exit point!")

    # Drop obj_name from exit_frame, now that we have sorted and compared them
    exit_frame.drop(columns=obj_name, axis=1, inplace=True)

    # Turn obj_name column back to a string 
    enter_frame[obj_name] = enter_frame[obj_name].astype(str) 

    # Concatenate columns to improve ease of use later
    obj_frame_chunks = pd.concat([enter_frame.reset_index(drop=True), exit_frame.reset_index(drop=True)], axis=1)
    obj_frame_chunks.columns = [obj_name, 'EnterFrame', 'ExitFrame']

    # Drop df rows that have click_type_name values of 3 or 4
    df = df[~df[click_type_name].isin([3, 4])]

    # Modify df so it has obj_name as its index
    df = df.set_index(obj_name)

    return obj_frame_chunks, df

def get_jpg_paths(jpg_dir):
    """
    Compiles a list of paths for all JPGs in the provided directory. 

    Parameters
    ----------
    jpg_dir : str
        The full path to the directory containing JPGs

    Returns
    -------
    list
        List of sorted JPG paths
    """

    # Grab all files with extensions .jpg, .jpeg, .JPG, .JPEG in jpg_dir
    jpg_files = glob.glob(os.path.join(jpg_dir, '*.[jJ][pP][gG]'))
    jpeg_files = glob.glob(os.path.join(jpg_dir, '*.[jJ][pP][eE][gG]'))

    # TODO: make these Path objects 
    jpg_paths = jpg_files + jpeg_files

    return sorted(jpg_paths)

def draw_masks(mask_dict, frame_path, colors, alpha=0.6, device="cuda"):
    """
    For each mask provided in `mask_dict`, draws masks on top of the 
    image provided by `frame_path`. 

    Parameters
    ----------
    mask_dict : dict of sparse tensors
        Dictionary with keys corresponding to object IDs and 
        values representing the mask created for the object ID. 
    frame_path : str 
        The video frame corresponding to the provided `mask_dict`. 
    colors : list of tuples of ints
        A list of tuples representing RGB colors for each segmentation mask
    alpha : float 
        Alpha value for the segmentation masks 

    Returns
    -------
    image : Image tensor 
        Image tensor representing the frame with the 
        masks drawn on it
    centroids : dict of tuple
        Dictionary with keys corresponding to the object 
        ID and values a tuple representing the x, y 
        centroids of the object  
    """    

    # Read in frame and convert it to a tensor 
    image = decode_image(frame_path)
    image = image.to(device)
    
    # Dictionary that will hold calculated centroids 
    centroids = {}

    # Draw each mask on top of the image representing the frame
    if mask_dict:
        for obj_id, mask in mask_dict.items():

            # Convert sparse tensor to dense and drop first channel dimension 
            mask = mask.to_dense()
            mask = mask.to(device)
            mask = mask.squeeze(0)

            # Get centroid for object ID
            centroids[obj_id] = plot_utils.get_centroid(mask)

            # Draw masks on image 
            image = draw_segmentation_masks(image, mask, colors=colors[obj_id], alpha=alpha)

    return image, centroids

def write_output_video(frame_dir, frame_masks_file, video_file, video_fps, 
                       video_frame_size, font_size=16, font_color="red", alpha=0.6, device="cuda"):
    """
    Constructs an MP4 of all frames in `frame_dir` and draws masks 
    on said frames using the masks found in `frame_masks_file`. 

    Parameters
    ----------
    frame_dir : str 
        Directory containing JPGs corresponding to the frames of the video
    frame_masks_file : str
        A pickle file composed of sparse tensors representing the generated 
        masks for each video frame
    video_fps : int
        The frames per second for the video 
    video_frame_size : list or tuple of ints
        Specifies the frame size for the video, with the first 
        element representing the width and the second corresponding
        to the height
    font_size : int
        Font size for drawn object IDs
    font_color : str
        Color of font for the drawn object IDs
    alpha : float 
        Alpha value for the segmentation masks 
    device : torch.device 
        A `torch.device` class specifying the device to use for mask drawing 

    Raises
    ------
    RuntimeError
        If no images are found in `frame_dir`

    Examples
    --------
    >>> write_output_video(frame_dir="/path/to/jpgs", frame_masks_file="masks.pkl", 
                           video_file="./test_video.mp4", video_fps=3, 
                           video_frame_size=[900, 600])
    """

    # Open and load the pickle file holding the masks 
    with open(frame_masks_file, 'rb') as file:
        frame_masks = pickle.load(file)

    # Generate a list of RGB colors for segmentation masks 
    colors = plot_utils.get_spaced_colors(100)

    # Paths to the video frames
    frame_paths = get_jpg_paths(frame_dir)

    if not frame_paths:
        raise RuntimeError(f"No images found in the path: {frame_dir}.")

    # Set the width and height of the video 
    width = video_frame_size[0]
    height = video_frame_size[1]
    
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, video_fps, (width, height))

    # Write each image to the video and draw masks on images that contain them
    for frame_idx, img_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):

        # Draw masks on the frame, if they exist
        image, centroids = draw_masks(mask_dict=frame_masks[frame_idx], frame_path=img_path, 
                                      colors=colors, alpha=alpha, device=device)

        # Get original image dimensions (before resizing)
        orig_height, orig_width = image.shape[1:]

        # Define the transformation to resize the image
        resize_transform = transforms.Resize((height, width))  # Resize to width x height

        # Apply the resize transformation to the image tensor
        image = resize_transform(image)

        # Rearrange image tensor from (C, H, W) to (H, W, C)
        image = image.permute(1, 2, 0).cpu().numpy()

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        # Display the image
        ax.imshow(image)

        # Scaling factors for centroids 
        scale_x = width / orig_width  # Scaling factor for width
        scale_y = height / orig_height  # Scaling factor for height

        # Draw object ID on object, if a centroid exists for it 
        if centroids:
            for obj_id, centroid in centroids.items():
                # Skip empty masks that occur
                if centroid is not None:
                    ax.text(centroid[0]*scale_x, centroid[1]*scale_y, obj_id, fontsize=font_size, color=font_color)

        # Set title with frame number
        ax.set_title(f"SAM2 frame: {frame_idx}", fontsize=16)

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