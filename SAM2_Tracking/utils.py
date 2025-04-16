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
from sam2_fish_segmenter import SAM2FishSegmenter

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

def lol_check(variable):
    """"
    Reads in a variable and checks if it is a list of lists (lol).
    
    Parameters
    ----------
    variable : any
        The input variable to check. 
    Returns
    -------
    bool
        True, if variable is a list of lists, False otherwise.
    """
    return isinstance (variable, list) and all(isinstance(item, list) for item in variable)

def extract_config_lens(configs):
    """
    Validates and extracts the number of trials from a configuration dictionary.

    Special handling is included for the "video_frame_size" key, which may contain a list of
    lists (e.g., [[1920, 1080], [1280, 720], ...]) and is included in the validation if so.

    Parameters
    ----------
    configs : dict
        Dictionary of configuration parameters, where each value is either a single value
        or a list of values representing multiple trials.

    Returns
    -------
    int
        The number of trials specified across multi-trial configurations.

    Raises
    ------
    ValueError
        If multiple configuration keys have differing numbers of trials.
    
    Examples
    --------
    >>> configs = {
    ...     'frame_dir': ['path1', 'path2'],
    ...     'model_cfg': ['cfg1', 'cfg2'],
    ...     'fps': 30
    ... }
    >>> extract_config_lens(configs)
    There are 2 trials provided for processing.
    2
    """
    # Get length of provided values for each listed config key as a dictionary 
    config_counts = {key: len(value) for key, value in configs.items() if isinstance(value,list) 
                     and key != "video_frame_size"}
    # If multiple video_frame_size trials are provided, add count to config_counts
    if lol_check(configs["video_frame_size"]):
        config_counts["video_frame_size"] = len(configs["video_frame_size"])

    # Extract the unique lengths of configuration values
    unique_counts = set(config_counts.values())
    
    # Check for mismatches in provided configuration lengths,
    # accounting for instances where the user provided a single value in list format
    mismatch = len([x for x in unique_counts if x!= 1]) > 1

 # Confirm that all provided configurations are in a list of the same length
    if mismatch:
        # Find the inconsistent key lists
        # Raise a helpful error stating the length of each value for keys that have multiple trials
        err_msg = "Inconsistent configuration lengths found:\n"
        err_msg += "All configuration parameters that have \n"
        err_msg += "multiple trials, must have the same length. \n"
        for key, count in config_counts.items():
            err_msg += f" - {key} trial count: {count}\n"
        raise ValueError(err_msg)
    else:
        if len(unique_counts) > 1: # Multiple trials, some contain lists of a single entry
            unique_counts.discard(1)
            trial_count = unique_counts.pop()
            return trial_count
        if not unique_counts: # The set is empty because all values are provided as a single entry
            trial_count = 1
            return trial_count
        else: # All list configs are provided in the same length (no single lists)
            trial_count = unique_counts.pop()
            return trial_count

def get_trial_config(configs, i):
    """
    Extracts a specific trial configuration from a general configuration dictionary.

    This function supports configurations where values can be:
    - A single value (int, str, etc.)
    - A list of values (used for multiple trials)
    - A list of lists (specifically for the "video_frame_size" key)

    Parameters
    ----------
    configs : dict
        Dictionary containing configuration values. Each key's value can be:
        - a scalar (same for all trials),
        - a list (each entry for a separate trial), or
        - for "video_frame_size" specifically, a list of lists or a single list.
    i : int
        Index of the trial to extract configuration for.

    Returns
    -------
    trial_config : dict
        Dictionary containing configuration values for the i-th trial.

    Notes
    -----
    This function depends on a helper function `lol_check(value)` that determines
    whether a value is a list of lists.

    Examples
    --------
    >>> configs = {
    ...     "fps": [32, 64],
    ...     "out_fps": 3,
    ...     "video_frame_size": [[640, 480], [1280, 720]]
    ... }
    >>> get_trial_config(configs, 1)
    {'fps': 64, 'out_fps': 3, 'video_frame_size': [1280, 720]}
    """
    
    trial_config = {}
    for key, value in configs.items():
        # Handle "video_frame_size" separately
        if key == "video_frame_size":
            # Check for list of lists
            if not lol_check(value):
                trial_config[key] = value # Assign the entire value if not a list of lists
            else: 
                trial_config[key] = value[i] # Assign the i-th list if not a list of lists
        elif isinstance(value,list):
            if len(value) > 1: # Lists with more than one value
                trial_config[key] = value[i] # Assign the i-th value
            else: # List with a single value
                trial_config[key] = value[0] # Assign the first (only) value
        else: # single non-list value
            trial_config[key] = value
    return trial_config

def run_segmentation(config_file, device):
    """
    Runs SAM2 segmentation and mask propagation for one or more trial configurations.

    This function reads a YAML configuration file and processes it to extract 
    parameters for each trial. Each configuration parameter in the YAML file must 
    either be a single value (applied to all trials) or a list of values (one per trial). 
    For each trial, the function initializes a `SAM2FishSegmenter` object with the appropriate 
    configuration and executes the segmentation and propagation process.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file containing all segmentation parameters.
        Each parameter should either be a scalar (applied to all trials) or a list of values
        (with one entry per trial).
    device : torch.device 
            A `torch.device` class specifying the device to use for `build_sam2_video_predictor`

    Returns
    -------
    None
        The function does not return anything explicitly. However, it saves a pickled 
        dictionary of masks (as specified by `masks_dict_file` in each trial config) 
        for each trial after segmentation and propagation.

    Notes
    -----
    - The number of trials is determined by the length of list-type parameters in the 
      configuration file. All such parameters must have the same length.
    - The function uses `read_config_yaml`, `extract_config_lens`, and `get_trial_config` 
      as helpers in utils to parse and manage configurations.
    
    Warnings
    --------
    If the configuration specifies only a single `masks_dict_file` name while running multiple 
    trials, the output masks will be written to the same file, and results will be 
    overwritten. To prevent this, provide a list of unique `masks_dict_file` names — 
    one for each trial.
    
    Examples
    --------
    >>> run_segmentation("template_configs.yaml", device=torch.device("cuda"))
    Processing Trial 0: Frames from ./data/frames1, Annotations from ./data/annotations1.npy, Masks saving to ./generated_frame_masks1.pkl
    Processing Trial 1: Frames from ./data/frames2, Annotations from ./data/annotations2.npy, Masks saving to ./generated_frame_masks2.pkl
    """
    # Load the YAML configuration file
    configs = read_config_yaml(config_file)
    
    # Retrieve trial count from the length of values provided for each configuration key
    trial_count = extract_config_lens(configs)
    print(f"There are {trial_count} trials provided for processing")

    # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = get_trial_config(configs, i)

        # Initialize the segmenter with modified trial configs
        segmenter = SAM2FishSegmenter(configs = trial_config, device = device)
        print(f"Processing Trial {i}: Frames from {trial_config['frame_dir']}, Annotations from {trial_config['annotations_file']}, Masks saving to {trial_config['masks_dict_file']}")
        segmenter.run_propagation()

def adjust_annotations(annotations_file=None, fps=None, out_fps=None, SAM2_start=None, 
                       df_columns=None, frame_col_name=None):
    """
    Reads in a dictionary of annotations and converts it to a Pandas 
    DataFrame. Additionally, adjusts provided annotations so the 
    frame values from annotations of the unreduced video align 
    with the reduced frames provided to SAM2. 
    Adjustment is dictated by the formula: 
    `(frame_index - SAM2_start) / (fps / out_fps)`
    This formula takes the frame index provided from the unreduced video 
    and subtracts the SAM2_start value to align with the specified 
    frame delay at extraction (which can be used for syncing sequential videos).
    The resulting value is then divided by the inverval at which frames were extracted,
    i.e., `fps / out_fps`, to adjust the provided annotation frame values to the extracted 
    frame values ingested by SAM2. The final frame value is then rounded, if necessary, 
    and turned into an integer. 

    Parameters
    ----------
    annotations_file : str
        The full path to the annotation file
    fps : int
        The FPS of the unreduced video that the annotations were
        initially meant for
    out_fps: int
        The FPS of the reduced video or extracted frames ingested by SAM2.
        Also the final temporal resolution of the data. 
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

    Raises
    ------
    RuntimeWarning
        If adjustment of frame values results in frame 
        values that need to be rounded 

    Examples
    --------
    >>> annotations_file = "./my_annotations.npy"
    >>> fps = 24
    >>> out_fps = 3
    >>> SAM2_start = 0
    >>> df_columns = ['Frame', 'ClickType', 'FishLabel', 'Location']
    >>> frame_col_name = 'Frame'
    >>> adjust_annotations(annotations_file, fps, out_fps, SAM2_start, 
                           df_columns, frame_col_name)
    """

    # TODO: check that all inputs are correctly provided 

    # Read in npy file corresponding to dict of annotations
    annotations = np.load(annotations_file, allow_pickle=True)

    # Convert dict to DataFrame 
    df = pd.DataFrame(list(annotations))

    # Drop all columns, except those in df_columns 
    df = df[df_columns]

    # Correct annotation frame value, so it coincides with video frame value
    df[frame_col_name] = (df[frame_col_name] - SAM2_start) / (fps / out_fps)

    # Store original frame values
    original_values = df[frame_col_name].copy()

    # Round and convert frame value to an integer 
    df[frame_col_name] = round(df[frame_col_name]).astype(int)

    # Identify which values were rounded
    rounded_indices = original_values != df[frame_col_name]

    # Print warning for affected frames
    if rounded_indices.any():
        rounded_values = original_values[rounded_indices]
        new_values = df.loc[rounded_indices, frame_col_name]
        for original, new in zip(rounded_values, new_values):
            print(f"Warning: Frame value {original} was rounded to {new}.")

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

def run_video_processing(configs, device):
    """
    Generates output videos visualizing SAM2 segmentation results for one or more trials.

    This function reads a YAML configuration file and extracts trial-specific parameters 
    to generate annotated output videos using `write_output_video()`. Each trial uses 
    previously computed masks from `SAM2FishSegmenter` and overlays them on input 
    frames to produce a visual result.

    Parameters
    ----------
    configs : str
        Path to the YAML configuration file containing video generation settings. 
        Each parameter must either be a single value (applied to all trials) or a list 
        of values with one entry per trial.
    device : torch.device 
            A `torch.device` class specifying the device to use for `build_sam2_video_predictor`

    Returns
    -------
    None
        The function does not return any values. It generates and saves a video file 
        (as specified in `video_file` in each trial configuration) for each trial.

    Notes
    -----
    - This function assumes that the segmentation masks (stored in `masks_dict_file`) 
      have already been generated for each trial.
    - Trial count is inferred from the number of values provided for list-type parameters.
      All list-type parameters must have the same length.
    - The function relies on `read_config_yaml`, `extract_config_lens`, and 
      `get_trial_config` in utils to handle configuration management.
    - `write_output_video()` is responsible for the actual rendering and saving of the video.
    
    Warnings
    --------
    If the configuration specifies only a single `video_file` name while running multiple 
    trials, the output videos will be written to the same file, and results will be 
    overwritten. To prevent this, provide a list of unique `video_file` names — one for each trial.
    
    Examples
    --------
    >>> run_video_processing("template_configs.yaml", device=torch.device("cuda"))
    Creating video: ./output_trial1.mp4 from ./frames1 and ./generated_frame_masks1.pkl
    Creating video: ./output_trial2.mp4 from ./frames2 and ./generated_frame_masks2.pkl
    """
    # Load the YAML configuration file
    configs = read_config_yaml(configs)
    
    # Retrieve trial count from the length of values provided for each configuration key
    trial_count = extract_config_lens(configs)
    print(f"There are {trial_count} trials provided for processing")

     # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = get_trial_config(configs,i)
        
        # Write the output with modified trial configs
        print(f"Creating video: {trial_config['video_file']} from {trial_config['frame_dir']} and {trial_config['masks_dict_file']}")

        write_output_video(
            frame_dir = trial_config["frame_dir"],
            frame_masks_file = trial_config["masks_dict_file"],
            video_file=trial_config["video_file"],
            out_fps=trial_config["out_fps"],
            video_frame_size=trial_config["video_frame_size"],
            fps=trial_config["fps"],
            SAM2_start=trial_config["SAM2_start"],
            font_size=trial_config["font_size"],
            font_color=trial_config["font_color"],
            alpha=trial_config["alpha"],
            device=device
            )
        
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

def draw_masks(mask_dict, frame_path, colors, device, alpha=0.6):
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
    device : torch.device 
        A `torch.device` class specifying the device to use for mask drawing 

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

def write_output_video(frame_dir, frame_masks_file, video_file, out_fps, 
                       video_frame_size, fps, SAM2_start, font_size=16, font_color="red", alpha=0.6, device="cuda"):
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
    video_file : str
        The name of the video file to be created demonstrating the generated 
        masks on each frame.
    out_fps : int
        The frames per second for the video 
    video_frame_size : list or tuple of ints
        Specifies the frame size for the video, with the first 
        element representing the width and the second corresponding
        to the height
    fps : int 
        The FPS of the unreduced video that the annotations were 
        initially created for
    SAM2_start : int 
        Value that ensures the annotated frame value matches up with 
        the fames that will be ingested by SAM2
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
                           video_file="./test_video.mp4", out_fps=3, 
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
    video = cv2.VideoWriter(video_file, fourcc, out_fps, (width, height))

    # Write each image to the video and draw masks on images that contain them
    for frame_idx, img_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):

        # Draw masks on the frame, if they exist
        image, centroids = draw_masks(mask_dict=frame_masks[frame_idx], frame_path=img_path, 
                                      colors=colors, device=device, alpha=alpha)

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
        ax.set_title(f"SAM2 frame: {frame_idx}, Annotation frame: {frame_idx * (fps/out_fps) + SAM2_start}", fontsize=16)

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