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

def filter_annotations(annotations_file, fps, SAM2_start):

    ####### READ NPY FILE CONTAINING GUI ANNOTATIONS###########
    annotations = np.load(annotations_file, allow_pickle=True)

    # Filter out "Fish_Fam" key and modify "Frame" values
    annotations_filtered = []
    for ann in annotations:
        filtered_ann = {}
        for key, value in ann.items():
            if key != "Fish_Fam":
                if key == "Frame":
                    frame_value = (value - SAM2_start) / (fps / 3)
                    #Check if frame_valu    e is a decimal
                    if not frame_value.is_integer():
                        rounded_frame_value = round(frame_value)
                        print(f"Warning: Frame value {frame_value:.2f} was rounded to {rounded_frame_value}")
                        frame_value = rounded_frame_value
                    filtered_ann[key] = int(frame_value)
                else:
                    filtered_ann[key] = value
        annotations_filtered.append(filtered_ann)

    return annotations_filtered    

def process_annotations_and_predict(annotations_filtered, predictor, inference_state):

    ##### USE ANNOTATIONS FROM GUI TO PREDICT MASKS  ########
    prompts = {}
    # Iterate over the annotations and process each one
    for i, annotation in enumerate(annotations_filtered):
        # Extract values from the current annotation
        ann_frame_idx = annotation['Frame']  # Frame index
        ann_obj_id = int(annotation['fishLabel'])  # Object ID
        points = np.array([annotation['Location']], dtype=np.float32)  # Point coordinates
        labels = np.array([annotation['clickType']], dtype=np.int32)  # Positive/Negative click

        # Update the prompts dictionary
        if ann_obj_id not in prompts:
            # Initialize with the first points and labels for this object ID
            prompts[ann_obj_id] = (points, labels)
        else:
            # Append the new points and labels
            existing_points, existing_labels = prompts[ann_obj_id]
            updated_points = np.vstack((existing_points, points))
            updated_labels = np.hstack((existing_labels, labels))
            prompts[ann_obj_id] = (updated_points, updated_labels)

        # Explicitly call predictor.add_new_points_or_box for this specific annotation
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
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

def draw_and_save_frame_seg(bool_masks, img_save_dir, frame_paths, out_frame_idx, out_obj_ids, colors, 
                            font_size=75, font_color="red", alpha=0.6):
    """
    Draws segmentation masks on top of the frame and saves 
    the generated image to `img_save_dir`. 

    Parameters
    ----------
    bool_masks : Tensor of bools
        A tensor of shape (number of masks, frame pixel height, frame pixel width)
        representing the generated segmentation masks
    img_save_dir : str
        The path where we want to save the generated image
    frame_paths : list of str
        A list of JPG paths representing the frames 
    out_frame_idx : int
        Index representing the frame we predicted the masks for 
    out_obj_ids : list of int
        A list of integers representing the ids for each mask
    colors : list of tuples of ints
        A list of tuples representing RGB colors for each segmentation mask
    font_size : int
        Font size for drawn object IDs
    font_color : str
        Color of font for the drawn object IDs
    alpha : float 
        Alpha value for the segmentation masks 

    Returns
    -------
    list
        List of sorted JPG paths
    """    

    # Draw each mask on top of image representing the frame
    image_w_seg = decode_image(frame_paths[out_frame_idx])
    for i in range(bool_masks.shape[0]):

        # Only draw masks that contain True values 
        if bool_masks[i].any():
            image_w_seg = draw_segmentation_masks(image_w_seg, bool_masks[i], colors=colors[i], alpha=alpha)

    # Convert image with drawn segmentation masks to PIL Image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(image_w_seg)

    # Draw text annotations
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default(size=font_size)

    # Compute centroids for all masks so we can place the object ID on the centroid 
    centroids = [plot_utils.get_centroid(mask) for mask in bool_masks]

    # Draw the object ID at the centroid
    for centroid, label in zip(centroids, out_obj_ids):
        if centroid:
            draw.text(centroid, str(label), fill=font_color, font=font)

    # Get frame name using the stem of the frame JPG
    frame_id = Path(frame_paths[out_frame_idx]).stem

    # Save the final image
    img_pil.save(img_save_dir + f"/{frame_id}.jpg")