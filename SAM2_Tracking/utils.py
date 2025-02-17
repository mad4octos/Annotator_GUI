import numpy as np
import matplotlib.pyplot as plt
import yaml 
import glob
import pickle
import os  

# Function to read configuration YAML file
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # safe_load avoids arbitrary code execution
    return config

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def add_text(ax, text, position, fontsize=12, color='yellow'):
    """Adds a text box to the given axis."""
    x, y = position
    # Transform position from pixel coordinates to axis coordinates
    x_axis, y_axis = ax.transData.transform((x, y))
    ax.figure.text(
        x_axis / ax.figure.bbox.width,
        y_axis / ax.figure.bbox.height,
        text,
        fontsize=fontsize,
        color=color,
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'),
        ha='center',  # Center-align horizontally
        va='bottom'   # Align the text above the mask
    )

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
        
def count_files_in_directory(directory):
    count = sum(1 for entry in os.scandir(directory) if entry.is_file())
    return count

def reconstruct_video_segments(video_seg_save_dir):
    # TODO: figure out if there is a way where we don't need to reconstruct the full dictionary 

    video_segments = {}

    # Grab all .pkl files in video_seg_save_dir
    pickle_file_paths = glob.glob(f'{video_seg_save_dir}*.pkl')

    # Sort the file paths
    pickle_file_paths = sorted(pickle_file_paths)
    
    # List all pickle files in the specified directory
    for pickle_path in pickle_file_paths:
        try:
            # Open the pickle file and load the dictionary
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                # Merge the data (assuming it's a dictionary)
                if isinstance(data, dict):
                    video_segments.update(data)
                else:
                    print(f"Warning: {filename} does not contain a dictionary.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return video_segments