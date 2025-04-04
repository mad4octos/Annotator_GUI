import yaml
import torch
import utils
from sam2_fish_segmenter import SAM2FishSegmenter

def run_batch_processing(config_file, device):
    # Load the YAML configuration file
    configs = utils.read_config_yaml(config_file)
    
    # Retrieve list of configuration keys
    keys = utils.extract_keys(configs)
    
    # Extract lists of frame directories, annotation files, and output files
    frame_dirs = configs.get("frame_dir", [])
    annotation_files = configs.get("annotations_file", [])
    mask_files = configs.get("masks_dict_file", [])
    video_files = configs.get("video_file", [])
    
    # Ensure main lists are of the same length
    if not (len(frame_dirs) == len(mask_files) == len(annotation_files) == len(video_files)):
        raise ValueError("Mismatch between number of annotation files, frame directories, mask files, and video files")  
        # Note: the video files are not used in this script, but will be used later in create_video.py, and 
        # I think it's better to be aware of this issue before proceeding.  
        
    # Calculate number of trials    
    trial_count = len(frame_dirs)
    print(f"Number of trials to process: {trial_count}")
    
    trials = list(zip(frame_dirs, annotation_files, mask_files))
    
    # Iterate over each trial and run segmentation
    for frame_dir, annotation_file, mask_file in trials: 
        trial_config = configs.copy()
        trial_config["frame_dir"] = frame_dir
        trial_config["annotations_file"] = annotation_file
        trial_config["masks_dict_file"] = mask_file
        
        # Get trial index
        i = trials.index((frame_dir, annotation_file, mask_file))
        
        # Iterate through configuration keys check for any other values that changed between trials
        for key in keys:
            trial_config[key] = utils.get_trial_value(configs, key, i, trial_count)
        
        # Initialize the segmenter with modified trial configs
        segmenter = SAM2FishSegmenter(configs = trial_config, device = device)
        print(f"Processing: {frame_dir} with {annotation_file} and saving masks to {mask_file}")
        segmenter.run_propagation()

configs= "./batch_template_configs.yaml"
        
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_batch_processing(configs, device)