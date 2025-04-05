import yaml
import torch
import utils
from sam2_fish_segmenter import SAM2FishSegmenter

def run_batch_processing(config_file, device):
    # Load the YAML configuration file
    configs = utils.read_config_yaml(config_file)
    
    # Retrieve dictionary of the length of values provided for each configuration key
    trial_count = utils.extract_config_lens(configs)

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