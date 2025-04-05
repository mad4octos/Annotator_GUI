import yaml
import torch
import utils
from sam2_fish_segmenter import SAM2FishSegmenter

def run_batch_processing(config_file, device):
    # Load the YAML configuration file
    configs = utils.read_config_yaml(config_file)
    
    # Retrieve dictionary of the length of values provided for each configuration key
    trial_count = utils.extract_config_lens(configs)
    
    # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = {
            key: value[0] if len(value) == 1 else value[i]
            for key, value in configs.items()
        }
        
        # Initialize the segmenter with modified trial configs
        segmenter = SAM2FishSegmenter(configs = trial_config, device = device)
        print(f"Processing Trial {i}: Frames from {trial_config['frame_dir']}, Annotations from {trial_config['annotations_file']}, Masks saving to {trial_config['masks_dict_file']}")
        segmenter.run_propagation()

configs= "./batch_template_configs.yaml"
        
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process batch of videos
run_batch_processing(configs, device)