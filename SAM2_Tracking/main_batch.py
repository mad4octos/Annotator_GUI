import yaml
import torch
import utils
from sam2_fish_segmenter import SAM2FishSegmenter

def run_batch_processing(config_file, device):
    # Load the YAML configuration file
    configs = utils.read_config_yaml(config_file)
    
    # Retrieve trial count from the length of values provided for each configuration key
    trial_count = utils.extract_config_lens(configs)
    
    # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = utils.get_trial_config(configs, i)

        # Initialize the segmenter with modified trial configs
        segmenter = SAM2FishSegmenter(configs = trial_config, device = device)
        print(f"Processing Trial {i}: Frames from {trial_config['frame_dir']}, Annotations from {trial_config['annotations_file']}, Masks saving to {trial_config['masks_dict_file']}")
        segmenter.run_propagation()

config_file = "./template_configs_batch.yaml"
        
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process batch of videos
run_batch_processing(config_file, device)