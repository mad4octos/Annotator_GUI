import utils
import torch 

def run_batch_video_processing(configs, device):
    # Load the YAML configuration file
    configs = utils.read_config_yaml(configs)
    
    # Retrieve trial count from the length of values provided for each configuration key
    trial_count = utils.extract_config_lens(configs)

     # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = utils.get_trial_config(configs,i)
        
        # Write the output with modified trial configs
        print(f"Creating video: {trial_config['video_file']} from {trial_config['frame_dir']} and {trial_config['masks_dict_file']}")

        utils.write_output_video(
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

# Specify the path to the configuration YAML file
configs = "./template_configs_batch.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run batch processing
run_batch_video_processing(configs, device)
