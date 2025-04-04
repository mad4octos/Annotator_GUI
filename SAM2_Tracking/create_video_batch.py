from utils import read_config_yaml, write_output_video, extract_keys, get_trial_value
import torch 

# Specify the path to the configuration YAML file
configs = "./template_configs_batch.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_batch_video_processing(configs, device):
    # Load the YAML configuration file
    configs = read_config_yaml(configs)
    
    # Retrieve list of configuration keys
    keys = extract_keys(configs)
    
    # Extract lists of main relevant parameters
    frame_dirs = configs.get("frame_dir", [])
    mask_files = configs.get("masks_dict_file", [])
    video_files = configs.get("video_file", [])
    
    # Ensure main lists are of the same length
    if not (len(frame_dirs) == len(mask_files) == len(video_files)):
        raise ValueError("Mismatch between number of frame directories, mask files, and video files")  
        
    # Calculate number of trials
    trial_count = len(frame_dirs)
    print(f"Number of videos to create: {trial_count}")
    
    # Create list of trials
    trials = list(zip(frame_dirs, mask_files, video_files))
    
    # Run video creation for each trial
    for frame_dir, mask_file, video_file in trials:
        print(f"Creating video: {video_file} from {frame_dir} and {mask_file}")
        trial_config = configs.copy()
        
        # Get trial index
        i = trials.index((frame_dir, mask_file, video_file))
    
        # Iterate through configuration keys check for any other values that changed between trials
        for key in keys:
            trial_config[key] = get_trial_value(configs, key, i, trial_count)

        write_output_video(
            frame_dir = trial_config["frame_dir"],
            frame_masks_file = trial_config["masks_dict_file"],
            video_file=trial_config["video_file"],
            out_fps=trial_config["out_fps"],
            video_frame_size=configs["video_frame_size"],
            fps=trial_config["fps"],
            SAM2_start=trial_config["SAM2_start"],
            font_size=trial_config["font_size"],
            font_color=trial_config["font_color"],
            alpha=trial_config["alpha"],
            device=device
            )

run_batch_video_processing(configs, device)
