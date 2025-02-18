from utils import read_yaml, reconstruct_video_segments
from workflow_functions import write_output_video

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Read and load the configuration YAML
configs = read_yaml(yaml_file_path)

# TODO: investigate better ways to construct the video 
# It will be very memory intensive to reconstruct the full video_segments
video_segments = reconstruct_video_segments(configs["video_seg_save_dir"])

write_output_video(configs["output_video_path"], configs["fps"], configs["out_fps"], configs["SAM2_start"], 
                   configs["frame_size"], configs["video_dir"], video_segments)