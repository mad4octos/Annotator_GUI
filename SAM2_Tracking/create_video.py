from utils import read_config_yaml, write_output_video

# Specify the path to the configuration YAML file
# yaml_file_path = "./test_configs.yaml"
yaml_file_path = "/projects/brre2566/annotator_gui_configs/test_configs.yaml"

# Read and load the configuration YAML
configs = read_config_yaml(yaml_file_path)

write_output_video(masked_imgs_dir=configs["jpg_save_dir"], video_file=configs["video_file"], 
                   video_fps=configs["video_fps"], video_frame_size=configs["video_frame_size"])