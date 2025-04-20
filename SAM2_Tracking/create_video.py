import utils
import torch 

# Specify the path to the configuration YAML file
configs = "./template_configs.yaml"

# Run batch processing
utils.run_video_processing(configs)
