from sam2_fish_segmenter import SAM2FishSegmenter
import torch
import utils  

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segmenter = SAM2FishSegmenter(configs=yaml_file_path, device=device)

segmenter.run_propagation()