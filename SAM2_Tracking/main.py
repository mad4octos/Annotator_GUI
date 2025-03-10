from sam2_fish_segmenter import SAM2FishSegmenter
import torch 

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize class using configurations 
segmenter = SAM2FishSegmenter(configs=yaml_file_path, device=device)

# Run mask propagation workflow 
segmenter.run_propagation()