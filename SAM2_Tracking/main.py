import torch
import utils

config_file = "./template_configs.yaml"
        
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process batch of videos
utils.run_segmentation(config_file, device)