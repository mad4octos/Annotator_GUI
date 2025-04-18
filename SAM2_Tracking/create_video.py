import utils
import torch 

# Specify the path to the configuration YAML file
configs = "./template_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run batch processing
utils.run_video_processing(configs, device)
