import utils
import torch 

# Specify the path to the configuration YAML file
configs = "./template_configs.yaml"

# Specify the device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.check_configs(configs, device)
