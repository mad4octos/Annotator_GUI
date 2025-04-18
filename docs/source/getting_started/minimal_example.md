# Minimal Example 

# Configuration YAML

TODO: put in details here 

## Generating Masks

```
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
```

# Creating Masked Videos

```
from utils import read_config_yaml, write_output_video
import torch 

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read and load the configuration YAML
configs = read_config_yaml(yaml_file_path)

write_output_video(frame_dir=configs["frame_dir"], frame_masks_file=configs["masks_dict_file"], 
                   video_file=configs["video_file"], video_fps=configs["video_fps"], 
                   video_frame_size=configs["video_frame_size"], fps=configs["fps"],
                   SAM2_start=configs["SAM2_start"], font_size=configs["font_size"], 
                   font_color=configs["font_color"], alpha=configs["alpha"], device=device)
```