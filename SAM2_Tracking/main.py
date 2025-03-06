from sam2_fish_segmenter import SAM2FishSegmenter
import torch
import utils  

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segmenter = SAM2FishSegmenter(configs=yaml_file_path, device=device)

segmenter.set_inference_state()

# TODO: should we create a config file specifically for these? 
annotations_filtered  = utils.filter_annotations(annotations_file=segmenter.configs["annotations_file"], 
                                                 fps=segmenter.configs["fps"], SAM2_start=segmenter.configs["SAM2_start"])

segmenter.add_annotations(annotations=annotations_filtered)

segmenter.run_propagation()