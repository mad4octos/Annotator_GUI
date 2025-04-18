import workflow

config_file = "./template_configs.yaml"
        
# Set device for PyTorch
device = "cuda"

# Process batch of videos
workflow.run_segmentation(config_file, device)