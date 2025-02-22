from workflow_functions import * 
import sys 
import os 

# Specify the path to the configuration YAML file
yaml_file_path = "./template_configs.yaml"

# Read and load the configuration YAML
configs = utils.read_config_yaml(yaml_file_path)

# Append install directory so we can use sam2_checkpoints and model configurations 
sys.path.append(configs["sam2_install_dir"])

if configs["save_jpgs"]:
    # Create directory that will store saved frames with masks, if it does not exist 
    os.makedirs(configs["jpg_save_dir"], exist_ok=True)

predictor = initialize_predictor(configs["sam2_checkpoint"], configs["model_cfg"], configs["non_overlap_masks"])

inference_state = predict_masks(predictor, configs["annotations_file"], configs["fps"], configs["SAM2_start"], 
                                configs["frame_dir"], configs["offload_video_to_cpu"], 
                                configs["offload_state_to_cpu"], configs["async_loading_frames"])

run_propagation(predictor, inference_state, frame_dir=configs["frame_dir"], save_jpgs=configs["save_jpgs"], 
                save_masks=configs["save_masks"], jpg_save_dir=configs["jpg_save_dir"], 
                masks_dict_file=configs["masks_dict_file"], font_size=configs["font_size"], 
                font_color=configs["font_color"], alpha=configs["alpha"])

#Save SAM2 model checkpoint
# updated_checkpoint_path=os.path.join(save_points_dir, "updated_checkpoint_2.pt")
# torch.save(predictor.state_dict(), updated_checkpoint_path)
# print(f"Model checkpoint saved to: {updated_checkpoint_path}")