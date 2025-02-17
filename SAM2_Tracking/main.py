from workflow_functions import * 
import sys 
import os 

# Specify the path to the configuration YAML file
yaml_file_path = "./test_configs.yaml"
# yaml_file_path = "./GX137102_configs.yaml"

# Read and load the configuration YAML
configs = utils.read_yaml(yaml_file_path)

# Append install directory so we can use sam2_checkpoints and model configurations 
sys.path.append(configs["sam2_install_dir"])

# Create directory that will store video_segments, if it does not exist 
os.makedirs(configs["video_seg_save_dir"], exist_ok=True)

predictor = initialize_predictor(configs["sam2_checkpoint"], configs["model_cfg"])

inference_state = predict_masks(predictor, configs["annotations_file"], configs["fps"], configs["SAM2_start"], 
                                configs["video_dir"], configs["offload_video_to_cpu"], 
                                configs["offload_state_to_cpu"], configs["async_loading_frames"])

propagate_masks(predictor, inference_state, video_dir=configs["video_dir"], 
                video_seg_batch=configs["video_seg_batch"], video_seg_sav_dir=configs["video_seg_save_dir"])

#Save SAM2 model checkpoint
# updated_checkpoint_path=os.path.join(save_points_dir, "updated_checkpoint_2.pt")
# torch.save(predictor.state_dict(), updated_checkpoint_path)
# print(f"Model checkpoint saved to: {updated_checkpoint_path}")