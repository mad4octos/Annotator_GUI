from sam2_fish_segmenter import SAM2FishSegmenter
import torch
import utils  

# Specify the path to the configuration YAML file
# yaml_file_path = "./template_configs.yaml"
yaml_file_path = "/projects/brre2566/annotator_gui_configs/test_configs.yaml"

# Set device for PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segmenter = SAM2FishSegmenter(configs=yaml_file_path, device=device)

segmenter.set_inference_state()

# TODO: should we create a config file specifically for these? 
# annotations_filtered  = utils.filter_annotations(annotations_file=segmenter.configs["annotations_file"], 
#                                                  fps=segmenter.configs["fps"], SAM2_start=segmenter.configs["SAM2_start"])
import numpy as np 
import pandas as pd 
annotations_filtered = np.load(segmenter.configs["annotations_file"], allow_pickle=True)
annotations_filtered = pd.DataFrame(list(annotations_filtered))

fish_frame_chunks, annotations_filtered = utils.get_frame_chunks_df(df=annotations_filtered)

# print(f"fish_frame_chunks: {fish_frame_chunks}")

# Create list of keys corresponding to the Frame value 
keys = [str(i) for i in range(len(segmenter.frame_paths))]

# Initialize a dictionary with provided keys and set values as empty dictionaries
frame_masks = {key: {} for key in keys}

annotations = annotations_filtered
for fish_label in fish_frame_chunks.index:

    enter_frame = fish_frame_chunks.loc[fish_label]['EnterFrame']
    exit_frame = fish_frame_chunks.loc[fish_label]['ExitFrame']
    num_frames = exit_frame - enter_frame

    # Get all of the annotations for the given fishLabel
    annotation_fish = annotations.loc[fish_label]

    # Get all annotations that have Frame values between enter_frame and exit_frame inclusive 
    annotation_chunk = annotation_fish[(annotation_fish['Frame'] >= enter_frame) & (annotation_fish['Frame'] <= exit_frame)]

    # Reset inference state for the new incoming annotations 
    segmenter.predictor.reset_state(segmenter.inference_state)

    # Add point annotations for provided annotation chunk 
    segmenter.add_annotations(annotations=annotation_chunk)

    # Run propagation on annotation chunk of frames
    segmenter.run_propagation(start_frame_idx=enter_frame, max_frame_num_to_track=exit_frame)