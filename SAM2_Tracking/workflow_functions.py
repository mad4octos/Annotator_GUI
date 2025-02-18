import torch 
from sam2.build_sam import build_sam2_video_predictor
import numpy as np 
import os 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import utils 
import pickle 

def initialize_predictor(sam2_checkpoint, model_cfg):

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    return predictor

def predict_masks(predictor, annotations_file, fps, SAM2_start, video_dir, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False):

    ####### READ NPY FILE CONTAINING GUI ANNOTATIONS###########
    annotations = np.load(annotations_file, allow_pickle=True)

    annotations_filtered  = utils.filter_annotations(annotations_file, fps, SAM2_start)

    ######### INFERENCE STATE ###########
    # Reference of init: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L63
    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=offload_video_to_cpu, 
                                           offload_state_to_cpu=offload_state_to_cpu, async_loading_frames=async_loading_frames)

    utils.process_annotations_and_predict(annotations_filtered, predictor, inference_state)

    return inference_state

def propagate_masks(predictor, inference_state, video_dir, video_seg_batch, video_seg_sav_dir):

    num_frames = utils.count_files_in_directory(video_dir)

    # you can modify the start and number of frames 
    # see: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L546

    video_segments = {}  # video_segments contains the per-frame segmentation results
    batch_id = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        if (out_frame_idx != 0) and ((out_frame_idx % video_seg_batch == 0) or (out_frame_idx == num_frames - 1)):
            # Save the dictionary with pickle
            with open(video_seg_sav_dir + "video_segments_" + str(batch_id) + ".pkl", "wb") as file:
                pickle.dump(video_segments, file)
                
            del video_segments
            video_segments = {}  # video_segments contains the per-frame segmentation results
            batch_id += 1

            # torch.cuda.empty_cache()

def write_output_video(output_video_path, fps, out_fps, SAM2_start, frame_size, video_dir, video_segments):

    ### Save output as an .mp4 file
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, frame_size)

    for out_frame_idx in range(0, len(frame_names)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}/{out_frame_idx * (fps/3) + SAM2_start}")

        image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        ax.imshow(image)

        if out_frame_idx in video_segments:
            print(f"out_frame_idx ns: {out_frame_idx}")
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                utils.show_mask(out_mask, ax, obj_id=out_obj_id)

                #Calculate the max y coordinate for the mask
                mask_coords = np.column_stack(np.where(out_mask))
                if mask_coords.size>0: #Ensure mask contains points
                    min_y = mask_coords[:,0].min()
                    min_x = mask_coords[mask_coords[:,0] == min_y][:,1].mean()
                    utils.add_text(ax, f"ID: {out_obj_id}", position=(min_x, min_y - 10))
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape((height, width, 4))
        frame_rgb = frame[:, :, :3]
        
        color_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(color_frame)

        plt.close(fig)

    video_writer.release()
        
    # TODO: big time suck is utils.show_mask(out_mask, ax, obj_id=out_obj_id) due to ax.imshow(mask_image)
    # is there a way to create all of the masks at once? 
    # we could instead apply coloring in propagate_masks