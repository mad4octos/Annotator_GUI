from utils import * 


# Read in pre-initilized model with predicted masks
device = torch.device("cuda")
save_points_dir = "save_points"
updated_checkpoint_path = os.path.join(save_points_dir, "sam2_full_model.pt")
predictor = torch.load(updated_checkpoint_path, map_location=device)
predictor.to(device)  # Ensure it's on the correct device

print("Model successfully loaded!")

######### INFERENCE STATE ###########

with open(os.path.join(save_points_dir, "inference_state.pkl"), "rb") as f:
    inference_state = pickle.load(f)

print("inference_state successfully loaded!")

# Run video propagation- create masks for each ID on each frame
batch_size = 10 # Number of frames per batch
batch_index = 0 # Track batch number
video_segments = {}  # video_segments contains the per-frame segmentation results

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx]= {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    # Check if batch is complete
    if (out_frame_idx + 1) % batch_size == 0:
        #Save the current batch to a pickle file
        file_path = os.path.join(save_points_dir, f'video_segments_batch_{batch_index}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(video_segments, f)
        # REset the dictionary and increment batch index
        video_segments={}
        batch_index += 1
if video_segments:
    file_path = os.path.join(save_points_dir, f'video_segments_batch_{batch_index}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(video_segments, f)
    
# Save masks
np.save(os.path.join(save_points_dir, "out_mask_logits.npy"), out_mask_logits.cpu().numpy())

#Save Full Model
updated_checkpoint_path = os.path.join(save_points_dir, "sam2_full_model.pt")
torch.save(predictor, updated_checkpoint_path)