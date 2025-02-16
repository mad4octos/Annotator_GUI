from utils import * 


video_dir = "./frames"
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print("input frames:")
print(frame_names)





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
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    if out_frame_idx
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
    # Save every 5 frames - to a file (pickle dictionary, named according to frames)
    # ReSet video_segments={}
    # go again
    


print(predictor.propagate_in_video(inference_state))


with open(os.path.join(save_points_dir, "video_segments.pkl"), "wb") as f:
	pickle.dump(video_segments, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save(os.path.join(save_points_dir, "out_mask_logits.npy"), out_mask_logits.cpu().numpy())

updated_checkpoint_path = os.path.join(save_points_dir, "sam2_full_model.pt")
torch.save(predictor, updated_checkpoint_path)