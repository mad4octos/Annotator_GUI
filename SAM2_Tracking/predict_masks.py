from utils import *

#USER EDITS HERE- # this should become variables that can be input from the run_vidpred_seq.sh, not hard coded here
annotations_file = "BA_051524_site2_east_A_Left_GX137102_annotations.npy" 

fps = 24 # Replace with your fps
SAM2_start=0 #Replace with the SAM2_start, as provided by your project and used in your GUI annotations
video_dir="./GX137102_frames"
####### READ NPY FILE CONTAINING GUI ANNOTATIONS###########

annotations = np.load(annotations_file, allow_pickle=True)

# Filter out "Fish_Fam" key and modify "Frame" values
annotations_filtered = []
for ann in annotations:
    filtered_ann = {}
    for key, value in ann.items():
        if key != "Fish_Fam":
            if key == "Frame":
                frame_value = (value - SAM2_start) / (fps / 3)
                #Check if frame_value is a decimal
                if not frame_value.is_integer():
                    rounded_frame_value = round(frame_value)
                    print(f"Warning: Frame value {frame_value:.2f} was rounded to {rounded_frame_value}")
                    frame_value = rounded_frame_value
                filtered_ann[key] = int(frame_value)
            else:
                filtered_ann[key] = value
    annotations_filtered.append(filtered_ann)


print('Filtered and Rounded Annotations:', annotations_filtered)
#Note: this data type is a 1dimenional numpy.ndarray. Each element of the array is a dict. 

# Read in pre-initilized model
device = torch.device("cuda")
save_points_dir = "save_points"
updated_checkpoint_path = os.path.join(save_points_dir, "sam2_full_model.pt")
predictor = torch.load(updated_checkpoint_path, map_location=device)
predictor.to(device)  # Ensure it's on the correct device

print("Model successfully loaded!")

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print("input frames:")
print(frame_names)

######### INFERENCE STATE ###########

inference_state = predictor.init_state(video_path=video_dir)

##### USE ANNOTATIONS FROM GUI TO PREDICT MASKS  ########
prompts = {}

# Iterate over the annotations and process each one
for i, annotation in enumerate(annotations_filtered):
    # Extract values from the current annotation
    ann_frame_idx = annotation['Frame']  # Frame index
    ann_obj_id = int(annotation['fishLabel'])  # Object ID
    points = np.array([annotation['Location']], dtype=np.float32)  # Point coordinates
    labels = np.array([annotation['clickType']], dtype=np.int32)  # Positive/Negative click

    # Update the prompts dictionary
    if ann_obj_id not in prompts:
        # Initialize with the first points and labels for this object ID
        prompts[ann_obj_id] = (points, labels)
    else:
        # Append the new points and labels
        existing_points, existing_labels = prompts[ann_obj_id]
        updated_points = np.vstack((existing_points, points))
        updated_labels = np.hstack((existing_labels, labels))
        prompts[ann_obj_id] = (updated_points, updated_labels)

    # Explicitly call predictor.add_new_points_or_box for this specific annotation
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # Print debug information for verification
    print(f"Iteration {i}: Frame: {ann_frame_idx}, Object ID: {ann_obj_id}, Points: {points}, Labels: {labels}")

#### SAVE POINT ####
updated_model_path = os.path.join(save_points_dir, "sam2_full_model.pt")
torch.save(predictor, updated_model_path)
print(f"SAM2 Full Model with predicted masks saved to: {updated_checkpoint_path}.")

with open(os.path.join(save_points_dir, "inference_state.pkl"), "wb") as f:
	pickle.dump(inference_state, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"SAM2 Inference State saved to: {save_points_dir}.")

updated_checkpoint_path=os.path.join(save_points_dir, "updated_checkpoint.pt")
torch.save(predictor.state_dict(), updated_checkpoint_path)
print(f"Model checkpoint saved to: {updated_checkpoint_path}")




'''
#De-bugging below:
# Print the entire prompts dictionary
print("Prompts:", prompts)
#visualize first frame with all click points:
ann_frame_idx = 0
plt.figure(figsize=(6,4))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("test_frame_0.png") 
print("Check test_frame.png to ensure points align with fish and fish is masked")

'''