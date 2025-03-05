import utils 
import plot_utils
import shutil
import os 
import sys 
import torch 
import numpy as np 
import pickle 
from sam2.build_sam import build_sam2_video_predictor


class SAM2FishSegmenter:
    """
    A class used to conduct fish segmentation using SAM2. Specifically, 
    it provides the ability to create a SAM2 predictor, create an 
    inference state, add annotations, and run the propagation of the 
    segmentation through a provided video. 

    Methods
    -------
    __init__(self, configs, device)
        Initializes the predictor model and sets `self.configs`
    set_inference_state(self)
        Obtains the inference state for `self.predictor` and 
        sets `self.frame_paths`
    add_annotations(self, annotations)
        Adds provided annotations to predictor
    run_propagation(self)
        Propagates the prompts to get the masklet across the video using the 
        class predictor and inference state. Additionally, creates masked JPG
        frames or pkl of sparse tensors, based on provided configurations.
    """  

    def __init__(self, configs=None, device=None):
        """
        Initializes the predictor model and sets `self.configs`

        Parameters
        ----------
        configs : dict or str
            A dictionary of configurations or a yaml file 
            specifying configurations
        device : torch.device 
            A `torch.device` class specifying the device to use for `build_sam2_video_predictor` 

        Raises
        ------
        ValueError
            If `configs` is not a `str` or `dict`. 
        RuntimeError
            If `device.type` is not equal to `cuda`. 

        Examples
        --------
        >>> yaml_file_path = "./template_configs.yaml"
        >>> device = torch.device('cuda')
        >>> segmenter = SAM2FishSegmenter(configs=yaml_file_path, device=device)
        """

        if isinstance(configs, str): 
            # Read and load the configuration YAML
            self.configs = utils.read_config_yaml(configs)
            # TODO: a routine for checking the provided configs should be ran

        elif isinstance(configs, dict):
            # Set class variable configs to the provided dict
            self.configs = configs
            # TODO: a routine for checking the provided configs should be ran

        else:
            raise TypeError("configs was not a str or dict!")

        if self.configs["copy_frame_dir"]:
            # Copy the entire directory # TODO: should we be doing this for the user? 
            print(f"Since copy_frame_dir=True, copying directory {self.configs["frame_dir"]} to {self.configs["jpg_save_dir"]} ...")
            shutil.copytree(self.configs["frame_dir"], self.configs["jpg_save_dir"], dirs_exist_ok=True)

        # TODO: determine if this is the best place to put this, might be worth removing
        # Append install directory so we can use sam2_checkpoints and model configurations 
        sys.path.append(self.configs["sam2_install_dir"])

        # Set appropriate data types for SAM2
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            raise RuntimeError(f"Device of type {device.type} not supported!")  

        # Initialize SAM2 video predictor 
        # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py#L100
        # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L19
        self.predictor = build_sam2_video_predictor(self.configs["model_cfg"], ckpt_path=self.configs["sam2_checkpoint"], 
                                                    device=device, non_overlap_masks=self.configs["non_overlap_masks"])


    def set_inference_state(self):
        """
        Obtains the inference state for `self.predictor` for a provided 
        video path (specified by `self.configs["frame_dir"]`) and sets 
        it as `self.inference_state`. Additionally, sets 
        `self.frame_paths`, which are all of the JPG paths representing 
        the frames.

        Raises
        ------
        ValueError
            If `self.configs["frame_dir"]` was not of type `str`. 

        Examples
        --------
        >>> segmenter.set_inference_state()
        """

        if not isinstance(self.configs["frame_dir"], str): 
            raise TypeError(f"config frame_dir was not of type str!")

        # Gather all the JPG paths representing the frames 
        self.frame_paths = utils.get_jpg_paths(self.configs["frame_dir"])

        # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L42
        self.inference_state = self.predictor.init_state(video_path=self.configs["frame_dir"], 
                                                         offload_video_to_cpu=self.configs["offload_video_to_cpu"], 
                                                         offload_state_to_cpu=self.configs["offload_state_to_cpu"], 
                                                         async_loading_frames=self.configs["async_loading_frames"])


    def add_annotations(self, annotations=None):
        """
        Adds provided `annotations` to `self.predictor` using the 
        `SAM2VideoPredictor` method `add_new_points_or_box`.

        Parameters
        ----------
        annotations : Pandas.DataFrame
            DataFrame specifying points with index `obj_id_name`
            and columns: `frame_idx_name`, `points_name`, and 
            `labels_name`, with values as specified in the 
            configuration yaml

        Raises
        ------
        ValueError
            If `annotations` was not a `Pandas.DataFrame`

        Examples
        --------
        >>> segmenter.add_annotations(annotations=ann_df)
        """

        if not isinstance(annotations, pd.DataFrame):
            raise TypeError("annotations should be a Pandas DataFrame!")

        for index, row in annotations.iterrows():

            # Extract values from the current annotation
            ann_frame_idx = row[self.configs['frame_idx_name']]  # Frame index
            ann_obj_id = int(index)  # Object ID
            points = np.array([row[self.configs['points_name']]], dtype=np.float32)  # Point coordinates
            labels = np.array([row[self.configs['labels_name']]], dtype=np.int32)  # Positive/Negative click

            # Explicitly call predictor.add_new_points_or_box for annotation
            # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L161
            # TODO: determine if it is helpful to add out_obj_ids and out_mask_logits as class variables
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

    def get_masks(self, start_frame_idx=None, max_frame_num_to_track=None):
        """
        Propagates the prompts to get the masklet across the video using the 
        class predictor and inference state. If configuration variable 
        `save_jpgs` is True, overwrites the JPGs in `jpg_save_dir` with 
        segmentation masks drawn on them. If configuration variable `save_masks` 
        is True, creates a dictionary of sparse Tensors representing the masks 
        and saves these to a pkl file specified by configuration variable 
        `masks_dict_file`. 

        Parameters
        ----------
        start_frame_idx : None or int
            The start frame for SAM2 `propagate_in_video`
        max_frame_num_to_track : None or int 
            The number of frames to track for SAM2 `propagate_in_video`

        Returns
        -------
        dict or None 
            If configuration `save_masks` is `True`, returns `frame_masks` 
            filled with each object's mask for each frame, else returns `None`

        Examples
        --------
        >>> segmenter.run_propagation(start_frame_idx=0, max_frame_num_to_track=100)
        """

        if self.configs["save_jpgs"]:
            # Generate a list of RGB colors for segmentation masks 
            colors = plot_utils.get_spaced_colors(100)

        if self.configs["save_masks"]:
            # Initialize dictionary of masks for each frame
            frame_masks = {key: {} for key in range(len(self.frame_paths))}


        # Perform prediction of masklets across video frames 
        # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L546
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state,
                                                                                             start_frame_idx=start_frame_idx, 
                                                                                             max_frame_num_to_track=max_frame_num_to_track):

            # Create Bool mask and delete unneeded tensor
            bool_masks = out_mask_logits > 0.0
            del out_mask_logits

            # There's an extra dimension (1) to the masks, remove it
            bool_masks = bool_masks.squeeze(1)

            if self.configs["save_masks"]: 
                # Convert mask tensor to sparse format and store it
                for obj_id in out_obj_ids:
                    frame_masks[out_frame_idx][obj_id] = bool_masks.to_sparse().cpu()

            if self.configs["save_jpgs"]: 
                utils.draw_and_save_frame_seg(bool_masks=bool_masks, jpg_save_dir=self.configs["jpg_save_dir"], 
                                              frame_paths=self.frame_paths, out_frame_idx=out_frame_idx, 
                                              out_obj_ids=out_obj_ids, colors=colors, font_size=self.configs["font_size"], 
                                              font_color=self.configs["font_color"], alpha=self.configs["alpha"])

        if self.configs["save_masks"]: 
            return frame_masks

    def run_propagation(self):
        """
        Runs entire workflow: setting the inference state,
        collecting and adding annotations, getting SAM2
        provided masks, and saving masks. This function 
        expects annotations that have `labels_name` with 
        enter and exit values of 3 and 4, respectively.
        """

        # Set inference state for SAM2
        self.set_inference_state()

        # Get keys in annotations that will become DataFrame columns
        df_columns = [self.configs["frame_idx_name"], self.configs["labels_name"], 
                      self.configs["obj_id_name"], self.configs["points_name"]]

        # Convert annotations to a DataFrame and adjust frame values 
        annotations = utils.adjust_annotations(annotations_file=self.configs["annotations_file"], fps=self.configs["fps"], 
                                               SAM2_start=self.configs["SAM2_start"], df_columns=df_columns, 
                                               frame_col_name=self.configs["frame_idx_name"])

        # Get object frame chunks and modified annotations (that have labels_name rows with 3/4 values dropped)
        obj_frame_chunks, annotations = utils.get_frame_chunks_df(df=annotations)

        frame_masks = {}
        for obj_label in obj_frame_chunks.index:   # TODO: need to modify for multiple index values and change .loc to .iloc

            # Get the enter, exit, and number of frames for obj label 
            enter_frame = fish_frame_chunks.loc[obj_label]['EnterFrame']
            exit_frame = fish_frame_chunks.loc[obj_label]['ExitFrame']
            num_frames = exit_frame - enter_frame

            # Get all of the annotations for the given object label
            obj_annotation = annotations.loc[obj_label]

            # Get all annotations that have Frame values between enter_frame and exit_frame inclusive 
            annotation_chunk = obj_annotation[(obj_annotation[self.configs["frame_idx_name"]] >= enter_frame) & (obj_annotation[self.configs["frame_idx_name"]] <= exit_frame)]

            # Reset inference state for the new incoming annotations 
            self.predictor.reset_state(self.inference_state)    # TODO: might want to skip for the first entry? 

            # Add point annotations for provided annotation chunk 
            self.add_annotations(annotations=annotation_chunk)

            # Run propagation on annotation chunk of frames
            frame_masks += self.get_masks(start_frame_idx=enter_frame, max_frame_num_to_track=exit_frame)

        if self.configs["save_masks"]: 
            # Save frame_masks as pkl file 
            with open(self.configs["masks_dict_file"], "wb") as file:
                    pickle.dump(frame_masks, file)