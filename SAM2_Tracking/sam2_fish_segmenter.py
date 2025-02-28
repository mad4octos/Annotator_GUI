import utils 
import plot_utils
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

        if self.configs["save_jpgs"]:
            # Create directory that will store saved frames with masks, if it does not exist 
            os.makedirs(self.configs["jpg_save_dir"], exist_ok=True)

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
        annotations : List of dict
            List of dictionaries specifying points with keys: 
            Frame, fishLabel, Location, and clickType

        Raises
        ------
        ValueError
            If `annotations` was not a `list` of `dict`

        Examples
        --------
        >>> segmenter.add_annotations(annotations=my_annotations)
        """

        if not isinstance(annotations, list):
            raise TypeError("annotations should be of type list!")

        for annotation in annotations:

            if not isinstance(annotation, dict):
                raise TypeError(f"annotations element {annotation} is not of type dict!")

            # Extract values from the current annotation
            ann_frame_idx = annotation['Frame']  # Frame index
            ann_obj_id = int(annotation['fishLabel'])  # Object ID
            points = np.array([annotation['Location']], dtype=np.float32)  # Point coordinates
            labels = np.array([annotation['clickType']], dtype=np.int32)  # Positive/Negative click

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

    def run_propagation(self):
        """
        Propagates the prompts to get the masklet across the video using the 
        class predictor and inference state. If configuration variable 
        `save_jpgs` is True, generates frame JPGs with segmentation masks
        drawn on them. If configuration variable `save_masks` is True, 
        creates a dictionary of sparse Tensors representing the masks and 
        saves these to a pkl file specified by configuration variable 
        `masks_dict_file`. 

        Examples
        --------
        >>> segmenter.run_propagation()
        """

        if self.configs["save_jpgs"]:
            # Generate a list of RGB colors for segmentation masks 
            colors = plot_utils.get_spaced_colors(100)

        if self.configs["save_masks"]:
            # Initialize dictionary of masks for each frame
            frame_masks = {}

        # Perform prediction of masklets across video frames 
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            # TODO: In the future, we may want to modify the start and number of frames for propagation. If so, see:
            # https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L546

            # Create Bool mask and delete unneeded tensor
            bool_masks = out_mask_logits > 0.0
            del out_mask_logits

            # There's an extra dimension (1) to the masks, remove it
            bool_masks = bool_masks.squeeze(1)

            if self.configs["save_masks"]: 
                # Convert mask tensor to sparse format and store it
                frame_masks[out_frame_idx] = bool_masks.to_sparse().cpu()

            if self.configs["save_jpgs"]: 
                utils.draw_and_save_frame_seg(bool_masks=bool_masks, img_save_dir=self.configs["jpg_save_dir"], 
                                              frame_paths=self.frame_paths, out_frame_idx=out_frame_idx, 
                                              out_obj_ids=out_obj_ids, colors=colors, font_size=self.configs["font_size"], 
                                              font_color=self.configs["font_color"], alpha=self.configs["alpha"])

        if self.configs["save_masks"]: 
            # Save frame_masks as pkl file 
            with open(self.configs["masks_dict_file"], "wb") as file:
                    pickle.dump(frame_masks, file)
