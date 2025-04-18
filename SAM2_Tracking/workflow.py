import utils 
from sam2_fish_segmenter import SAM2FishSegmenter
import multiprocessing
import functools
import parallel_utils as pu


def serial_segmentation(configs, trial_count, device_input):
    """

    """

    device = torch.device(device_input)

    # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = utils.get_trial_config(configs, i)

        # Initialize the segmenter with modified trial configs
        segmenter = SAM2FishSegmenter(configs = trial_config, device = device)
        print(f"Processing Trial {i}: Frames from {trial_config['frame_dir']}, Annotations from {trial_config['annotations_file']}, Masks saving to {trial_config['masks_dict_file']}")
        segmenter.run_propagation()

def parallel_segmentation(configs, trial_count, device_input, num_workers):

    core_assignments = pu.get_core_assignments(num_workers)

    data = range(trial_count)

    # Create the partial function (allows us to pass in additional inputs)
    partial_func = functools.partial(pu.worker_function, num_workers=num_workers, device_input=device_input)

    lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0) # Rank counter starts at 0

    with multiprocessing.Pool(processes=num_workers, initializer=pu.init_worker, initargs=(core_assignments, lock, counter)) as pool:
        pool.map(partial_func, data)

def run_segmentation(config_file, device):
    """
    Runs SAM2 segmentation and mask propagation for one or more trial configurations.

    This function reads a YAML configuration file and processes it to extract 
    parameters for each trial. Each configuration parameter in the YAML file must 
    either be a single value (applied to all trials) or a list of values (one per trial). 
    For each trial, the function initializes a `SAM2FishSegmenter` object with the appropriate 
    configuration and executes the segmentation and propagation process.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file containing all segmentation parameters.
        Each parameter should either be a scalar (applied to all trials) or a list of values
        (with one entry per trial).
    device : torch.device 
            A `torch.device` class specifying the device to use for `build_sam2_video_predictor`

    Returns
    -------
    None
        The function does not return anything explicitly. However, it saves a pickled 
        dictionary of masks (as specified by `masks_dict_file` in each trial config) 
        for each trial after segmentation and propagation.

    Notes
    -----
    - The number of trials is determined by the length of list-type parameters in the 
      configuration file. All such parameters must have the same length.
    - The function uses `read_config_yaml`, `extract_config_lens`, and `get_trial_config` 
      as helpers in utils to parse and manage configurations.
    
    Warnings
    --------
    If the configuration specifies only a single `masks_dict_file` name while running multiple 
    trials, the output masks will be written to the same file, and results will be 
    overwritten. To prevent this, provide a list of unique `masks_dict_file` names — 
    one for each trial.
    
    Examples
    --------
    >>> run_segmentation("template_configs.yaml", device=torch.device("cuda"))
    Processing Trial 0: Frames from ./data/frames1, Annotations from ./data/annotations1.npy, Masks saving to ./generated_frame_masks1.pkl
    Processing Trial 1: Frames from ./data/frames2, Annotations from ./data/annotations2.npy, Masks saving to ./generated_frame_masks2.pkl
    """
    # Load the YAML configuration file
    # configs = utils.read_config_yaml(config_file)
    
    # Retrieve trial count from the length of values provided for each configuration key
    # trial_count = utils.extract_config_lens(configs)

    # TODO: replace with configs!
    run_in_parallel = True 
    num_workers = 3  # Number of worker processes

    if device == "cuda":
        device = "cuda:0"

    if run_in_parallel:
        trial_count = 10
        configs = {}
        print(f"Running segmentation for {trial_count} trial(s) in parallel")
        parallel_segmentation(configs, trial_count, device, num_workers)
    else:
        print(f"Running segmentation for {trial_count} trial(s)")
        serial_segmentation(configs, trial_count, device_input)

    
def run_video_processing(configs, device):
    """
    Generates output videos visualizing SAM2 segmentation results for one or more trials.

    This function reads a YAML configuration file and extracts trial-specific parameters 
    to generate annotated output videos using `write_output_video()`. Each trial uses 
    previously computed masks from `SAM2FishSegmenter` and overlays them on input 
    frames to produce a visual result.

    Parameters
    ----------
    configs : str
        Path to the YAML configuration file containing video generation settings. 
        Each parameter must either be a single value (applied to all trials) or a list 
        of values with one entry per trial.
    device : torch.device 
            A `torch.device` class specifying the device to use to draw the masks.

    Returns
    -------
    None
        The function does not return any values. It generates and saves a video file 
        (as specified in `video_file` in each trial configuration) for each trial.

    Notes
    -----
    - This function assumes that the segmentation masks (stored in `masks_dict_file`) 
      have already been generated for each trial.
    - Trial count is inferred from the number of values provided for list-type parameters.
      All list-type parameters must have the same length.
    - The function relies on `read_config_yaml`, `extract_config_lens`, and 
      `get_trial_config` in utils to handle configuration management.
    - `write_output_video()` is responsible for the actual rendering and saving of the video.
    
    Warnings
    --------
    If the configuration specifies only a single `video_file` name while running multiple 
    trials, the output videos will be written to the same file, and results will be 
    overwritten. To prevent this, provide a list of unique `video_file` names — one for each trial.
    
    Examples
    --------
    >>> run_video_processing("template_configs.yaml", device=torch.device("cuda"))
    Creating video: ./output_trial1.mp4 from ./frames1 and ./generated_frame_masks1.pkl
    Creating video: ./output_trial2.mp4 from ./frames2 and ./generated_frame_masks2.pkl
    """
    # Load the YAML configuration file
    configs = utils.read_config_yaml(configs)
    
    # Retrieve trial count from the length of values provided for each configuration key
    trial_count = utils.extract_config_lens(configs)
    print(f"Creating masked video(s) for {trial_count} trial(s)")

     # Iterate over each trial and extract configuration values
    for i in range(trial_count): 
        trial_config = utils.get_trial_config(configs,i)
        
        # Write the output with modified trial configs
        print(f"Creating video: {trial_config['video_file']} from {trial_config['frame_dir']} and {trial_config['masks_dict_file']}")

        utils.write_output_video(
            frame_dir = trial_config["frame_dir"],
            frame_masks_file = trial_config["masks_dict_file"],
            video_file=trial_config["video_file"],
            out_fps=trial_config["out_fps"],
            video_frame_size=trial_config["video_frame_size"],
            fps=trial_config["fps"],
            SAM2_start=trial_config["SAM2_start"],
            font_size=trial_config["font_size"],
            font_color=trial_config["font_color"],
            alpha=trial_config["alpha"],
            device=device
            )