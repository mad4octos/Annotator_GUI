# Annotator_GUI / SAM2 Interactive Tracker / EcoTrack

These scripts can be used to rapidly mark the positions of multiple objects from a video in an interactive Graphical User Interface (GUI), then run the Segment Anything Model 2 (SAM2) for automated tracking. Our SAM2 workflow is adapted from the [Meta Github repository](https://github.com/facebookresearch/sam2), and modified to drastically improve memory efficiency and iterate over multiple tracked individuals, thus avoiding the loss of performance that results from SAM2 tracking multiple objects simultaneously.

> [!NOTE]
> For the purposes of 3D recreation, left and right paired videos should be synced before beginning to collect annotations. Our code to sync videos is not yet publicly available within this GitHub repository, but will be soon. Contact M. Hair if desired. 

CU students and members of the Gil Lab, please refer to `SAM2_Tracking_GUI-UserManual.pdf` for specific instructions using CU's HPC: Alpine

## Annotator GUI Installation

To run the Annotator GUI portion of this workflow and collect positional annotations, GPU resources are not needed and it can be run on a local machine. However, several dependencies need to be installed. In this section, we provide instructions for installing these dependencies using a 
[Mamba](https://mamba.readthedocs.io/en/latest/) environment. 

To install all necessary dependencies for the Annotator GUI, create a Mamba environment using the provided `GUI_environment.yaml`:
```
mamba env create -f GUI_environment.yaml
```
> [!NOTE]  
> The provided `GUI_environment.yaml` is functional on MacOS (14.4.1). 
> If modifications are necessary for Windows machine, please raise an issue.

## Using the Annotator GUI
To launch the GUI, you should first activate the GUI environment with the necessary dependencies: 
```
mamba activate annotate-env
```

To launch the GUI, run:
```
python3 clickpointGUI_0929.py
```



This will pull up the GUI window. Use the __Browse Video__ button to select the video you wish to annotate. The default frame cache size is 50 frames, meaning that only 50 frames will be loaded at a time. This can be adjusted in the script by changing `frame_cache_size`. A higher frame cache will result in smoother playback but higher memory demand. 

Once the video loads, you can use the player control buttons to pause and play the video, adjust the playback rate, and move frame-by frame. There is a scroll bar beneath the video player that can be used to move to a different time. The current time, current frame, and playback speed are shown below the video player.

### Hot Keys: 

The arrow keys can also be used to quickly advance or move backward to the nearest SAM2 frame. The "." and "," keys can be used to move forward and backward a single frame at a time. The spacebar can be used to pause-play the video. The mouse wheel can be used to zoom in and out in the video. The "a", "s", "w", "d" keys can be used to pan left, right, up, and down in the zoomed video player window. The Enter button can be used to add an annotation. 

> [!NOTE] 
> The hotkeys cannot be used if the cursor is within any of the textboxes. Click on the annotation table to enable use of the hotkeys, rather than typing in the textbox. 

 
### SAM2 Start Frame

The SAM2 Start Frame is a function for ensuring that the annotated frames correspond with the frames extracted for SAM2. The SAM2 Start Frame specifies which frame to begin counting at, then will display a message "SAM2 Frame: Annotate Fish Position" on the 3 frames per second that will be processed by SAM2.
> [!Note] 
> It is the default assumption that frames will be extracted from the raw video at 3 FPS. 
> If a different temporal resolution is desired, line 25 of `clickpointGUI_0929.py` can be edited to change `3`to your desired extraction frame rate. 
As a default, the SAM2 Start Frame will be 0, and can remain as 0 for videos where left-right video syncing has already been completed or is not necessary. 

### Click Types

Positive click types should be used to mark where the object is located. Typically, only one positive click is necessary to segment a well-contrasted object.

Negative click types can be used to mark where the object is __not__ located. Typically, these should only be used when correcting SAM2 mask outputs, because they can confuse SAM2 and impair performance if used too much. 
> [!NOTE]
> WE have not found negative clickpoints to be useful in improving SAM2 performance.

Bite clicks can be used to mark the location of bites, or any other discrete behavioral event. 

### Fish Family

This button can be toggled to specify which fish family, or object category, you are marking annotations for. The current options are Parrotfish, Surgeonfish, Damselfish, and Other.

### Fish Name

This option specifies which individual you are marking annotations for. This should be a unique integer for every individual. 

### Adding Annotations

When the SAM2 start frame, click type, fish family, and fish name are set to the correct values, use your cursor to click on the video where you wish to mark a click. Then press enter or the "Add Annotation" button to record the annotation in the annotation table in the right panel. The "Annotation Table" will display the frame that was annotated, the "Click Type" used (0 is a negative click, 1 is a positive click, 2 is a bite click), the fish name, family and the x,y coordinates of the click location.

> [!Note] 
> Only frames that will be extracted for SAM2 processing should be annotated with positive / negative clicks or entries/exits. These frames will display a message "SAM2 Frame: Annotate Fish Position". 
> Bite clicks can be annotated on any frame. 

Annotations can be deleted using the "Delete Selected" or "Delete All" buttons. The "Delete All" button will prompt for confirmation before deleting all annotations.

### Entries and Exits

On the first SAM2 frame where the individual is visible, an entry should be marked using the "Add Entry" button or the keystroke "e". On the last SAM2 frame where the individual is visible, an exit should be marked using the "Add Exit" button or the keystroke "x". No location is associated with entries and exits, but the correct fish family and individual name should be specified before marking entries and exits. 
Multiple entries and exits can be associated for each object; these will essentially tell SAM2 when to start and stop tracking the object. If the object leaves the field of view temporarily, it is best to mark it as an exit and re-entry; otherwise, SAM2 will keep looking for that object and may incorrectly mask a different but similar-looking object.

If your cursor is in one of the input text boxes, pressing "e" or "x" will simply type in the text box and not create a new entry or exit. Click on the annotation table to remove your cursor from the input text box and use the hotkey shortcuts.

### Saving Annotations

To save your annotations, write the file name you wish to save it as in the provided input textbox. If this file name has already been used, it will overwrite any existing files. 

By default, saving your annotations will create two files that will be saved in the working directory where you launched the GUI from. One of these will be `filename_bites.csv`, and this will contain your bites annotations. Another file named `filename_annotations.npy` will contain your positive and negative positional annotations, entries, and exits.  The `filename_annotations.npy` file is what you will submit to SAM2 to track your individuals.
Use the checkboxes at the top of the GUI to specify which files you'd like to save before clicking the "Save Annotations" button.

To edit previous annotations or continue previous progress, use the "Import Previous Annotations" button to re-load your bites and locations annotations into the GUI.

When all individuals to be tracked in a video have been marked with an entry, an exit, at least one positive click, you can save your annotations, exit the GUI, and proceed to SAM2 frame extraction for processing.

## Extract Frames for SAM2

The frames from a video will need to be extracted and stored within a folder before SAM2 processing. This can be done from the command line using FFmpeg:
```
# Set the frame to begin extracting.
# This must match the SAM2 Start Frame used in the GUI
SAM2_start_frame=0 

# Specify the original fps of the video.
video_fps=23.997

# Specify the path to the video that will be processed
video_name="path/to/GX137102.MP4"

# Set the frame rate to be extracted. Determines the temporal resolution of the data. Must match the frequency used in the GUI (default of 3fps)
fps=3

# Calculate the frame interval to be extracted
frame_interval=$((video_fps/fps))

# Extract frames into a folder called "frames"
ffmpeg -i video_name -vf "select='not(mod(n-${start_frame},${frame_interval}))',setpts=N/FRAME_RATE/TB" -vsync vfr frames/'%05d.jpg'
```
When this code is done running, you should have a folder frames to process with your annotations.

## SAM2 Installation 

To run the SAM2 segmentation portions of this workflow, several
dependencies need to be installed. In this section, we provide 
instructions for installing these dependencies using a 
[Mamba](https://mamba.readthedocs.io/en/latest/) environment. 

To install all necessary dependencies for the segmentation workflow 
(not including SAM2), create a Mamba environment using the provided 
`SAM2_environment.yaml`: 
```
mamba env create -f SAM2_environment.yaml
```
> [!NOTE]  
> The provided `SAM2_environment.yaml` was constructed for NVIDIA GPUs compatible with 
> CUDA 12.6. The dependencies may need to be altered for other GPUs. 

Now that all core dependencies have been installed, we can proceed to 
the SAM2 installation: 
```
mamba activate sam2-env
cd /path/where/sam2/will/be/installed
git clone https://github.com/facebookresearch/sam2.git 
cd sam2/
SAM2_BUILD_ALLOW_ERRORS=0 python setup.py develop
cd checkpoints/
./download_ckpts.sh
```
SAM2 only needs to be installed once. 

## Running SAM2


When object positional and entry/exit annotations have been collected and saved in an annotations file e.g. `annotations.npy`, and the video frames have been extracted and stored in a frames folder, the main SAM2 workflow can be followed to process the folder of frames and predict masks for every individual annotated.
The main SAM2 workflow must be run on a machine with GPU access.

A folder should be set up containing the `annotations.npy` file, the `frames` subfolder, and the necessary scripts for the SAM2 workflow: `main.py`, `sam2_fish_segmenter.py`, `template_configs.yaml`, `utils.py`, and `plot_utils.py`. All these files can be obtained from the repo's `SAM2_Tracking` directory. In the future, we will make this a Python package, so that transferring files is not necessary. 

The `template_configs.yaml` file should be edited to specify the paths to the SAM2 installation and provided checkpoints, the FPS of the original video that was annotated in the GUI, the `SAM2_start` frame that was used in both the GUI and the Extract Frames step, and the paths to the the annotations NumPy file and folder of frames for each trial. 

Lines 41 - 51 specify the key used in the annotations file created by the GUI. The most recent GUI uses different labels than previous versions and these may need to be altered:
```
# Key in the annotation corresponding to SAM2 
frame_idx_name: 'Frame'

# Key in the annotation corresponding to SAM2 obj_id
obj_id_name: 'ObjID'

# Key in the annotation corresponding to SAM2 points
points_name: 'Location'

# Key in the annotation corresponding to SAM2 labels
labels_name: 'ClickType'
```

If a different extracted frame rate has been used (i.e., other than 3 FPS), change the `out_fps` value on line 30. 

Other values in the `template_configs.yaml` can be left as the default, or can be changed as desired. 

Once these changes are saved, activate the SAM2 environment and run the main workflow:
```
mamba activate sam2-env
python3 main.py
```

If default values are used, when the code is done running, it should produce a dictionary of sparse tensors within `generated_frame_masks.pkl`. To visualize the generated masks, create a `test_video.mp4` displaying all the predicted masks:
```
mamba activate sam2-env
python3 create_video.py
```
This output video can be viewed to validate SAM2 predictions.

## Running SAM2 on multiple trials
If a user desires to process multiple trials in a single batch, they can specify multiple values for each parameter within the `template_configs.yaml`. Each parameter can be specified with either a single value (which will be applied to all processed trials) or a list of *n* values, where *n* = number of trials. For example: 

```
########################################
# annotation specific configurations   #
########################################

# Directories containing JPGs corresponding to the frames of the video
frame_dir: 
    - "/path/to/frames/trial1"
    - "/path/to/frames/trial2"
    - "/path/to/frames/trial3"

# File specifying annotations for video frames 
annotations_file: 
    - "/path/to/test_annotations_trial1.npy" 
    - "/path/to/test_annotations_trial2.npy" 
    - "/path/to/test_annotations_trial3.npy" 

# The FPS of the unreduced video that the annotations were 
# initially created with
fps: 24

# Value that ensures the annotated frame value matches up 
# with the fames that will be ingested by SAM2
SAM2_start: 
    - 0 # Trial 1 SAM2 start
    - 2 # Trial 2 SAM2 start
    - 2 # Trial 3 SAM2 start

# Reduced frame rate. Must match with the extracted frame rate.
out_fps: 3

# The name and location to save the dictionary of masks.
masks_dict_file: 
    - './trial_1_generated_frame_masks.pkl'
    - './trial_2_generated_frame_masks.pkl'
    - './trial_3_generated_frame_masks.pkl'

# The name of the video file to be created
video_file: 
    - "./trial_1_test_video.mp4"
    - "./trial_2_test_video.mp4"
    - "./trial_3_test_video.mp4"
```
With the example configuration above, running both the `main.py` and `create_video.py` scripts will process 3 trials using their respective frames, annotations, and SAM2 start value, and will save a dictionary of masks and output video for each trial.

For every trial to be fully processed and visualized, a name for `frame_dir`, `annotation_file`, `masks_dict_file`, and `video_file` should be specified in the `template_configs.yaml`. Other values in the `template_configs.yaml` can be left as the default, or can be specified as desired. 

> [!Note] 
> If multiple values are not specified for the `masks_dict_file` and `video_file`, 
> the SAM2 outputs from multiple trials will overwrite each other. 

After adjusting the `template_configs.yaml` to specify all trials to be processed, use the `check_data.py` script to confirm that all your configurations are correct and compatible with SAM2 processing. 
```
cd path/to/working/directory

mamba activate sam2-env
python3 check_data.py
```

If there are no warnings, the SAM2 processing and video creation can be run as normal:
```
python3 main.py
python3 create_video.py
```

Please raise an issue or contact M. Hair (madelyn.hair@colorado.edu) if you experience issues using this code. 