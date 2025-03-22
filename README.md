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
mamba activate annotate_env
python3 LocalAnnotationBitesGUI_0226.py
```
This will pull up the GUI window. Use the __Browse Video__ button to select the video you wish to annotate. Depending on the file size, the video may take a few minutes to load during which Python will show `Application not responding` - this is normal, do not exit the GUI. 

Once the video loads, you can use the player control buttons to pause and play the video, adjust the playback rate, and move frame-by frame. There is a scroll bar beneath the video player that can be used to move to a different time. The arrow keys can also be used to quickly advance or move backward frames. 
The current time, current frame, and playback speed are shown at the top of the right panel. 

### SAM2 Start Frame

The SAM2 Start Frame is a function for ensuring that the annotated frames correspond with the frames extracted for SAM2. The SAM2 Start Frame specifies which frame to begin counting at, then will display a message "SAM2 Frame: Annotate Fish Position" on the 3 frames per second that will be processed by SAM2.
> [!Note] 
> It is assumed (and hard-coded within this GUI function) that frames will be extracted from the raw video at 3 FPS. 
> If a different temporal resolution is desired, line 58 of `LocalAnnotationBitesGUI_0226.py` can be edited to adjust the `special_frame_interval`. 
As a default, the SAM2 Start Frame will be 0, and can remain as 0 for videos where left-right video syncing has already been completed or is not necessary. 

### Click Types

Positive click types should be used to mark where the object is located. Typically, only one positive click is necessary to segment a well-contrasted object.

Negative click types can be used to mark where the object is __not__ located. Typically, these should only be used when correcting SAM2 mask outputs, because they can confuse SAM2 and impair performance if used too much. 

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

Annotations can be deleted using the Delete Selected or Delete All buttons. The Delete All button will prompt for confirmation before deleting all annotations.

### Entries and Exits

On the first SAM2 frame where the individual is visible, an Entry should be marked using the Add Entry button or the keystroke 'e'. On the last SAM2 frame where the individual is visible, an Exit should be marked using the Add Exit button or the keystroke 'x'. No location is associated with entries and exits, but the correct fish family and individual name should be specified before marking entries and exits. 
Multiple entries and exits can be associated for each object; these will essentially tell SAM2 when to start and stop tracking the object. If the object leaves the field of view temporarily, it is best to mark it as an exit and re-entry; otherwise, SAM2 will keep looking for that object and may incorrectly mask a different but similar-looking object.

If your cursor is in one of the input text boxes, pressing "e" or "x" will simply type in the text box and not create a new entry or exit. Click on the annotation table to remove your cursor from the input text box and use the hotkey shortcuts.

### Saving Annotations

To save your annotations, write the file name you wish to save as in the provided input textbox. If this file name has already been used, it will overwrite any existing files. 
By default, saving your annotations will create two files that will be saved in the working directory where you launched the GUI from. One of these will be `filename_bites.csv`, and this will contain your bites annotations. Another file named `filename_annotations.npy` will contain your positive and negative positional annotations, entries, and exits.  The `filename_annotations.npy` file is what you will submit to SAM2 to track your individuals.
Use the checkboxes at the top of the GUI to specify which files you'd like to save before clicking the Save Annotations button.

To edit previous annotations or continue previous progress, use the Import Previous Annotations button to re-load your bites and locations annotations into the GUI.

When all individuals to be tracked in a video have been marked with an entry, an exit, at least one positive click, you can save your annotations, exit the GUI, and proceed to SAM2 frame extraction for processing.

## Extract Frames for SAM2

The frames from a video will need to be extracted and stored within a folder before SAM2 processing:

```
# Set the frame to begin extracting.
# This must match the SAM2 Start Frame used in the GUI
SAM2_start_frame=0 

# Specify the original fps of the video.
video_fps=24

# Specify the path to the video that will be processed
video_name= "path/to/GX137102.MP4"

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
`environment.yaml`: 
```
mamba env create -f SAM2_environment.yaml
```
> [!NOTE]  
> The provided `environment.yaml` was constructed for NVIDIA GPUs compatible with 
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


When object positional and entry/exit annotations have been collected and saved in a filename_annotations.npy file, and the video frames have been extracted and stored in a frames folder, the main SAM2 workflow can be followed to process the folder of frames and predict masks for every individual annotated. 
The main SAM2 workflow must be run on a machine with GPU access.

A folder should be set up containing the `annotations.npy` file, the `frames` subfolder, and the necessary scripts for the SAM2 workflow: `main.py`, `sam2_fish_segmenter.py`, `template_configs.yaml`, `utils.py`, and `plot_utils.py`. 

The `template_configs.yaml` file should be edited to specify the paths to the SAM2 installation and provided checkpoints, the FPS of the original video that was annotated in the GUI, the `SAM2_start` frame that was used in both the GUI and the Extract Frames step, and the name of the annotations NumPy file. 

Lines 32 - 42 specify the key used in the annotations file created by the GUI. The most recent GUI uses different labels than previous versions and these may need to be altered:
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

If a different extracted frame rate has been used (i.e., other than 3 FPS), change the `video_fps` value on line 83. 

Other values in the `template_configs.yaml` can be left as the default, or can be changed as desired. 

Once these changes are saved, activate the SAM2 environment and run the main workflow:
```
mamba activate sam2-env
python3 main.py
```

If default values are used, when the code is done running, it should produce a `test_video.mp4` displaying all the predicted masks; this video can be viewed to validate SAM2 predictions. The actual masks are saved as a dictionary of sparse tensors within `generated_frame_masks.pkl`.

Please raise an issue or contact M.Hair if you experience issues using this code. 