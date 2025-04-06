#!/bin/bash
#SBATCH --partition amilan # Partition or queue
#SBATCH --job-name=Extract_Frames # Job name
#SBATCH --nodes=1
#SBATCH --time=1:00:00 # Time limit hrs:min:sec
#SBATCH --output=log_%j.out # Standard output and error log
#SBATCH --error=log_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maha7624@colorado.edu #Change this to your own CU email address

# Load ffmpeg
module purge
module load ffmpeg

# Specify path to video
video_file="path/to/input/video.mp4"

# Assign SAM2 start frame.  to match the metadata for your video. 
start_frame=0

# Assign this is as the rounded frame rate (frames per second) of your video.
# Can be rounded already or exact
video_fps=23.997

# Convert to rounded integer if not already
video_fps=$(printf "%.0f" "$video_fps")

# Frame rate to be extracted. Determines the temporal resolution of the data. 
# Must align with GUI out_fps and SAM2 out_fps
out_fps=3 

# Calculate frame interval
frame_interval=$((video_fps/out_fps))
echo ${frame_interval}

# Extract exact frames to be processed by SAM2
ffmpeg -i "$video_file" -vf "select='not(mod(n-${start_frame},${frame_interval}))',setpts=N/FRAME_RATE/TB" -vsync vfr frames/'%05d.jpg'



