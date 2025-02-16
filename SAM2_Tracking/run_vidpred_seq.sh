#!/bin/bash
#SBATCH --partition aa100 # Partition or queue
#SBATCH --job-name=SAM2_GH010367 # Job name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=21 #GPU's * 3 is max tasks
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00 # Time limit hrs:min:sec
#SBATCH --account=ucb523_asc1
#SBATCH --output=log_%j.out # Standard output and error log
#SBATCH --error=log_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maha7624@colorado.edu #Change this to your email address

#load conda environment
module purge
module load mambaforge
conda activate sam2.1_pytorch241_cuda124
echo "SAM2 conda environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize the video predictor model
python3 initialize_predictor.py
echo "Predictor model initialized. Moving on to predicting masks"

# Read the annotations.npy file and predict masks
python3 predict_masks.py
echo "Masks predicted from annotations. Moving on to propagation" 

# Propagate masks across the entire video
python3 propagate_masks.py
echo "Masks propagated across entire video. Moving on to video writing"

python3 write_outputvid.py
echo "Output video created demonstrating masks"

