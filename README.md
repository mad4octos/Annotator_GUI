# Annotator_GUI

# To Do: Add full Description here

For now, please refer to `SAM2_Tracking_GUI-UserManual.pdf`

## Installation 

Install all necessary dependencies for code (not including SAM2):
```
mamba env create -f environment.yaml
```

Installation of SAM2: 
```
mamba activate sam2-env
cd /path/where/sam2/will/be/installed
git clone https://github.com/facebookresearch/sam2.git 
cd sam2/
SAM2_BUILD_ALLOW_ERRORS=0 python setup.py develop
cd checkpoints/
./download_ckpts.sh
```