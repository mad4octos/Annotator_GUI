# Installation 

To run the SAM2 segmentation portions of this workflow, several
dependencies need to be installed. In this section, we provide 
instructions for installing these dependencies using a 
[Mamba](https://mamba.readthedocs.io/en/latest/) environment. 
To install all necessary dependencies for the segmentation workflow 
(not including SAM2), create a Mamba environment using the provided 
`environment.yaml`: 
```
mamba env create -f environment.yaml
```

```{note}
The provided `environment.yaml` was constructed for NVIDIA GPUs compatible with 
CUDA 12.6. The dependencies may need to be altered for other GPUs. 
```

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

