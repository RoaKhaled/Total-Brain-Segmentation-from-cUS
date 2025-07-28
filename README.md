# Total-Brain-Segmentation-from-cUS
This repository contains codes and models for neonatal Total Brain segmentation from cUS.
We trined 2D UNet models using input slices across the three anatomical planes, in addition to a Contextual UNet model, where 3 adjacent slices were used as input.
We also trained nnUNet, due to their large sizes, the nn UNet models and sample cUS data are shared here: https://huggingface.co/RoaaKhaled/Total-Brain-Segmentation-from-cUS

The structure of this repo is as follows:

Total-Brain-Segmentation-from-cUS/
├── 2D and Contextual UNet
          ├── Codes
          ├── Trained Models
          ├── UNet2D_env_requirements.txt
├── nnUNet
          ├── Codes
          ├── nnUNet_env_requirements.txt

├── README.md

To do inference with low-resource CPU using any of the trained 2D UNet models (axial, sagittal, coronal) or Contextual UNet, use the script: local_cpu_inference_2dUNet.py after setting all directories in UNet2D_pyImageSearch/local_inference_config.py
To train your own model using our architecture and configuration, use the script: train_2d_or_Contextual_UNet.py after setting all directories in UNet2D_pyImageSearch/training_config.py

To do inference with low-resource CPU using our trained nnUNet (2d or 3d, or an ensemble), use the script: local_cpu_inference_nnUNet.py after downloading the models from (https://huggingface.co/RoaaKhaled/Total-Brain-Segmentation-from-cUS).
To train your own nnUNet refer to the owner's repo here: https://github.com/MIC-DKFZ/nnUNet.git
