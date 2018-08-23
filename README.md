# VESPCN-TensorFlow (Recently I have restarted updating the codes, and will be done in a few days. For one who want the old version codes, please browe the branch before my recent commits)

TensorFlow implementation of ESPCN [1]/VESPCN [2] (ongoing)

## **How to run the code**
1. ESPCN-main.ipynb (will update main.py soon)
2. You can see the result in ./result/$(model_name)/ directory.PSNR.pdf & Loss.pdf shows the PSNR/Loss pattern. 
   Output images are saved in images directory. (current version has some bugs in image saving functionality, but I will fix it soon)

## **TODO list**
- [] Update ESPCN
- [] Update Motion compensator
- [] Update VESPCN

[1] W. Shi et al, “Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network,” IEEE CVPR 2016.

[2] J. Caballero et al, “Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation,” IEEE CVPR 2017.
