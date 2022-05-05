# cpsc8810
CPSC 8810 - Spring 2022 
Max Faykus and Mikaila Gossman

Improving K-Means Image Segmentation


The folder contains all of the source code used for our final project
of CPSC 8810. Our project looked at improving K-Means execution time
in preparation for real-time image segmentation. To do this we provide
a parallelized version of the K-Means algorithm written in CUDA-C. 
We provide various optimizations such as informed starting predicitons,
utilizing shared memory, and finding regions of interest.

The CUDA kernels and all of their helper functions are located in 
blur_kernels.cu. It contains both the naive global implmentation as
well as the optimized shared memory kernel. Here you will also find
where the optimized starting predictions are made.

Our test script can be seen in main.cpp. For our experiments we load 10
images from the Rellis-3D dataset and perform image segmentation on them
using our custom CUDA kernels. This file also contains our code for 
findingregions of interest in the images such that they are cropped 
before copying data over to the GPU.

To run the code on the Palmetto cluster first

module load opencv/4.2.0-gcc

module load cuda/11.1.0-gcc/8.4.1

module load openmpi/3.1.5-gcc/8.3.1-cuda11_0-ucx

module load gcc/9.3.0

then cd into the git folder

run make ./kmenas thread_count

example: ./kmeans 16

