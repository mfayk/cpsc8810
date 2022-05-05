# cpsc8810
group project for cpsc8810

blur_kernels.cu is the kmeans cuda kernel

main.cpp has all the setup and serial code


To run the code on the Palmetto cluster first 

module load opencv/4.2.0-gcc

module load cuda/11.1.0-gcc/8.4.1

module load openmpi/3.1.5-gcc/8.3.1-cuda11_0-ucx

module load gcc/9.3.0

then cd into the git folder

run make
./kmenas thread_count


example:
./kmeans 16
