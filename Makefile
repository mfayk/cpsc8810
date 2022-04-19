NVCC=nvcc 

OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/software/spackages/linux-centos8-x86_64/gcc-8.3.1/cuda-10.2.89-bsydnscqeeeoytpr2buzenenj2uex3ux/lib64/


GCC_OPTS=-std=c++17 -g -O3 -Wall 
CUDA_LD_FLAGS=-L $(CUDA_INCLUDEPATH) -lcuda -lcudart



final: main.o blur.o
	g++ -o kmeans main.o blur_kernels.o -I $(CUDA_INCLUDEPATH) $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS)

main.o:main.cpp gaussian_kernel.h utils.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDE_PATH) main.cpp


blur.o: blur_kernels.cu gaussian_kernel.h  utils.h
	$(NVCC) -c blur_kernels.cu 

clean:
	rm *.o kmeans
