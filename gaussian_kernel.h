#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"
#include <vector>




void your_kmeans(int rows, int cols, int numclasses, int *predictions, std::vector<float> *centroids, uchar4 *h_img, int max_its, int block);
void your_kmeans_shared(int rows, int cols, int numclasses, int *predictions, std::vector<float> *centroids, uchar4 *h_img, int max_its, int block);
void your_kmeans_contrast(int rows, int cols, int numclasses, int *predictions, std::vector<float> *centroids, uchar4 *h_img, int max_its, float contrast);
void your_kmeans_contrast_shared(int rows, int cols, int numclasses, int *predictions, std::vector<float> *centroids, uchar4 *h_img, int max_its, float contrast);
