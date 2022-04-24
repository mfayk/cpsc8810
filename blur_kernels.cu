#include <iostream>
#include "./gaussian_kernel.h" 

using namespace std;


#define BLOCK 4
#define TILE_WIDTH 40

//test

__global__ 
void im2Gray(uchar4 *d_in, float *d_grey, int numRows, int numCols, float *gray_sums){

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (y < numRows && x < numCols){
  	int grayOffset = y * numCols + x;
  	unsigned char r = d_in[grayOffset].x;
  	unsigned char g = d_in[grayOffset].y;
  	unsigned char b = d_in[grayOffset].z;
  	
  	d_grey[grayOffset] = 0.299*r + 0.587*g + 0.114*b;

    (float)atomicAdd(gray_sums,d_grey[grayOffset]);
  }

}


__global__ 
void centroid_update_shared(int rows, int cols, int numclasses,  float *centroids, int *new_predictions, uchar4 *d_img, int *predictions_d){
  
  int chosenclass;
  int pixelClass;
  int oldClass;

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

  // thread block id
  int local_col = threadIdx.x;
  int local_row = threadIdx.y;

  //get idx of thread at the block level
	const int s_idx = threadIdx.x;

  __shared__ float share_cen[4* 4];




  if(local_row == 0 && local_col == 0){
    //share_img[r*cols+c] = d_img[r*cols+c];

    for(int i=0; i<numclasses; i++){
      for(int j=0; j<4; j++){
          share_cen[j*numclasses+i] = centroids[j * numclasses + i];
      }
  }
  }

  __syncthreads();

  if(c < cols && r < rows){


    

    pixelClass = new_predictions[r*cols+c];
    oldClass = predictions_d[r*cols+c];


    if(pixelClass != oldClass){

      (float)atomicAdd(&share_cen[0 * numclasses + pixelClass],(float)d_img[r*cols+c].x);
      (float)atomicAdd(&share_cen[1 * numclasses + pixelClass],(float)d_img[r*cols+c].y);
      (float)atomicAdd(&share_cen[2 * numclasses + pixelClass],(float)d_img[r*cols+c].z);
      (float)atomicAdd(&share_cen[3 * numclasses + pixelClass],1.0);

      (float)atomicAdd(&share_cen[0 * numclasses + oldClass],-(float)d_img[r*cols+c].x);
      (float)atomicAdd(&share_cen[1 * numclasses + oldClass],-(float)d_img[r*cols+c].y);
      (float)atomicAdd(&share_cen[2 * numclasses + oldClass],-(float)d_img[r*cols+c].z);
      (float)atomicAdd(&share_cen[3 * numclasses + oldClass],-1.0);

    }
    __syncthreads();
    for(int i=0; i<numclasses; i++){
      for(int j=0; j<4; j++){
        centroids[j * numclasses + i] = share_cen[j*numclasses+i];
      }
  }


  }
} 


__global__ 
void homogeneous_check(int rows, int cols, int *new_predictions, int *predictions, bool *homogeneous){
  
  int chosenclass;


	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(c < cols && r < rows){
    //assume homogenous, if an element is not reset to false
    *homogeneous = true;

      if(predictions[r*cols+c] != new_predictions[r*cols+c]){
        *homogeneous = false;
      }

    predictions[r*cols+c] = new_predictions[r*cols+c];

  }

} 


__global__ 
void centroid_update(int rows, int cols, int numclasses,  float *centroids, int *new_predictions, uchar4 *d_img, int *predictions_d){
  
  int chosenclass;
  int pixelClass;
  int oldClass;

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(c < cols && r < rows){

    pixelClass = new_predictions[r*cols+c];
    oldClass = predictions_d[r*cols+c];


    if(pixelClass != oldClass){
      /*
      centroids[0 * numclasses + pixelClass] += (float)d_img[r*cols+c].x; //add the blue
      centroids[1 * numclasses + pixelClass] += (float)d_img[r*cols+c].y; //add the green
      centroids[2 * numclasses + pixelClass] += (float)d_img[r*cols+c].z; //add the red
      centroids[3 * numclasses + pixelClass] += 1; //add 1 to the population
      */
//      __syncthreads();
      (float)atomicAdd(&centroids[0 * numclasses + pixelClass],(float)d_img[r*cols+c].x);
      (float)atomicAdd(&centroids[1 * numclasses + pixelClass],(float)d_img[r*cols+c].y);
      (float)atomicAdd(&centroids[2 * numclasses + pixelClass],(float)d_img[r*cols+c].z);
      (float)atomicAdd(&centroids[3 * numclasses + pixelClass],1);

      /*
      centroids[0 * numclasses + oldClass] -= (float)d_img[r*cols+c].x; //add the blue
      centroids[1 * numclasses + oldClass] -= (float)d_img[r*cols+c].y; //add the green
      centroids[2 * numclasses + oldClass] -= (float)d_img[r*cols+c].z; //add the red
      centroids[3 * numclasses + oldClass] -= 1; //add 1 to the population
      */
      (float)atomicAdd(&centroids[0 * numclasses + oldClass],-(float)d_img[r*cols+c].x);
      (float)atomicAdd(&centroids[1 * numclasses + oldClass],-(float)d_img[r*cols+c].y);
      (float)atomicAdd(&centroids[2 * numclasses + oldClass],-(float)d_img[r*cols+c].z);
      (float)atomicAdd(&centroids[3 * numclasses + oldClass],-1);


    }
  }
} 

__global__ 
void k_operations(int rows, int cols, int numclasses, float *centroids,  int *new_predictions, uchar4 *d_img, int *predictions_d){

  int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
  
  float min_dist = INFINITY;
  int close = 0;
  float dist;
  int chosenclass;
  int pixelClass;
  int oldClass;
  
  unsigned char b_avg;
  unsigned char g_avg;
  unsigned char r_avg;

  if(c < cols && r < rows){


    for(int k = 0; k < numclasses; k++)
    {   
        r_avg = centroids[0 * numclasses + k]/centroids[3 * numclasses + k];
        g_avg = centroids[1 * numclasses + k]/centroids[3 * numclasses + k];
        b_avg = centroids[2 * numclasses + k]/centroids[3 * numclasses + k];
        
        dist = sqrt(pow((r_avg - (float)d_img[r*cols+c].x),2) + pow((g_avg - (float)d_img[r*cols+c].y),2) + pow((b_avg - (float)d_img[r*cols+c].z),2)* 1.0);
        if(dist < min_dist){
            min_dist = dist;
            close = k;
        }   
    }
    __syncthreads();
    new_predictions[r*cols+c] = close;

  }
}  

__global__
void starting_predictions(int rows, int cols, int *predictions, uchar4 *img){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index;

  if(x < cols && y < rows)
	{
    index = y * cols + x;
    unsigned char r = img[index].x; 
		unsigned char g = img[index].y;
		unsigned char b = img[index].z;
		
		int grey_pixel = 0.224f*r + 0.587f*g + 0.111*b;

		if(grey_pixel > 155)
      predictions[index] = 1;
    else
      predictions[index] = 0;
	}
}


void your_kmeans(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its){

  vector<float> distances;
  int new_predictions[rows*cols];
  bool *homogeneous_h;
  int iteration = 0;
  int *predictions_d;
  int *predictions_h;
  int *predictions_h_new;
  float *centroids_d;
  bool *homogeneous_d;
  int *new_predictions_d;
  float *distances_d;
  int count =0;
  uchar4 *d_img;

  int iterations = max_its;

  dim3 blockSize(BLOCK,BLOCK,1);
  dim3 gridSize(ceil(cols/BLOCK)+1,ceil(rows/BLOCK)+1,1);


  float *centroids_arr;
  
  predictions_h = (int *)malloc(rows*cols*sizeof(int));
  predictions_h_new = (int *)malloc(rows*cols*sizeof(int));
  centroids_arr = (float*)malloc(4*numclasses*sizeof(float));
  homogeneous_h = (bool *)malloc(sizeof(bool));
  *homogeneous_h = false;

  for (int i = 0; i < numclasses; i++) {  
    cout << "Elements at index "
         << i << ": ";

    int j=0;
    for (auto it = centroids[i].begin();
         it != centroids[i].end(); it++) {

        cout << *it << ' ';
        centroids_arr[j * numclasses + i] = *it;
        j++;
    }
    cout << endl;
}

  cout << "classes " << numclasses << endl;

  for(int i=0; i<numclasses; i++){
    for(int j=0; j<4; j++){
        cout << centroids_arr[j * numclasses + i] << " ";
    }
       cout << endl;
}


  checkCudaErrors(cudaMalloc((void**)&predictions_d, sizeof(int)*rows*cols));
  checkCudaErrors(cudaMalloc((void**)&centroids_d, sizeof(float)*numclasses*4));
  checkCudaErrors(cudaMalloc((void**)&new_predictions_d, sizeof(int)*rows*cols));
  checkCudaErrors(cudaMalloc((void**)&homogeneous_d, sizeof(bool)));
  checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(uchar4)*rows*cols));

  checkCudaErrors(cudaMemcpy(d_img, h_img, sizeof(uchar4)*rows*cols, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(centroids_d, centroids_arr, sizeof(float)*4*numclasses, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpy(predictions_d, predictions, sizeof(int)*rows*cols, cudaMemcpyHostToDevice));


  starting_predictions<<<gridSize, blockSize>>>(rows, cols, predictions_d, d_img);
  cout << "loop start gpu" << endl;

    for(count=0; count<iterations; count++)
    {
        std::cout << "starting iteration " << iteration << "\n";

        k_operations<<<gridSize, blockSize>>>(rows,cols,numclasses,centroids_d,new_predictions_d,d_img,predictions_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        centroid_update<<<gridSize, blockSize>>>(rows,cols,numclasses,centroids_d,new_predictions_d,d_img,predictions_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        homogeneous_check<<<gridSize, blockSize>>>(rows,cols,new_predictions_d,predictions_d,homogeneous_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(homogeneous_h, homogeneous_d, sizeof(bool), cudaMemcpyDeviceToHost));
        if(*homogeneous_h == true && iteration != 0){
          cout << "converged\n";
          break;
        }
        iteration++;
    }

    
checkCudaErrors(cudaMemcpy(predictions, new_predictions_d, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost));

cudaFree(predictions_d);
cudaFree(centroids_d);
cudaFree(new_predictions_d);
cudaFree(homogeneous_d);
cudaFree(d_img);
free(predictions_h);
free(predictions_h_new);
free(centroids_arr);
free(homogeneous_h);


}

 void your_kmeans_shared(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its){

    vector<float> distances;
  int new_predictions[rows*cols];
  bool *homogeneous_h;
  int iteration = 0;
  int *predictions_d;
  int *predictions_h;
  int *predictions_h_new;
  float *centroids_d;
  bool *homogeneous_d;
  int *new_predictions_d;
  float *distances_d;
  int count =0;
  uchar4 *d_img;

  int iterations = max_its;

  dim3 blockSize(BLOCK,BLOCK,1);
  dim3 gridSize(ceil(cols/BLOCK)+1,ceil(rows/BLOCK)+1,1);


  float *centroids_arr;
  
  predictions_h = (int *)malloc(rows*cols*sizeof(int));
  predictions_h_new = (int *)malloc(rows*cols*sizeof(int));
  centroids_arr = (float*)malloc(4*numclasses*sizeof(float));
  homogeneous_h = (bool *)malloc(sizeof(bool));
  *homogeneous_h = false;

  for (int i = 0; i < numclasses; i++) {  
    cout << "Elements at index "
         << i << ": ";

    int j=0;
    for (auto it = centroids[i].begin();
         it != centroids[i].end(); it++) {

        cout << *it << ' ';
        centroids_arr[j * numclasses + i] = *it;
        j++;
    }
    cout << endl;
}

  cout << "classes " << numclasses << endl;

  for(int i=0; i<numclasses; i++){
    for(int j=0; j<4; j++){
        cout << centroids_arr[j * numclasses + i] << " ";
    }
       cout << endl;
}


  checkCudaErrors(cudaMalloc((void**)&predictions_d, sizeof(int)*rows*cols));
  checkCudaErrors(cudaMalloc((void**)&centroids_d, sizeof(float)*numclasses*4));
  checkCudaErrors(cudaMalloc((void**)&new_predictions_d, sizeof(int)*rows*cols));
  checkCudaErrors(cudaMalloc((void**)&homogeneous_d, sizeof(bool)));
  checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(uchar4)*rows*cols));

  checkCudaErrors(cudaMemcpy(d_img, h_img, sizeof(uchar4)*rows*cols, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(centroids_d, centroids_arr, sizeof(float)*4*numclasses, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(predictions_d, predictions, sizeof(int)*rows*cols, cudaMemcpyHostToDevice));

  cout << "loop start gpu" << endl;

    for(count=0; count<iterations; count++)
    {
        std::cout << "starting iteration " << iteration << "\n";

        k_operations<<<gridSize, blockSize>>>(rows,cols,numclasses,centroids_d,new_predictions_d,d_img,predictions_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        centroid_update_shared<<<gridSize, blockSize>>>(rows,cols,numclasses,centroids_d,new_predictions_d,d_img,predictions_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        homogeneous_check<<<gridSize, blockSize>>>(rows,cols,new_predictions_d,predictions_d,homogeneous_d);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(homogeneous_h, homogeneous_d, sizeof(bool), cudaMemcpyDeviceToHost));
        if(*homogeneous_h == true && iteration != 0){
          cout << "converged\n";
          break;
        }
        iteration++;
    }

    
checkCudaErrors(cudaMemcpy(predictions, new_predictions_d, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost));

cudaFree(predictions_d);
cudaFree(centroids_d);
cudaFree(new_predictions_d);
cudaFree(homogeneous_d);
cudaFree(d_img);
free(predictions_h);
free(predictions_h_new);
free(centroids_arr);
free(homogeneous_h);

 }
