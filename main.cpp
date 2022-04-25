#include <iostream>
#include <fstream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <cmath> 
#include <vector>
#include <chrono> 
#include <filesystem>
using namespace std::chrono;

#include "utils.h"
#include "gaussian_kernel.h"


using namespace std;
namespace fs=std::filesystem;

/* 
 * Compute if the two images are correctly 
 * computed. The reference image can 
 * either be produced by a software or by 
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems){

    std::cerr << "num Elements: " << numElems << "\n";
    for(int i = 0; i < numElems; i++){
        if(ref[i] - gpu[i] > 1){
            std::cerr << "Error at position " << i << "\n"; 

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}



void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img; 

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);


    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows*ref_img.cols*ref_img.channels());
    std::cout << "PASSED!\n";


}

void serial_grayscale(int rows, int cols, uchar4 *h_img, int *grey_img){
    for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                grey_img[i*cols+j] = 0.0722*(float)h_img[i*cols+j].x + 0.7152*(float)h_img[i*cols+j].y + 0.2126*(float)h_img[i*cols+j].z;
            }

        }
}


void serial_kmeans(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its)
{
    vector<float> distances;
    int chosenclass;
    int new_predictions[rows*cols];
    bool homogeneous = false;
    int itr = 0;
    int pixelClass, oldClass;
    int dist = 0;
    int min_dst;
    int min_loc;
    unsigned char b_avg;
    unsigned char g_avg;
    unsigned char r_avg;
    
    while(!homogeneous && itr < max_its)
    {
        std::cout << "starting iteration " << itr << "\n";
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                for(int k = 0; k < numclasses; k++)
                {
                    //calcuate avg pixel val
                    r_avg = centroids[k].at(0)/centroids[k].at(3);
                    g_avg = centroids[k].at(1)/centroids[k].at(3);
                    b_avg = centroids[k].at(2)/centroids[k].at(3);

                    /**Note .x -> B, .y -> G, .z -> R bc fuck opencv **/
                    //calculate the distance between point and cluster centroid pixel color
                    dist = sqrt(pow((r_avg - (float)h_img[i*cols+j].x),2) + pow((g_avg - (float)h_img[i*cols+j].y),2) 
                                + pow((b_avg - (float)h_img[i*cols+j].z),2)* 1.0);
                    if(k == 0 || dist < min_dst)
                    {
                        min_loc = k;
                        min_dst = dist;
                    }   
                }

                //update prediction 
                new_predictions[i*cols+j] = min_loc;

                //update cluster centroid
                if(min_loc != predictions[i*cols+j])
                {
                    oldClass = predictions[i*cols+j];
                    //add to new
                    centroids[min_loc].at(0) += (float)h_img[i*cols+j].x; //add the blue
                    centroids[min_loc].at(1) += (float)h_img[i*cols+j].y; //add the green
                    centroids[min_loc].at(2) += (float)h_img[i*cols+j].z; //add the red
                    centroids[min_loc].at(3) += 1; //add 1 to the population

                    //remove from old
                    centroids[oldClass].at(0) -= (float)h_img[i*cols+j].x;
                    centroids[oldClass].at(1) -= (float)h_img[i*cols+j].y;
                    centroids[oldClass].at(2) -= (float)h_img[i*cols+j].z;
                    centroids[oldClass].at(3) -= 1;
                }
             
                //predictions[i*cols+j] = new_predictions[i*cols+j];

            }
        }

        //assume homogenous, if an element is not reset to false
        homogeneous = true;

        for(int i = 0; i < rows*cols; i++)
        {
            if(predictions[i] != new_predictions[i])
                homogeneous = false;

            predictions[i] = new_predictions[i];
        }
        if(homogeneous)
            std::cout << "converged\n";
        itr++;
    }
}

void make_segment(int *predictions, uchar4 *o_img, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            switch(predictions[i*cols+j])
            {
                case 0:
                    o_img[i*cols+j] = make_uchar4(255,0,0,255);
                    break;
                case 1:
                    o_img[i*cols+j] = make_uchar4(0,255,0,255);
                    break;
                case 2:
                    o_img[i*cols+j] = make_uchar4(0,0,0,255);
                    break;
                case 3:
                    o_img[i*cols+j] = make_uchar4(100,50,75,255);
                    break;
                case 4: 
                    o_img[i*cols+j] = make_uchar4(200,20,80,255);
                    break;
                case 5:
                    o_img[i*cols+j] = make_uchar4(5,150,25,255);
                    break;
                case 6:
                    o_img[i*cols+j] = make_uchar4(150,150,150,255);
                    break;
                case 7:
                    o_img[i*cols+j] = make_uchar4(50,50,10,255);
                    break;
                case 8:
                    o_img[i*cols+j] = make_uchar4(40,20,200,255);
                    break;
                case 9:
                    o_img[i*cols+j] = make_uchar4(75,75,150,255);
                    break;
                
            }
        }
    }

}
/* find the horizon through a simple and nonrobust method - use grayscale images */
int find_horizon(int rows, int cols, int *img){
    cout << "rows: " << rows << ", cols: " << cols << endl;
    float prev_row_avg = 0, curr_row_avg;
    float diff = 0;
//    std::string modded_outfile = std::string("_cropped") + outfile.c_str();
    float max_diff = 0;
    int i_max = 0;

    for(int i = 0; i < rows; i++){
        curr_row_avg = 0;
        for(int j = 0; j < cols; j++){
            curr_row_avg += img[i * cols + j];
        }
        if(i == 0) {
            prev_row_avg = curr_row_avg / cols;
            continue;
        }
        curr_row_avg /= cols; //cout << "row: " << i << ", curr avg: " << curr_row_avg << ", ";
        diff = abs(curr_row_avg - prev_row_avg);
        //cout << "prev avg: " << prev_row_avg << ", diff: " << diff << "\n";
        if(diff > 1 && i > 15 && i < 1185) {
            //cout << "condition met.. return i: " << i << "\n";
            if(diff > max_diff){
                max_diff = diff;
                i_max = i;
            }
        }
        prev_row_avg = curr_row_avg;
    }
    cout << "max_i = " << i_max <<  endl;
    for (int k = 0; k < cols; k++){
        img[i_max * cols + k] = 255;
    }

/*
    cv::Mat horizon_img(rows, cols, CV_32SC1, (void*)img);
    cout << "write image" << endl;
    bool suc_gpu = cv::imwrite(modded_outfile.c_str(), horizon_img);
    if(!suc_gpu){
        std::cerr << "Couldn't write gpu image!\n";
        exit(1);
    }
    */
    return i_max;
}

void gray_segment(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its, const std::string &outfile){

	cv::Mat imrgba, o_img, h_out_img;
	uchar4 *h_o_img, *h_i_img;
	h_out_img.create(rows, cols, CV_8UC4);
    	
	h_o_img = (uchar4 *)h_out_img.ptr<unsigned char>(0);
	std::string modded_outfile = std::string("_cropped") + outfile.c_str();


	cout << "test 2" << endl;
//SERIAL
	//serial_kmeans(rows, cols, numclasses, predictions, centroids, h_img, max_its);

//SHARED MEMORY
    std::cout << "starting GPU k means shared...\n";

    
    your_kmeans_shared(rows, cols, numclasses, predictions, centroids, h_img, max_its);
    //your_kmeans(rows, cols, numclasses, predictions, centroids, h_img, max_its);

    //generate output image


/////////////////////////////////////////



	//generate output image
    
	make_segment(predictions, h_o_img, rows, cols);
	cv::Mat output_s(rows, cols, CV_8UC4, (void*)h_o_img); // generate serial output image.
	cout << "write image" << endl;
    	bool suc = cv::imwrite(modded_outfile.c_str(), output_s);
    	if(!suc){
        	std::cerr << "Couldn't write serial image!\n";
        	exit(1);
    	}
}


int crop_imgs(int rows, int cols, cv::Mat org_img){
    cv::Mat img, imgrgba;
    uchar4 *h_in_img;
    cout << "start crop" << endl;

    img = org_img;
    cv::cvtColor(img, imgrgba, cv::COLOR_BGR2RGBA);
    h_in_img = (uchar4 *)imgrgba.ptr<unsigned char>(0);

    int grayImg[rows * cols];
    serial_grayscale(rows, cols, h_in_img, grayImg);
    int horizon_row = find_horizon(rows, cols, grayImg);

    cout << "horz row " << horizon_row << endl;
	 
	cv::Mat cropped_image = org_img(cv::Range(horizon_row,1200), cv::Range(0,1920));

    org_img = cropped_image;
	

    rows = rows - horizon_row;

    return rows;
}

void create_centroids(const std::string path, int block_size){

    cv::Mat img, imgrgba;
    cout << "start files" << endl;
 //////////////////////////////   
    uchar4 *h_in_img, *h_o_img, *r_o_img, *h_o_shared; // pointers to the actual image input and output pointers  
    uchar4 *d_in_img, *d_o_img;
    uchar4 *h_o_img_gpu, *h_o_img_gpu_shared;

    int numclasses = 2;
    int max_its = 50;

    int classes[2] = {1,2};

    cv::Mat imrgba, o_img, h_out_img, h_out_img_shared, r_out_img, h_out_img_gpu, h_out_img_gpu_shared, d_out_img, imgray;

    std::string infile;
    std::string outfile;
    std::string reference;
    std::string outfile_shared;
    std::string gray;
    
    
 //////////////////
    
    
    for (const auto & entry : std::filesystem::directory_iterator(path)){

	cout << entry.path().string() << endl;

        img = cv::imread(entry.path().c_str(), cv::IMREAD_COLOR);
        cout << " read img "<< endl;
        if(img.empty()){
            std::cerr << "Image file couldn't be read, exiting\n";
            exit(1);
        }

    
    int rows = img.rows;
    int cols = img.cols;

    auto start = high_resolution_clock::now();

    rows = crop_imgs(rows,cols,img);
    cout << "cropped rows: " << endl;


    cv::cvtColor(img, imgrgba, cv::COLOR_BGR2RGBA);
    h_in_img = (uchar4 *)imgrgba.ptr<unsigned char>(0);

	cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

    	o_img.create(img.rows, img.cols, CV_8UC4);
	h_out_img.create(img.rows, img.cols, CV_8UC4);
    	h_out_img_shared.create(img.rows, img.cols, CV_8UC4);
    	d_out_img.create(img.rows, img.cols, CV_8UC4);
    	r_out_img.create(img.rows, img.cols, CV_8UC4);
    	h_out_img_gpu.create(img.rows, img.cols, CV_8UC4);
    	h_out_img_gpu_shared.create(img.rows, img.cols, CV_8UC4);

    	h_in_img = (uchar4 *)imrgba.ptr<unsigned char>(0);
    	h_o_img = (uchar4 *)h_out_img.ptr<unsigned char>(0);
    	h_o_shared = (uchar4 *)h_out_img_shared.ptr<unsigned char>(0);
    	d_o_img = (uchar4 *)h_out_img.ptr<unsigned char>(0);
    	h_o_img_gpu = (uchar4 *) h_out_img_gpu.ptr<unsigned char>(0);
    	h_o_img_gpu_shared = (uchar4 *) h_out_img_gpu.ptr<unsigned char>(0);

    	const size_t  numPixels = img.rows*img.cols;
    	cout << "num cols: " << cols << endl;


	int predictions[numPixels];
    	vector<float> centroids[numclasses];
    	int rand_class;

    	int predictions_gpu[numPixels];
    	vector<float> centroids_gpu[numclasses];

    	int predictions_gpu_shared[numPixels];
    	vector<float> centroids_gpu_shared[numclasses];


	/*

	for(int i = 0; i < numclasses; i++){
        	for(int j = 0; j < 4; j++){
            		centroids[i].push_back(0.0);
            		centroids_gpu[i].push_back(0.0);
            		centroids_gpu_shared[i].push_back(0.0);
        	}
    	}

    	// Traversing of vectors v to print
    	// elements stored in it
    	for(int i = 0; i < numclasses; i++) {
        	for(auto it = centroids[i].begin();
             		it != centroids[i].end(); it++) {
	        }
    	}

  	//set up initital predicitons for each pixel
	////////////////////////////////////////////////////////////////////////////////////
	//PREDICTIONS
    	srand(time(0));
    	int g_class;
    	for(int i = 0; i < rows; i++){
        	for(int j = 0; j < cols; j++){
            g_class = (float)(rand() % 2 + 0);
            predictions[i*img.cols+j] = g_class;

            centroids[g_class].at(0) += h_in_img[i*cols+j].x;
            centroids[g_class].at(1) += h_in_img[i*cols+j].y;
            centroids[g_class].at(2) += h_in_img[i*cols+j].z;
            centroids[g_class].at(3) += 1;

        	}
    	}

	// Traversing of vectors v to print
    	// elements stored in it
    	for (int i = 0; i < numclasses; i++) {
        	// Displaying element at each column,
        	// begin() is the starting iterator,
        	// end() is the ending iterator
        	for (auto it = centroids[i].begin();
             	it != centroids[i].end(); it++) {

            // (*it) is used to get the
            // value at iterator it
        	}
    	}

        */

    gray_segment(rows, cols, numclasses, predictions, centroids, h_in_img, max_its,entry.path().filename().string());	

    auto stop = std::chrono::duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();ofstream stats;
    cout << "timing: " << stop << endl; 
    stats.open ("metrics.txt");
	    cout << "ERRROR OPENING OUTPUT FILE FOR STAT WRITING" << endl;
    stats << "GPU," << rows << "," << cols << "," << "," << block_size << "," << stop << endl;
    stats.close();
    }
}

int main(int argc, char const *argv[]) {

    std::string path = "/scratch1/mfaykus/cpsc8810/gray_depth_img/ppt_images";
    int block_size = atoi(argv[1]);
    create_centroids(path.c_str(),block_size);
    int rows, cols, numClasses;

    return 0;
}






