#include <iostream>
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
using namespace std::chrono;

#include "utils.h"
#include "gaussian_kernel.h"


using namespace std;

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

/*
void serial_grayscale(int rows, int cols, uchar4 *h_img, int *grey_img){
    for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                gray_img[i*cols+j] = 0.0722*(float)h_img[i*cols+j].x + 0.7152*(float)h_img[i*cols+j].y + 0.2126*(float)h_img[i*cols+j].z;
            }

        }
}
*/

//RMS Root Mean squared contrast
//standard deviation of the pixel intensities
void serial_contrast(int rows, int cols, uchar4 *h_img, float *contrast){

int gray_img[rows*cols];
int sum=0;
int counter=0;
//float contrast = 0.0;
float variance = 0.0;
float mean = 0.0;
float SD = 0.0;

 /**Note .x -> B, .y -> G, .z -> R bc fuck opencv **/
for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                gray_img[i*cols+j] = 0.0722*h_img[i*cols+j].x + 0.7152*h_img[i*cols+j].y + 0.2126*h_img[i*cols+j].z;
                sum += gray_img[i*cols+j];
                counter++; 
            }
        }

mean = sum/counter;
for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                variance += pow(gray_img[i*cols+j] - mean, 2);
            }
        }
variance = variance/counter;

SD = sqrt(variance);
*contrast = SD;

printf("mean = %f\nvariance = %f\nstandard deviation - contrast = %f\n",mean,variance,SD);
}


void serial_kmeans_contrast(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its)
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
    unsigned char gray_avg;
    int gray_img[rows*cols];
    float contrast = 0.0;
    float gray_contrast = 0.0;

    serial_contrast(rows, cols, h_img, &contrast);

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

                    gray_avg = 0.0722*b_avg + 0.7152*g_avg + 0.2126*r_avg;
                    gray_contrast = (contrast + gray_avg)/2;


                    
                    gray_img[i*cols+j] =  0.0722*(float)h_img[i*cols+j].x + 0.7152*(float)h_img[i*cols+j].y + 0.2126*(float)h_img[i*cols+j].z;

                    dist = sqrt(pow((gray_contrast - (float)gray_img[i*cols+j]),2));
                    
                    
                    
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

void serial_kmeans_contrast_color(int rows, int cols, int numclasses, int *predictions, vector<float> *centroids, uchar4 *h_img, int max_its)
{
    vector<float> distances;
    int chosenclass;
    int new_predictions[rows*cols];
    bool homogeneous = false;
    int itr = 0;
    int pixelClass, oldClass;
    int dist = 0;
    float dist_gray = 0.0;
    float dist_color = 0.0;
    int min_dst;
    int min_loc;
    unsigned char b_avg;
    unsigned char g_avg;
    unsigned char r_avg;
    unsigned char gray_avg;
    int gray_img[rows*cols];
    float contrast = 0.0;
    float gray_contrast = 0.0;

    serial_contrast(rows, cols, h_img, &contrast);

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

                    gray_avg = 0.0722*b_avg + 0.7152*g_avg + 0.2126*r_avg;
                    gray_contrast = (contrast + gray_avg)/2;


                    
                    gray_img[i*cols+j] =  0.0722*(float)h_img[i*cols+j].x + 0.7152*(float)h_img[i*cols+j].y + 0.2126*(float)h_img[i*cols+j].z;

                    dist_gray = sqrt(pow((gray_contrast - (float)gray_img[i*cols+j]),2));
                    

                    dist_color = sqrt(pow((r_avg - (float)h_img[i*cols+j].x),2) + pow((g_avg - (float)h_img[i*cols+j].y),2) 
                                + pow((b_avg - (float)h_img[i*cols+j].z),2)* 1.0);

                    
                    dist = (0.3*dist_gray + 0.7*dist_color);

                    
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


int main(int argc, char const *argv[]) {
   
    uchar4 *h_in_img, *h_o_img, *r_o_img, *h_o_shared; // pointers to the actual image input and output pointers  
    uchar4 *d_in_img, *d_o_img;
    uchar4 *h_o_img_gpu, *h_o_img_gpu_shared; 

    int numclasses = 4;
    int max_its = 100;

    int classes[4] = {1,2,3,4};

    cv::Mat imrgba, o_img, h_out_img, h_out_img_shared, r_out_img, h_out_img_gpu, h_out_img_gpu_shared, d_out_img; 

    std::string infile; 
    std::string outfile; 
    std::string reference;
    std::string outfile_shared; 


    switch(argc){
        case 2:
            infile = std::string(argv[1]);
            outfile = "classified_gpu.png";
            reference = "classified_serial.png";
            outfile_shared = "classified_gpu_shared.png";
        case 3:
            infile = std::string(argv[1]);
            outfile = std::string(argv[2]);
            reference = "classified_serial.png";
            outfile_shared = "classified_gpu_shared.png";
            break;
        case 4:
            infile = std::string(argv[1]);
            outfile = std::string(argv[2]);
            reference = std::string(argv[3]);
            outfile_shared = "classified_gpu_shared.png";
            break;
        default: 
                std::cerr << "Usage ./gblur <in_image> <out_image> <reference_file> \n";
                infile = "meeting.jpg";
                outfile = "classified_gpu.png";
                outfile_shared = "classified_gpu_shared.png";
                reference = "classified_serial.png";

   }
//hello
    // preprocess 
    cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR); 
    if(img.empty()){
        std::cerr << "Image file couldn't be read, exiting\n"; 
        exit(1);
    }

    cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);
    //cv::cvtColor(img, h_out_img, cv::COLOR_BGR2RGBA);
    //cv::cvtColor(img, r_out_img, cv::COLOR_BGR2RGBA);

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
    int rows = img.rows;
    int cols = img.cols;

    int predictions[numPixels];
    vector<float> centroids[numclasses];
    int rand_class;

    int predictions_gpu[numPixels];
    vector<float> centroids_gpu[numclasses];

    int predictions_gpu_shared[numPixels];
    vector<float> centroids_gpu_shared[numclasses];


    for(int i = 0; i < numclasses; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            centroids[i].push_back(0.0);
            centroids_gpu[i].push_back(0.0);
            centroids_gpu_shared[i].push_back(0.0);
        }
    }

        // Traversing of vectors v to print
    // elements stored in it
    for (int i = 0; i < numclasses; i++) {
  
        cout << "Elements at index "
             << i << ": ";
  
        // Displaying element at each column,
        // begin() is the starting iterator,
        // end() is the ending iterator
        for (auto it = centroids[i].begin();
             it != centroids[i].end(); it++) {
  
            // (*it) is used to get the
            // value at iterator is
            // pointing
            cout << *it << ' ';
        }
        cout << endl;
    }

    std::cout << "created vectors..\n";

    std::cout << "Image rows: " << rows << "image cols: " << cols << "\n";

    //set up initital predicitons for each pixel
    srand(time(0));
    int g_class;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            // if(h_in_img[i*cols+j].y > 100  && h_in_img[i*cols+j].z < 60)
            //     predictions[i*cols+j] = g_class = 0; //green
            // else if(h_in_img[i*cols+j].x < 50  && h_in_img[i*cols+j].z > 100)
            //     predictions[i*cols+j] = g_class = 1; //blue
            // else
            //     predictions[i*cols+j] = g_class = 2; //white
            // //std::cout << "Updating class number: " << rand_class << "\n";
            // centroids[g_class].at(0) += h_in_img[i*cols+j].x;
            // centroids[g_class].at(1) += h_in_img[i*cols+j].y;
            // centroids[g_class].at(2) += h_in_img[i*cols+j].z;
            // centroids[g_class].at(3) += 1;

            g_class = (float)(rand() % 2 + 0);
            predictions[i*img.cols+j] = g_class;
            //std::cout << "Updating class number: " << rand_class << "\n";
            centroids[g_class].at(0) += h_in_img[i*cols+j].x;
            centroids[g_class].at(1) += h_in_img[i*cols+j].y;
            centroids[g_class].at(2) += h_in_img[i*cols+j].z;
            centroids[g_class].at(3) += 1;


            centroids_gpu[g_class].at(0) = centroids[g_class].at(0);
            centroids_gpu[g_class].at(1) = centroids[g_class].at(1);
            centroids_gpu[g_class].at(2) = centroids[g_class].at(2);
            centroids_gpu[g_class].at(3) = centroids[g_class].at(3);

            centroids_gpu_shared[g_class].at(0) = centroids[g_class].at(0);
            centroids_gpu_shared[g_class].at(1) = centroids[g_class].at(1);
            centroids_gpu_shared[g_class].at(2) = centroids[g_class].at(2);
            centroids_gpu_shared[g_class].at(3) = centroids[g_class].at(3);



        }
    }

    // Traversing of vectors v to print
    // elements stored in it
    for (int i = 0; i < numclasses; i++) {
  
        cout << "Elements at index "
             << i << ": ";
  
        // Displaying element at each column,
        // begin() is the starting iterator,
        // end() is the ending iterator
        for (auto it = centroids[i].begin();
             it != centroids[i].end(); it++) {
  
            // (*it) is used to get the
            // value at iterator is
            // pointing
            cout << *it << ' ';
        }
        cout << endl;
    }

    std::cout << "starting serial k means...\n";

    //(int rows, int cols, int numclasses, int *predictions, vector<float> *centroidInfo)



//    serial_contrast( rows, cols, h_in_img);
    auto start = high_resolution_clock::now();
    //serial_kmeans(rows, cols, numclasses, predictions, centroids, h_in_img, max_its);
    //serial_kmeans_contrast(rows, cols, numclasses, predictions, centroids, h_in_img, max_its);
    serial_kmeans_contrast_color(rows, cols, numclasses, predictions, centroids, h_in_img, max_its);
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("The execution time in microseconds for serial implementation: ");
    std::cout << duration;
    printf("\n");

    //generate output image
    make_segment(predictions, h_o_img, rows, cols);
    cv::Mat output_s(img.rows, img.cols, CV_8UC4, (void*)h_o_img); // generate serial output image.
    bool suc = cv::imwrite(reference.c_str(), output_s);
    if(!suc){
        std::cerr << "Couldn't write serial image!\n";
        exit(1);
    }




    
    std::cout << "starting GPU k means...\n";

    float GPU_contrast = 0.0;
    serial_contrast(rows, cols, h_in_img, &GPU_contrast);

    start = high_resolution_clock::now();
    //serial_contrast(rows, cols, h_in_img, &GPU_contrast);
     //std::cout << "made it1\n";
    your_kmeans(rows, cols, numclasses, predictions_gpu, centroids_gpu, h_in_img, max_its);
    //your_kmeans_contrast(rows, cols, numclasses, predictions_gpu, centroids_gpu, h_in_img, max_its, GPU_contrast);
    stop = high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("The execution time in microseconds for GPU implementation: ");
    std::cout << duration;
    printf("\n");

    //generate output image

    cout << "make segment gpu" << endl;
    make_segment(predictions_gpu, d_o_img, rows, cols);
    cout << "segment gpu" << endl;
    cv::Mat output_s_gpu(img.rows, img.cols, CV_8UC4, (void*)d_o_img); // generate gpu output image.
    cout << "write image" << endl;
    bool suc_gpu = cv::imwrite(outfile.c_str(), output_s_gpu);
    if(!suc_gpu){
        std::cerr << "Couldn't write gpu image!\n";
        exit(1);
    }





    std::cout << "starting GPU k means shared...\n";

    start = high_resolution_clock::now();
    //serial_contrast(rows, cols, h_in_img, &GPU_contrast);
    your_kmeans_shared(rows, cols, numclasses, predictions_gpu_shared, centroids_gpu_shared, h_in_img, max_its);
    //your_kmeans_contrast_shared(rows, cols, numclasses, predictions_gpu_shared, centroids_gpu_shared, h_in_img, max_its, GPU_contrast);
    stop = high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("The execution time in microseconds for GPU implementation: ");
    std::cout << duration;
    printf("\n");

    //generate output image

    cout << "make segment gpu" << endl;
    make_segment(predictions_gpu_shared, h_o_shared, rows, cols);
    cout << "segment gpu" << endl;
    cv::Mat output_s_gpu_shared(img.rows, img.cols, CV_8UC4, (void*)h_o_shared); // generate gpu output image.
    cout << "write image" << endl;
    bool suc_gpu_shared = cv::imwrite(outfile_shared.c_str(), output_s_gpu_shared);
    if(!suc_gpu){
        std::cerr << "Couldn't write gpu image!\n";
        exit(1);
    }

    return 0;
}





