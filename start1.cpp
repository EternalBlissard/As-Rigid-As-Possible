
#include <iostream> 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc.hpp> 
using namespace cv; 
using namespace std; 

// Driver Code 
int main(int argc, char** argv) 
{ 
	cv::Mat image = imread("../image.png");
	cv::Mat out = image.clone();
	cv::Mat out2 = image.clone();
	// Check if the image is created 
	// successfully 
	if (!image.data) { 
		cout << "Could not open or find the image"; 

		return 0; 
	} 
	//Median Blur
	clock_t start = clock();
	medianBlur(image,out,5);
	double endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
	cout<<"Median Blur Time"<<endTime<<endl;
	cv::imshow("median blur",out);
	cv::waitKey(0);

	//GreyScale
	start = clock();
	cv::cvtColor(out, out2, COLOR_BGR2GRAY);
	endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
	cout<<"Grey Scale Time"<<endTime<<endl;
	cv::imshow("Grey Scale",out2);
	cv::waitKey(0);

	//Sobel Filter individual gradients
	Mat sobelx, sobely, sobelxy;
    // Sobel(out2, sobelx, CV_64F, 1, 0, 3);
    // Sobel(out2, sobely, CV_64F, 0, 1, 3);
	//Sobel(out2, sobelxy, CV_64F, 1, 1, 3);
    // // Display Sobel edge detection images
    // imshow("Sobel X", sobelx);
    // waitKey(0);
    // imshow("Sobel Y", sobely);
    // waitKey(0);
	// imshow("Sobel XY", sobelxy);
    // waitKey(0);

	//Sobel FIlter with white edges
	Mat edges;
	start = clock();
    Canny(out2, edges, 19, 20, 3, false);
	endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
	cout<<"Sobel Filter with white edges endTime"<<endTime<<endl;

    // Display canny edge detected image
    imshow("Canny edge detection", edges);
    waitKey(0);

	return 0; 
} 