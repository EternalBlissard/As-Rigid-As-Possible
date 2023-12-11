#include <iostream> 
#include <opencv2/highgui.hpp> 
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc.hpp> 
#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
using namespace Eigen;
using namespace std; 
using namespace cv; 

const double INF=1e9;

int accessPixelMedian(unsigned char * arr, int col, int row, int k, int width, int height,int ksize);
void Median_blur2D(unsigned char * arr, unsigned char * result, int width, int height,int ksize)
{
    //Finding the median blur a given image which has a kernel size of ksize
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {
            for (int k = 0; k < 3; k++) 
            {
                result[3 * row * width + 3 * col + k] = accessPixelMedian(arr, col, row, k, width, height,ksize);
            }
        }
    }
}

int accessPixelMedian(unsigned char * arr, int col, int row, int k, int width, int height,int ksize) 
{
    vector<int> freearr;
    int val = ksize/2;
    //val is ksize/2 as we need to loop ksize/2 steps up ksize/2 down and vice versa
    for (int j = -1*val; j <= val; j++) 
    {
        for (int i = -1*val; i <= val; i++) 
        {
            if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width) 
            {
                int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
                freearr.push_back(color);
            }
        }
    }
    int len = sizeof(freearr)/sizeof(freearr[0]);
    sort(freearr.begin(),freearr.end());
    if(len%2==0){
        return (freearr[len/2 -1]+freearr[len/2])/2;
    }
    return freearr[len/2];
}
void GrayScale(unsigned char* arr,unsigned char* out,int width, int height){
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {   
            //accessing each pixel 
            int r = arr[(row) * 3 * width + (col) * 3];
            int g = arr[(row) * 3 * width + (col) * 3 + 1];
            int b = arr[(row) * 3 * width + (col) * 3 + 2];

            //averaging rgb of each pixel and setting those
            int grey = (r+g+b)/3;
            out[(row) * 3 * width + (col) * 3] = grey;
            out[(row) * 3 * width + (col) * 3 + 1] = grey;
            out[(row) * 3 * width + (col) * 3 + 2] = grey;
        }
    }
}

void SobelFilter (unsigned char* arr,unsigned char* out,int width, int height){
    //xgradient
    vector<vector<int>> xKernel = {{-1,0,1},
                                   {-2,0,2},
                                   {-1,0,1}};

    //ygradient 
    vector<vector<int>> yKernel = {{1,2,1},
                                   {0,0,0},
                                   {-1,-2,-1}};

    int count=0;
    for (int row = 0; row < height; row++) 
    {
        
        for (int col = 0; col < width; col++) 
        {
            //for(int idx=0;idx<3;idx++){ 
                    int idx=0;
                    out[(row) * 3 * width + (col) * 3 + idx] = 0;
                    out[(row) * 3 * width + (col) * 3 + idx+1] = 0;
                    out[(row) * 3 * width + (col) * 3 + idx+2] = 0;
                    int xVal = 0;
                    int yVal = 0;
                    for (int l=-1;l<=1;l++){
                        for(int k=-1;k<=1;k++){
                            if((row+l)>=0 && (row+l)<height && (col+k)>=0 && (col+k)<width){
                                int color = arr[(row + l) * 3 * width + (col + k) * 3 + idx];
                                int weightx=xKernel[1+l][1+k];
                                int weighty=yKernel[1+l][1+k];
                                xVal+=color*weightx;
                                yVal+=color*weighty;
                            }
                        }
                    }
                    //Setting the resultant gradient for each pixel;
                    out[(row) * 3 * width + (col) * 3 + idx]= (abs(xVal)+abs(yVal))/2;
                    out[(row) * 3 * width + (col) * 3 + idx+1]= (abs(xVal)+abs(yVal))/2;
                    out[(row) * 3 * width + (col) * 3 + idx+2]= (abs(xVal)+abs(yVal))/2;
                
            //}
        }    
    }
}

void ThresholdFilter (unsigned char* arr,unsigned char* out,int width, int height,int threshold){
    //Chooses only those pixel which are above a particular value

    for (int row = 0; row < height; row++) 
    {
        
        for (int col = 0; col < width; col++) 
        {
            //for(int idx=0;idx<3;idx++){ 
                    if(arr[(row) * 3 * width + (col) * 3]>=threshold){
                        out[(row) * 3 * width + (col) * 3]=255;
                        out[(row) * 3 * width + (col) * 3+1]=255;
                        out[(row) * 3 * width + (col) * 3+2]=255;
                    }else{
                        out[(row) * 3 * width + (col) * 3]=0;
                        out[(row) * 3 * width + (col) * 3+1]=0;
                        out[(row) * 3 * width + (col) * 3+2]=0;
                    }
                
            //}
        }    
    }
}

void nonMaximalSuppression(unsigned char* arr,unsigned char* out,int width, int height,int threshold){
    
    //Chooses only those pixel which are above a particular value and maximum in neighbour hood of size 3

    for (int row = 0; row < height; row++) 
    {
        // cout<<row<<" ";
        for (int col = 0; col < width; col++) 
        {
            bool isMaximum = true;
            for (int neighRow=row-2;neighRow<=row+2;neighRow++){
                for(int neighCol=col-2;neighCol<=col+2;neighCol++){
                    if(((neighCol)>=0 && (neighCol)<width && (neighRow)>=0 && (neighRow)<height )){
                        if(arr[(neighRow) * 3 * width + (neighCol) * 3]>arr[(row) * 3 * width + (col) * 3] || arr[(row) * 3 * width + (col) * 3]<threshold){
                            isMaximum=false;
                            break;
                        }
                    }
                }
                if(!isMaximum){
                    break;
                }
            }
            if(isMaximum){
                out[(row) * 3 * width + (col) * 3]=255;
                out[(row) * 3 * width + (col) * 3+1]=255;
                out[(row) * 3 * width + (col) * 3+2]=255;
            }else{
                out[(row) * 3 * width + (col) * 3]=0;
                out[(row) * 3 * width + (col) * 3+1]=0;
                out[(row) * 3 * width + (col) * 3+2]=0;
            }
        }
    }
}


void setBoundaryzero(unsigned char* arr,unsigned char* out,int width, int height){

    //Sets the boundary pixel values to 0
    for (int row = 0; row < height; row++) 
    {
        // cout<<row<<" ";
        for (int col = 0; col < width; col++) 
        {
            if(col==0 || col==width-1 || row==0 || row==height-1){
                out[(row) * 3 * width + (col) * 3] = 0;
                out[(row) * 3 * width + (col) * 3 + 1] = 0;
                out[(row) * 3 * width + (col) * 3 + 2] = 0;
            }
            else{
                out[(row) * 3 * width + (col) * 3] = arr[(row) * 3 * width + (col) * 3];
                out[(row) * 3 * width + (col) * 3 + 1] = arr[(row) * 3 * width + (col) * 3+1];
                out[(row) * 3 * width + (col) * 3 + 2] = arr[(row) * 3 * width + (col) * 3+2];
            }
        }
    }
}

cv::Mat3b setClonezero(cv::Mat3b in,int width, int height){
    //clones a image and then set all corresponding values to 0

    cv::Mat3b out = in.clone();
    for (int row = 0; row < height; row++) 
    {
        // cout<<row<<" ";
        for (int col = 0; col < width; col++) 
        {
            out[row][col][0]=0;
            out[row][col][1]=0;
            out[row][col][2]=0;
        }
    }
    return out;
}

void getPoints(unsigned char* arr, vector<int>&xPoints,vector<int>&yPoints,int width, int height){
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {
            if(arr[(row) * 3 * width + (col) * 3]){
                xPoints.push_back(col);
                yPoints.push_back(row);
            }
        }
    }
}

void minPointsDistance(unsigned char* out, vector<int>&xPoints,vector<int>&yPoints,int width, int height,int minDist){
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {
            out[(row) * 3 * width + (col) * 3 + 0] = 0;
            out[(row) * 3 * width + (col) * 3 + 1] = 0;
            out[(row) * 3 * width + (col) * 3 + 2] = 0;
        }
    }
    int len = xPoints.size();
    vector<bool> lessDist(len,false);
    for(int i=0;i<len;i++){
        if(!lessDist[i]){
            for(int j=i+1;j<len;j++){
                if(!lessDist[j]){
                    int x1 = xPoints[i];
                    int y1 = yPoints[i];
                    int x2 = xPoints[j];
                    int y2 = yPoints[j];
                    int dist = ((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2));
                    if(dist<minDist*minDist){
                        lessDist[i] = true;
                    }
                }
            }
        }
        
    }

    for(int i=0;i<len;i++){
        if(!lessDist[i]){
            int x1 = xPoints[i];
            int y1 = yPoints[i];
            out[(y1) * 3 * width + (x1) * 3 + 2] = 255;
            out[(y1) * 3 * width + (x1) * 3 + 1] = 255;
            out[(y1) * 3 * width + (x1) * 3 + 0] = 255;
        }
    }
}

class Point2dim {
    public:
        double x, y;
        Point2dim(double Px, double Py){
            x=Px;
            y=Py;
        }
};


class Trianglee {
    public:
        Point2dim*points[3];
        Point2dim*Centroidpt;
        Trianglee* neighbors[3];
        bool isCentroidset;
        int mapped;
        Trianglee(Point2dim*p1, Point2dim*p2, Point2dim*p3) {
            points[0] = p1;
            points[1] = p2;
            points[2] = p3;
            neighbors[0] = NULL;
            neighbors[1] = NULL;
            neighbors[2] = NULL;
        }
        Trianglee(std::set<Point2dim*> &pt) {
            vector<Point2dim*> tri;
            for(auto Pt:pt){
                tri.push_back(Pt);
            }
            points[0] = tri[0];
            points[1] = tri[1];
            points[2] = tri[2];
            isCentroidset=false;
            // Centroidpt->x= (tri[0]->x+tri[1]->x+tri[2]->x)/3;
            // Centroidpt->y= (tri[0]->y+tri[1]->y+tri[2]->y)/3;
            neighbors[0] = NULL;
            neighbors[1] = NULL;
            neighbors[2] = NULL;
            mapped = 0;
        }
        void setCentroid(){
            // cout<<"hi1hii";
            
            double centX= (points[0]->x+points[1]->x+points[2]->x)/3;
            // cout<<"hi2hiii";
            // = Point2dim(col/100.0,row/100.0);
            double centY= (points[0]->y+points[1]->y+points[2]->y)/3;
            Centroidpt = new Point2dim(centX*100,centY*100);
            isCentroidset=true;
        }
        // SparseMatrix<double> toMatrix() const {
        //     SparseMatrix<double> matrix(3, 3);
        //     for (int i = 0; i < 3; ++i) {
        //         for (int j = 0; j < 3; ++j) {
        //             if (i != j) {
        //             matrix.setCoefficient(i, j, 1.0); // Mark connected edges
        //             }
        //         }
        //     }
        //     return matrix;
        // }
        void show(){
            cout<<"Point 1 x"<<points[0]->x<<" y "<<points[0]->y;
            cout<<"Point 2 x"<<points[1]->x<<" y "<<points[1]->y;
            cout<<"Point 3 x"<<points[2]->x<<" y "<<points[2]->y<<endl;
        }
};

void getPointform(unsigned char* arr,int width, int height,std::vector<Point2dim*> &points){
    int count=0;
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            if(arr[(row) * 3 * width + (col) * 3 + 0]){
                //cout<<"row"<<row<<"col"<<col<<endl;
                Point2dim* p = new Point2dim(col/100.0,row/100.0);
                points.push_back(p);
                count++;
            }
        }
    }
    cout<<"Point Count"<<count<<endl;
}
Trianglee* create_super_triangle( std::vector<Point2dim*>& points) {
    double min_x = points[0]->x; 
    double min_y = points[0]->y;
    double max_x = points[0]->x;
    double max_y = points[0]->y;
	cout<<"super tringle creation in progress"<<endl;
    for ( auto p : points) {
        min_x = std::min(min_x, p->x);
        min_y = std::min(min_y, p->y);
        max_x = std::max(max_x, p->x);
        max_y = std::max(max_y, p->y);
    }
    Point2dim*p1=new Point2dim(0     , -0);
    Point2dim*p2=new Point2dim(2*max_x, 0);
    Point2dim*p3=new Point2dim(0, 2*max_y);
    cout<<"Point A"<<p1->x*100<<"  "<<p1->y<<endl;
    cout<<"Point B"<<p2->x*100<<"  "<<p2->y<<endl;
    cout<<"Point C"<<p3->x*100<<"  "<<p3->y<<endl;
    return new Trianglee(p1, p2, p3);
}
//Given 2 Points P,Q generates their line ax+by=c 
void lineFromPoints(Point2dim* P, Point2dim* Q,double& a, double& b,double& c)
{
	a = Q->y - P->y;//value of a
	b = P->x - Q->x;//value of b 
	c = a * (P->x) + b * (P->y); // corresponding c
    //the values are goverened by (y-y1)/(y2-y1)= (x-x1)/(x2-x1)
    //the above equation simplyfies to give the respective values
}
//Given 2 points P,Q with their line being ax+by=c, finds and makes their perpendicular
//bisector given by -bx+ay = c'
void perpenBisectorFromLine(Point2dim* P, Point2dim* Q,double& a, double& b,double& c)
{

	// Finding the  the mid point
	// x coordinates of the mid point
	double mid_pointx = (P->x + Q->x) / 2;

	// y coordinates of the mid point
	double mid_pointy = (P->y + Q->y) / 2;

	//As  c = -bx + ay
	c = -b * (mid_pointx)+ a * (mid_pointy);

	// Assign the coefficient of a and b
	double temp = a;
	a = -b;
	b = temp;
}

// Returns the intersection point of two lines
pair<double,double> LineInterXY(double a1, double b1,double c1, double a2,double b2, double c2)
{

	// Find determinant of the line inter section
	double determ = a1 * b2 - a2 * b1;

	double x = (b2 * c1 - b1 * c2);
    //cout<<"x"<<x<<"detx"<<determ<<endl;
    double y = (a1 * c2 - a2 * c1);
	//cout<<"y"<<y<<"detx"<<determ<<endl;
    if(determ!=0){
        x /= determ;
        y /= determ;
    }
    pair<double,double>Pairing= make_pair(x,y);
	return Pairing;
}


// Function to find if the point lies in/on the circumcenter or not
bool is_point_inside_circumcircle(Point2dim*point, Trianglee*triangle) {

    // cout<<"checking...";

    //Getting the Point P with whom the cicumcenter of triangle ABC would be tested with
    //Getting the Points ABC from the triangle 
    Point2dim* P = point;
    Point2dim* A = triangle->points[0];
    Point2dim* B = triangle->points[1];
    Point2dim* C = triangle->points[2];

	double a, b, c, e, f, g;
    //Line AB is made and is denoted by ax+by=c;
	lineFromPoints(A, B, a, b, c);

	// Line BC is made and represented as ex + fy = g
	lineFromPoints(B, C, e, f, g);

	// Converting lines AB and BC into their perpendicular bisectors.
	// After this, 
    // _|_ of AB : L = ax + by = c
	// _|_ of BC:  M = ex + fy = g
	perpenBisectorFromLine(A, B, a, b, c);
	perpenBisectorFromLine(B, C, e, f, g);

	// Getting The point of intersection of L and M which is the circumcenter of ABC
    pair<double,double> interPair = LineInterXY(a, b, c, e, f, g);
	double rx = interPair.first;
	double ry = interPair.second;
    Point2dim* r = new Point2dim(rx,ry);
	// Length of radius of circumcircle of ABC
	double rad = (r->x - A->x) * (r->x - A->x)+ (r->y - A->y) * (r->y - A->y);
    //cout<<"q"<<q;
	// Distance between radius and the given point P
	double dis = (r->x - P->x) * (r->x - P->x) + (r->y - P->y) * (r->y - P->y);
	//Condition for point lies inside circumcircle or on the circumcenter
	if (dis <= rad) {
		return true;
	}
    else{
        return false;
    }
    
   
}
void tryDrawingTrianglesV(Mat image, std::vector<Trianglee*> tri,int c);
void bowyer_watson(std::vector<Point2dim*>& points,std::vector<Trianglee*>&Tangle,cv::Mat3b img) {
    Trianglee* super_triangle = create_super_triangle(points);
    //std::vector<Trianglee*> triangulation = {super_triangle};
    int count =0;
    std::set<Point2dim*> superPoints={super_triangle->points[0],super_triangle->points[1],super_triangle->points[2]};
    std::set<set<Point2dim*>> triPoints={superPoints};
    //cout<<"hi";
    for (auto point : points){
    	//cout<<"bye";
    	// return;
        std::set<set<Point2dim*>> badTriangle ={};
        for(auto tri: triPoints){
            Trianglee* triangle = new Trianglee(tri);
            //triangle->show();
            if(is_point_inside_circumcircle(point,triangle)){
                badTriangle.insert(tri);
                //cout<<"hi >.<";
            }else{
                //cout<<"bye><><";
            }
        }
        //std::set<set<Point2dim*>> polyGon={};
        std::map<set<Point2dim*>,int> polyGon={};
        for(auto tri: badTriangle){
            Trianglee* triangle = new Trianglee(tri);
            set<Point2dim*> set1 ({triangle->points[0],triangle->points[1]});
            set<Point2dim*> set2 ({triangle->points[1],triangle->points[2]});
            set<Point2dim*> set3 ({triangle->points[2],triangle->points[0]});
            //triangle->show();
            polyGon[set1]++;
            polyGon[set2]++;
            polyGon[set3]++;
            
    
        }
        for(auto triRemove: badTriangle){
            for(auto triangle : triPoints){
                if(triRemove == triangle){
                    //cout<<"hi?";
                    triPoints.erase(triangle);
                    break;
                }
            }
        }

        for(auto edge: polyGon){
            // vector<Point2dim*> p;
            // for(auto pt:edge){
            //     p.push_back(pt);
            // }
            // Trianglee* triangle = new Trianglee(p[0],p[1],point);
            // triangulation.push_back(triangle);
            if(edge.second==1){
                set<Point2dim*> set1 = edge.first;
                set1.insert(point);
                triPoints.insert(set1);
            }
            
        }


        std::vector<Trianglee*> triangulation1;
        //cout<<"Triangulation"<<count++<<endl;
        count++;
        // for(auto tri:triPoints){
            
        //     Trianglee* triangle = new Trianglee(tri);
        //     triangle->show();
        //     triangulation1.push_back(triangle);
        // }
        //tryDrawingTriangles(image,  triangulation1,50);

    }
    for (auto tri:triPoints){
        Trianglee* triangle = new Trianglee(tri);
        bool pt1 = triangle->points[0]==super_triangle->points[0] || triangle->points[0]==super_triangle->points[1] || triangle->points[0]==super_triangle->points[2];
        bool pt2 = triangle->points[1]==super_triangle->points[0] || triangle->points[1]==super_triangle->points[1] || triangle->points[1]==super_triangle->points[2];
        bool pt3 = triangle->points[2]==super_triangle->points[0] || triangle->points[2]==super_triangle->points[1] || triangle->points[2]==super_triangle->points[2];
        if(pt1 || pt2 || pt3){
            triPoints.erase(tri);
        }
    }
    // Delete the super triangle as it was created with `new`
    delete super_triangle->points[0];
    delete super_triangle->points[1];
    delete super_triangle->points[2];
    delete super_triangle;

    cout<<"Worked on"<<count<<"points for making the triangulation"<<endl;

    for(auto tri:triPoints){
        Trianglee* triangle = new Trianglee(tri);
        Tangle.push_back(triangle);
    }
    std::vector<Trianglee*>Tangle_;
    for(auto tri:Tangle){
        if(tri->points[0]->x*100<1 || tri->points[0]->x>15){
            continue;
        }
        if(tri->points[1]->x*100<1 || tri->points[1]->x>15){
            continue;
        }
        if(tri->points[2]->x*100<1 || tri->points[2]->x>15){
            continue;
        }
        if(tri->points[0]->y*100<1 || tri->points[0]->y>15){
            continue;
        }
        if(tri->points[1]->y*100<1 || tri->points[1]->y>15){
            continue;
        }
        if(tri->points[2]->y*100<1 || tri->points[2]->y>15){
            continue;
        }
        Tangle_.push_back(tri);
    }
    Tangle=Tangle_;
    cout<<"Number of triangles in the final triangulation"<<Tangle.size()<<endl;
    
}





void tryDrawingTrianglesV(Mat image, std::vector<Trianglee*> tri,int c){

    if (!image.data) { 
		cout << "Could not open or find"
			<< " the image"; 

		return; 
	} 

    int count =0;
    cout<<"drawing now Vector triangulation now"<<endl;
    for (auto triangle:tri){
        if(triangle->points[0]->x*100<1 || triangle->points[0]->x>15){
            continue;
        }
        if(triangle->points[1]->x*100<1 || triangle->points[1]->x>15){
            continue;
        }
        if(triangle->points[2]->x*100<1 || triangle->points[2]->x>15){
            continue;
        }
        if(triangle->points[0]->y*100<1 || triangle->points[0]->y>15){
            continue;
        }
        if(triangle->points[1]->y*100<1 || triangle->points[1]->y>15){
            continue;
        }
        if(triangle->points[2]->y*100<1 || triangle->points[2]->y>15){
            continue;
        }
        Point p1(int(triangle->points[0]->x*100), int(triangle->points[0]->y*100));
        Point p2(int(triangle->points[1]->x*100), int(triangle->points[1]->y*100)); 
        Point p3(int(triangle->points[2]->x*100), int(triangle->points[2]->y*100)); 
        int thickness = 1; 
        
        //cout<<"hi"<<"Points 1 "<<triangle->points[0]->x<<" "<<triangle->points[0]->y<<" 2 "<<triangle->points[1]->x<<" "<<triangle->points[1]->y<<" 3 "<<triangle->points[2]->x<<" "<<triangle->points[2]->y<<endl;
        line(image, p1, p2, Scalar(255, 0, 0), thickness, LINE_AA); 
        line(image, p1, p3, Scalar( 255,0, 0), thickness,LINE_AA); 
        line(image, p3, p2, Scalar(255,0, 0), thickness, LINE_AA); 

        //break;
        count++;
        if(count==c){
            break;
        }
    }
	// Show our image inside window 
	imshow("Output"+to_string(count), image); 
	waitKey(0); 
}

void tryDrawingTriangles(Mat image, map<pair<Trianglee*,Trianglee*>,double> mapping,int c){

    if (!image.data) { 
		cout << "Could not open or find"
			<< " the image"; 

		return; 
	} 

    int count =0;
    cout<<"drawing now triangulation now"<<endl;
    for (auto p:mapping){
        auto triangle = p.first.first;
        auto triangle2 = p.first.second;
        double val = p.second;
        if(triangle->points[0]->x*100<1 || triangle->points[0]->x>15){
            continue;
        }
        if(triangle->points[1]->x*100<1 || triangle->points[1]->x>15){
            continue;
        }
        if(triangle->points[2]->x*100<1 || triangle->points[2]->x>15){
            continue;
        }
        if(triangle->points[0]->y*100<1 || triangle->points[0]->y>15){
            continue;
        }
        if(triangle->points[1]->y*100<1 || triangle->points[1]->y>15){
            continue;
        }
        if(triangle->points[2]->y*100<1 || triangle->points[2]->y>15){
            continue;
        }
        Point p1(triangle->points[0]->x*100, triangle->points[0]->y*100);
        Point p2(triangle->points[1]->x*100, triangle->points[1]->y*100); 
        Point p3(triangle->points[2]->x*100, triangle->points[2]->y*100); 
        int thickness = 1; 
        Point P1(triangle2->points[0]->x*100, triangle2->points[0]->y*100);
        Point P2(triangle2->points[1]->x*100, triangle2->points[1]->y*100); 
        Point P3(triangle2->points[2]->x*100, triangle2->points[2]->y*100); 
        //cout<<"hi"<<"Points 1 "<<triangle->points[0]->x<<" "<<triangle->points[0]->y<<" 2 "<<triangle->points[1]->x<<" "<<triangle->points[1]->y<<" 3 "<<triangle->points[2]->x<<" "<<triangle->points[2]->y<<endl;
        if(val<20){
        line(image, p1, p2, Scalar(0, 0, 255), thickness, LINE_AA); 
        line(image, p1, p3, Scalar( 0,0, 255), thickness,LINE_AA); 
        line(image, p3, p2, Scalar(0,0, 255), thickness, LINE_AA);
        // line(image, P1, P2, Scalar(255, 0,0), thickness, LINE_AA); 
        // line(image, P1, P3, Scalar( 255,0,0), thickness,LINE_AA); 
        // line(image, P3, P2, Scalar(255,0, 0), thickness, LINE_AA); 
        }else{
        line(image, p1, p2, Scalar(0, 255, 0), thickness, LINE_AA); 
        line(image, p1, p3, Scalar( 0,255, 0), thickness,LINE_AA); 
        line(image, p3, p2, Scalar(0,255, 0), thickness, LINE_AA); }
        
        cout<<val<<" ";
        //break;
        count++;
        if(count==c){
            break;
        }
    }
	// Show our image inside window 
	imshow("SrcTarOutput"+to_string(count), image); 
	waitKey(0); 
}

void tryDrawingTriangles(Mat image, map<pair<Trianglee*,Trianglee*>,double> mapping,int num,int c){

    if (!image.data) { 
		cout << "Could not open or find"
			<< " the image"; 

		return; 
	} 

    int count =0;
    cout<<"drawing now triangulation now"<<endl;
    for (auto p:mapping){
        auto triangle = p.first.first;
        auto triangle2 = p.first.second;
        triangle->setCentroid();
        triangle2->setCentroid();
        double val = pow((triangle->Centroidpt->x-triangle2->Centroidpt->x),2)+pow((triangle->Centroidpt->y-triangle2->Centroidpt->y),2);
        if(triangle->points[0]->x*100<1 || triangle->points[0]->x>15){
            continue;
        }
        if(triangle->points[1]->x*100<1 || triangle->points[1]->x>15){
            continue;
        }
        if(triangle->points[2]->x*100<1 || triangle->points[2]->x>15){
            continue;
        }
        if(triangle->points[0]->y*100<1 || triangle->points[0]->y>15){
            continue;
        }
        if(triangle->points[1]->y*100<1 || triangle->points[1]->y>15){
            continue;
        }
        if(triangle->points[2]->y*100<1 || triangle->points[2]->y>15){
            continue;
        }
        Point p1(triangle->points[0]->x*100, triangle->points[0]->y*100);
        Point p2(triangle->points[1]->x*100, triangle->points[1]->y*100); 
        Point p3(triangle->points[2]->x*100, triangle->points[2]->y*100); 
        int thickness = 1; 
        Point P1(triangle2->points[0]->x*100, triangle2->points[0]->y*100);
        Point P2(triangle2->points[1]->x*100, triangle2->points[1]->y*100); 
        Point P3(triangle2->points[2]->x*100, triangle2->points[2]->y*100); 
        //cout<<"hi"<<"Points 1 "<<triangle->points[0]->x<<" "<<triangle->points[0]->y<<" 2 "<<triangle->points[1]->x<<" "<<triangle->points[1]->y<<" 3 "<<triangle->points[2]->x<<" "<<triangle->points[2]->y<<endl;
        if(val<20){
        line(image, p1, p2, Scalar(0, 0, 255), thickness, LINE_AA); 
        line(image, p1, p3, Scalar( 0,0, 255), thickness,LINE_AA); 
        line(image, p3, p2, Scalar(0,0, 255), thickness, LINE_AA);
        // line(image, P1, P2, Scalar(255, 0,0), thickness, LINE_AA); 
        // line(image, P1, P3, Scalar( 255,0,0), thickness,LINE_AA); 
        // line(image, P3, P2, Scalar(255,0, 0), thickness, LINE_AA); 
        }else{
        line(image, p1, p2, Scalar(0, 255, 0), thickness, LINE_AA); 
        line(image, p1, p3, Scalar( 0,255, 0), thickness,LINE_AA); 
        line(image, p3, p2, Scalar(0,255, 0), thickness, LINE_AA); }
        
        cout<<val<<endl;
        //break;
        count++;
        if(count==c){
            break;
        }
    }
	// Show our image inside window 
	imshow("SrcTarOutput"+to_string(count)+" Iter "+to_string(num), image); 
	waitKey(0); 
}


void imagePreProcessing(cv::Mat3b &img,int medianKSize,int minDist){
    cv::Mat3b out = setClonezero(img,img.cols, img.rows);
    cout<<"Display image"<<endl;
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::imshow("blank", out);
    cv::waitKey(0);
    // guassian_blur2D(img.data, out.data, img.cols, img.rows);
    cv::Mat3b out2 = setClonezero(img,img.cols, img.rows);
    
    cout<<"Display median blur image"<<endl;
    // Applying median blur
    clock_t start = clock();
    Median_blur2D(img.data, out.data, img.cols, img.rows,medianKSize);
    double endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Median Blur Time"<<endTime<<endl;

    cv::imshow("blur", out);
    cv::waitKey(0);
    // medianBlur(img,out,9);
    
    cout<<"Display greyscale image"<<endl;
    //Apply Greyscale
    start = clock();
    GrayScale (out.data,out2.data, img.cols, img.rows);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Greyscale Time"<<endTime<<endl;

    cv::imshow("grey", out2);
    cv::waitKey(0);
    
    cout<<"Display Sobel Filtered image"<<endl;
    //Applying Sobel Filter    
    cv::Mat3b out3 = setClonezero(img,img.cols, img.rows);
    start = clock();
    SobelFilter(out2.data,out3.data, img.cols, img.rows);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Sobel Filter Time"<<endTime<<endl;

    cv::imshow("Sobel", out3);
    cv::waitKey(0);
    
    cout<<"Display threshold filter image"<<endl;
    //Applying Threshold Filter    
    start = clock();
    cv::Mat3b out4 = setClonezero(img,img.cols, img.rows);
    ThresholdFilter(out3.data,out4.data,img.cols,img.rows,20);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Threshold Filter Time"<<endTime<<endl;

    cv::imshow("Thres",out4);
    cv::waitKey(0);
    
    cout<<"Display Non Max Sup image"<<endl;
    //Applying nonMax Sup
    start =clock();
    cv::Mat3b out5 = setClonezero(img,img.cols, img.rows);
    nonMaximalSuppression(out3.data,out5.data,img.cols,img.rows,10);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Non Max Suppression Time"<<endTime<<endl;

    cv::imshow("Non Max",out5);
    cv::waitKey(0);
    
    cout<<"Display boundary0 image"<<endl;
    //Applying Boundary val to 0
    cv::Mat3b out6 = setClonezero(img,img.cols, img.rows);
    start=clock();
    setBoundaryzero(out5.data,out6.data,img.cols,img.rows);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"BoundarySet to 0 Time"<<endTime<<endl;

    cv::imshow("Boundaryset to 0",out6);
    cv::waitKey(0);
    
    cout<<"Display distance filter image"<<endl;
    //Applying distance filter 
    cv::Mat3b out7 = setClonezero(img,img.cols, img.rows);
    start =clock();
    vector <int> xPoints;
    vector<int> yPoints;
    getPoints(out6.data,xPoints,yPoints,img.cols,img.rows);
    minPointsDistance(out7.data,xPoints,yPoints,img.cols,img.rows,minDist);
    endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    cout<<"min Distance Time"<<endTime<<endl;

    cv::imshow("minDist"+to_string(minDist),out7);
    cv::waitKey(2);
    img=out7;
    
}

//Finds the best alignment
Trianglee* findBestFitTriangle(Trianglee* srcTriangle,Trianglee* tarTriangle,double srcpointXmin,double srcpointYmin,double tarpointXmin,double tarpointYmin){
    Trianglee* finalTriangle = srcTriangle;
    Trianglee* triangle = tarTriangle;
    vector<double> minArray;
    double val;
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    val=0;
    val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
    val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
    val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
    minArray.push_back(val);
    int ind=0;
    for(int iter=1;iter<6;iter++){
        if(minArray[iter]<minArray[ind]){
            ind=iter;
        }
    }
    if(ind){
        if(ind==1){
            return new Trianglee(tarTriangle->points[0],tarTriangle->points[2],tarTriangle->points[1]);
        }
        if(ind=2){
            return new Trianglee(tarTriangle->points[1],tarTriangle->points[0],tarTriangle->points[2]);
        }
        if(ind==3){
            return new Trianglee(tarTriangle->points[1],tarTriangle->points[2],tarTriangle->points[0]);
        }
        if(ind=4){
            return new Trianglee(tarTriangle->points[2],tarTriangle->points[1],tarTriangle->points[0]);
        }else{
            return new Trianglee(tarTriangle->points[2],tarTriangle->points[0],tarTriangle->points[1]);
        }
    }
    return tarTriangle;
}
map<pair<Trianglee*,Trianglee*>,double> mapping(std::vector<Trianglee*> SrcTriangulation,std::vector<Trianglee*> TarTriangulation){
    double srcpointXmin=10000;
    double srcpointYmin=10000;
    double tarpointXmin=10000;
    double tarpointYmin=10000;
    for (auto triangle:SrcTriangulation){
        srcpointXmin = max(min(srcpointXmin,triangle->points[2]->x),0.01);
        srcpointXmin = max(min(srcpointXmin,triangle->points[1]->x),0.01);
        srcpointXmin = max(min(srcpointXmin,triangle->points[0]->x),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[2]->y),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[1]->y),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[0]->y),0.01);
    }
    for (auto triangle:TarTriangulation){
        tarpointXmin = max(min(tarpointXmin,triangle->points[2]->x),0.01);
        tarpointXmin = max(min(tarpointXmin,triangle->points[1]->x),0.01);
        tarpointXmin = max(min(tarpointXmin,triangle->points[0]->x),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[2]->y),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[1]->y),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[0]->y),0.01);
    }
    cout<<"Src"<<srcpointXmin<<" "<<srcpointYmin;
    cout<<endl;
    cout<<"Tar"<<tarpointXmin<<" "<<tarpointYmin;
    cout<<endl;
    map<pair<Trianglee*,Trianglee*>,double> ans;
    for (auto triangle:SrcTriangulation){
        Trianglee* minTri=NULL;
        double minNum = 1000000000;
        int numInd = 0;
        int ITERATOR=0;
        for(auto finalTriangle:TarTriangulation){
            ITERATOR++;
            if(finalTriangle->mapped){
                // cout<<"hi";
                continue;
            }
            vector<double> minArray;
            double val;
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            minArray.push_back(val);
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            minArray.push_back(val);
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            minArray.push_back(val);
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            minArray.push_back(val);
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            minArray.push_back(val);
            val=0;
            val+=pow(((finalTriangle->points[0]->x-tarpointXmin)-(triangle->points[2]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[0]->y-tarpointYmin)-(triangle->points[2]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[1]->x-tarpointXmin)-(triangle->points[0]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[1]->y-tarpointYmin)-(triangle->points[0]->y-srcpointYmin))*100,2);
            val+=pow(((finalTriangle->points[2]->x-tarpointXmin)-(triangle->points[1]->x-srcpointXmin))*100,2);
            val+=pow(((finalTriangle->points[2]->y-tarpointYmin)-(triangle->points[1]->y-srcpointYmin))*100,2);
            // if(((finalTriangle->points[0]->x-tarpointXmin)==(triangle->points[0]->x-srcpointXmin))&&((finalTriangle->points[0]->y-tarpointYmin)==(triangle->points[0]->y-srcpointYmin))){
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[1]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[1]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[2]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[2]->y-srcpointYmin))){
            //             cout<<"hi1";
            //         }
            //     }
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[2]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[2]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[1]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[1]->y-srcpointYmin))){
            //             cout<<"hi2";
            //         }
            //     }
            // }
            // if(((finalTriangle->points[0]->x-tarpointXmin)==(triangle->points[1]->x-srcpointXmin))&&((finalTriangle->points[0]->y-tarpointYmin)==(triangle->points[1]->y-srcpointYmin))){
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[0]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[0]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[2]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[2]->y-srcpointYmin))){
            //             cout<<"hi3";
            //         }
            //     }
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[2]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[2]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[0]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[0]->y-srcpointYmin))){
            //             cout<<"hi4";
            //         }
            //     }
            // }
            // if(((finalTriangle->points[0]->x-tarpointXmin)==(triangle->points[2]->x-srcpointXmin))&&((finalTriangle->points[0]->y-tarpointYmin)==(triangle->points[2]->y-srcpointYmin))){
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[0]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[0]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[1]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[1]->y-srcpointYmin))){
            //             cout<<"hi5";
            //         }
            //     }
            //     if(((finalTriangle->points[1]->x-tarpointXmin)==(triangle->points[1]->x-srcpointXmin)) && ((finalTriangle->points[1]->y-tarpointYmin)==(triangle->points[1]->y-srcpointYmin))){
            //         if(((finalTriangle->points[2]->x-tarpointXmin)==(triangle->points[0]->x-srcpointXmin)) && ((finalTriangle->points[2]->y-tarpointYmin)==(triangle->points[0]->y-srcpointYmin))){
            //             cout<<"hi6";
            //         }
            //     }
            // }
            minArray.push_back(val);
            for(int iter=0;iter<6;iter++){
                if(minArray[iter]<minNum){
                    minNum = minArray[iter];
                    numInd = iter;
                    minTri = finalTriangle;
                }
            }
        }
        if(ITERATOR<TarTriangulation.size()){
            cout<<"Oh NO";
        }
        if(minTri){
            pair <Trianglee*,Trianglee*> p= make_pair(triangle,minTri);
            ans[p] = minNum;
            minTri->mapped=1;
        }
        
    }
    return ans;
}

map<pair<Trianglee*,Trianglee*>,double> centroidMapping(std::vector<Trianglee*> SrcTriangulation,std::vector<Trianglee*> TarTriangulation,double maxDist, double &srcXmin, double &srcYmin, double &tarXmin, double &tarYmin){
    double srcpointXmin=10000;
    double srcpointYmin=10000;
    double tarpointXmin=10000;
    double tarpointYmin=10000;
    int it = 0;
    for (auto triangle:SrcTriangulation){
        // it++;
        // cout<<it<<" ";
        triangle->setCentroid();
        srcpointXmin = max(min(srcpointXmin,triangle->points[2]->x),0.01);
        srcpointXmin = max(min(srcpointXmin,triangle->points[1]->x),0.01);
        srcpointXmin = max(min(srcpointXmin,triangle->points[0]->x),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[2]->y),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[1]->y),0.01);
        srcpointYmin = max(min(srcpointYmin,triangle->points[0]->y),0.01);
    }
    it=0;
    cout<<"hi2"<<endl;
    for (auto triangle:TarTriangulation){
        // it++;
        // cout<<it<<" ";
        triangle->setCentroid();
        tarpointXmin = max(min(tarpointXmin,triangle->points[2]->x),0.01);
        tarpointXmin = max(min(tarpointXmin,triangle->points[1]->x),0.01);
        tarpointXmin = max(min(tarpointXmin,triangle->points[0]->x),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[2]->y),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[1]->y),0.01);
        tarpointYmin = max(min(tarpointYmin,triangle->points[0]->y),0.01);
        // if(triangle->points[2]->x==inf || triangle->points[2]->y==inf){
        //     cout<<2/0;
        // }
    }
    cout<<"Src"<<srcpointXmin<<" "<<srcpointYmin;
    cout<<endl;
    cout<<"Tar"<<tarpointXmin<<" "<<tarpointYmin;
    cout<<endl;
    map<pair<Trianglee*,Trianglee*>,double> ans;
    map<Trianglee*,priority_queue<pair<double,Trianglee*>,vector<pair<double,Trianglee*>>,greater<pair<double,Trianglee*>> > > intermediateMap;
    double maxVal = maxDist*maxDist;
    //pre computed map
    for (auto triangle:SrcTriangulation){
        //Trianglee* minTri=NULL;
        
        // int numInd = 0;
        // int ITERATOR=0;
        priority_queue<pair<double,Trianglee*>,vector<pair<double,Trianglee*>>,greater<pair<double,Trianglee*>> >pq;
        for(auto finalTriangle:TarTriangulation){
            //ITERATOR++;
            // if(finalTriangle->mapped){
            //     // cout<<"hi";
            //     continue;
            // }
            
            double dist = pow((finalTriangle->Centroidpt->x-tarpointXmin)-(triangle->Centroidpt->x-srcpointXmin),2);
            dist+=pow((finalTriangle->Centroidpt->y-tarpointYmin)-(triangle->Centroidpt->y-srcpointYmin),2);
            if(dist<=maxVal){
                pair<double,Trianglee*> p = make_pair(dist,finalTriangle);
                pq.push(p);
                // cout<<dist<<" ";
            }
            
            
            // minArray.push_back(val);
            // for(int iter=0;iter<6;iter++){
            //     if(minArray[iter]<minNum){
            //         minNum = minArray[iter];
            //         numInd = iter;
            //         minTri = finalTriangle;
            //     }
            // }
        }
        intermediateMap[triangle]=pq;
        cout<<endl;
        // if(minTri){
        //     pair <Trianglee*,Trianglee*> p= make_pair(triangle,minTri);
        //     ans[p] = minNum;
        //     minTri->mapped=1;
        // }
        
    }
    srcXmin = srcpointXmin;
    srcYmin = srcpointYmin;
    tarXmin = tarpointXmin;
    tarYmin = tarpointYmin;
    queue<Trianglee*> qTri;
    map<Trianglee*,pair<Trianglee*,double>> triangleeTracker; 
    //Value assignment
    for(auto triangle:SrcTriangulation){
        int flag =1;
        while(!intermediateMap[triangle].empty() && flag){
            pair<double,Trianglee*> p = intermediateMap[triangle].top();
            if(triangleeTracker.find(p.second)==triangleeTracker.end()){
                flag=0;
                triangleeTracker[p.second]=make_pair(triangle,p.first);
            }else{
                if(p.first<triangleeTracker[p.second].second){
                    qTri.push(triangleeTracker[p.second].first);
                    triangleeTracker[p.second]=make_pair(triangle,p.first);
                    flag=0;
                }else{
                    intermediateMap[triangle].pop();
                }
            }
        }
    }
    //queue handling
    while(!qTri.empty()){
        Trianglee* triangle = qTri.front();
        int flag=1;
        while(!intermediateMap[triangle].empty() && flag){
            pair<double,Trianglee*> p = intermediateMap[triangle].top();
            if(triangleeTracker.find(p.second)==triangleeTracker.end()){
                flag=0;
                triangleeTracker[p.second]=make_pair(triangle,p.first);
            }else{
                if(p.first<triangleeTracker[p.second].second){
                    qTri.push(triangleeTracker[p.second].first);
                    triangleeTracker[p.second]=make_pair(triangle,p.first);
                }else{
                    intermediateMap[triangle].pop();
                }
            }
        }
        
        qTri.pop();
    }

    //Triangle assignment
    for(auto triangle:SrcTriangulation){
        if(intermediateMap[triangle].empty()){
            continue;
        }
        pair<double,Trianglee*> p = intermediateMap[triangle].top();
        Trianglee* bestFitTri = findBestFitTriangle(triangle,p.second,srcpointXmin,srcpointYmin,tarpointXmin,tarpointYmin);
        pair<Trianglee*,Trianglee*> p_ = make_pair(triangle,p.second);
        ans[p_]=p.first;
    }
    //return ans;
    return ans;
}

// Eigen::Matrix3d computeRotationMatrix(vector<Trianglee*>&srcTriangles,vector<Trianglee*>&targetTriangles,map<pair<Trianglee*,Trianglee*>,double> SrcToTarMapping){
//     Eigen::MatrixXd centroidMatrix(SrcToTarMapping.size(),3);
//     int rowId=0;
//     for(auto mapElement:SrcToTarMapping){
//         Trianglee* src = mapElement.first.first ;
//         Trianglee* tar = mapElement.first.second;
//         Eigen::Vector3d srcCent = Eigen::Vector3d::Zero();
//         Eigen::Vector3d tarCent = Eigen::Vector3d::Zero();
//         for(int iter=0;iter<3;iter++){
//             srcCent+= Eigen::Vector3d(src->points[iter]->x,src->points[iter]->y,0.0);
//             tarCent+= Eigen::Vector3d(tar->points[iter]->x,tar->points[iter]->y,0.0);
//         }
//         srcCent/=3.0;
//         tarCent/=3.0;
//         centroidMatrix.row(rowId) = tarCent - srcCent;
//     }
//     Eigen::JacobiSVD<Eigen::MatrixXd>svd(centroidMatrix,Eigen::ComputeFullU | Eigen::ComputeFullV);
//     double detU = std::abs(svd.matrixU().determinant());
//     int rows = svd.matrixU().rows();
//     int cols = svd.matrixU().cols();
//     cout<<"rows"<<rows;
//     cout<<endl<<"cols"<<cols;
//     cout<<endl<<endl;
//     cout<<"rows"<<svd.matrixV().rows();
//     cout<<endl<<"cols"<<svd.matrixV().cols();
//     cout<<endl<<endl;
//     Eigen::DiagonalMatrix<double,svd.matrixU().cols()> diag_matrix(detU,detU,detU);
//     // Eigen::DiagonalMatrix<double> diag_matrix(Eigen::Vector3d::Constant (detU));
//     Eigen::Matrix3d rotMatrix = svd.matrixV() * diag_matrix * svd.matrixV().transpose() * svd.matrixU();
//     // Eigen::Matrix3d rotMatrix(SrcToTarMapping.size());
//     if(svd.matrixU().determinant()<0){
//         rotMatrix.col(2)*=-1;
//     }
//     return rotMatrix;
// }

// std::vector<Trianglee*> interpolateShapes(map<pair<Trianglee*,Trianglee*>,double> &SrcToTarMapping, double t) {
//   // Convert source and target shapes to matrices
//   SparseMatrix<double> matrix_source, matrix_target;
// //   for (const Trianglee* triangle : source) {
// //     matrix_source += triangle->toMatrix();
// //   }
// //   for (const Trianglee* triangle : target) {
// //     matrix_target += triangle->toMatrix();
// //   }
//     for(auto mapElement:SrcToTarMapping){
//         Trianglee* src = mapElement.first.first ;
//         Trianglee* tar = mapElement.first.second;
//             matrix_source += src->toMatrix();
//             matrix_target += tar->toMatrix();
//         }
   
 
//   // Perform SVD on the source matrix
//   EigenSolver<SparseMatrix<double>> solver(matrix_source);
//   EigenSolver<SparseMatrix<double>> solver_target(matrix_target);
//   Eigen::MatrixXd U = solver.compute().eigenvectorsLeft();
//   Eigen::VectorXd Sigma = solver.compute().eigenvalues().real();
//   Eigen::MatrixXd Vt = solver.compute().eigenvectorsRight();
// //   Eigen::VectorXd Sigma_target = solver_target.compute().eigenvalues().real();
// //   // Interpolate singular values
// //   Eigen::VectorXd Sigma_t = (1 - t) * Sigma + t * Sigma_target;
 
// //   // Construct intermediate shape matrix
// //   Eigen::MatrixXd A_t = U * Sigma_t.asDiagonal() * Vt.transpose();
 
// //   // Extract vertex coordinates from intermediate matrix
// //   std::vector<Point2dim*> intermediate_points;
// //   for (int i = 0; i < 3; ++i) {
// //     intermediate_points.push_back(new Point2dim(A_t(i, 0), A_t(i, 1)));
// //   }
 
//   // Reconstruct intermediate Trianglee objects
//   std::vector<Trianglee*> intermediate_triangles;
// //   for (int i = 0; i < SrcToTarMapping.size(); ++i) {
// //     intermediate_triangles.push_back(new Trianglee(intermediate_points[3 * i], intermediate_points[3 * i + 1], intermediate_points[3 * i + 2]));
// //   }
 
//   return intermediate_triangles;
// }

std::vector<Trianglee*> interpolateShapes(map<pair<Trianglee*,Trianglee*>,double> &SrcToTarMapping, double t,double &srcXmin, double &srcYmin, double &tarXmin, double &tarYmin) {
    map<Point2dim*,vector<Point2dim*>> PointMap;
    map<Point2dim*,vector<Trianglee*>> PointTriangleMap;
    //Creating pre computed map
    for(auto mapElement:SrcToTarMapping){
        Trianglee* triSrc = mapElement.first.first;
        Trianglee* triFinal = mapElement.first.second;
        for(int ind=0;ind<3;ind++){
            Point2dim* pt = triSrc->points[ind];
            if(PointMap.find(pt)==PointMap.end()){
                vector<Trianglee*> triArray;
                vector<Point2dim*> PointArray;
                triArray.push_back(triSrc);
                PointArray.push_back(triFinal->points[ind]);
                PointMap[pt] = PointArray;
                PointTriangleMap[pt] =triArray;
            }else{
                PointMap[pt].push_back(triFinal->points[ind]);
                PointTriangleMap[pt].push_back(triSrc);
            }
        }
    }
    map<Point2dim*,Point2dim*> newPointMap;

    //Conversion of Points
    for(auto ptElement : PointMap){
        Point2dim* pt = ptElement.first;
        int len = ptElement.second.size();
        // Point2dim* newPt = new Point2dim(0.0,0.0);
        double xVal=0;
        double yVal=0;
        for(int iter=0;iter<len;iter++){
            //cout<<"val"<<" "<<xVal<<" ";
            Point2dim* ptFinal = ptElement.second[iter];
            xVal+=(((ptFinal->x))/len);
            yVal+=(((ptFinal->y))/len);
            //cout<<"val"<<" "<<ptFinal->x<<" ";
        }
        xVal = (xVal+pt->x)/2;
        yVal = (yVal+pt->y)/2;
        Point2dim* newPt = new Point2dim(int(xVal*100)/100.0,int(yVal*100)/100.0);
        //cout<<"val"<<" "<<xVal<<endl;
        newPointMap[pt] = newPt;
    }

    //Assigning values to points
    for(auto ptElement : PointTriangleMap){
        int len = ptElement.second.size();
        for(int iter=0;iter<len;iter++){
            Trianglee* tri = PointTriangleMap[ptElement.first][iter];
            for(int ind=0;ind<3;ind++){
                if(ptElement.first==tri->points[ind]){
                    tri->points[ind]=newPointMap[ptElement.first];
                    break;
                }
            }
        }
    }
    vector<Trianglee*> res;
    for(auto mapElement: SrcToTarMapping){
        res.push_back(mapElement.first.first);
    }
    return res;
}


double calculateTriangleARAPEnergy(Trianglee* tri){
    // Point2dim* a = tri->points[0];
    // Point2dim* b = tri->points[1];
    // Point2dim* c = tri->points[2];

    // Eigen::Vector2d ab(b->x-a->x,b->x-a->x);
    // Eigen::Vector2d ac(c->x-a->x,c->x-a->x);

    // Eigen::Vector2d cross_product = ab.cross(ac);

    // return 0.5*cross_product.norm();
    double val=0;
    double abx=tri->points[1]->x-tri->points[0]->x;
    double acx=tri->points[2]->x-tri->points[0]->x;
    double aby=tri->points[1]->y-tri->points[0]->y;
    double acy=tri->points[2]->y-tri->points[0]->y;
    // double bax=tri->points[0]->x-tri->points[1]->x;
    // double bcx=tri->points[2]->x-tri->points[1]->x;
    // double bay=tri->points[0]->y-tri->points[1]->y;
    // double bcy=tri->points[2]->y-tri->points[1]->y;
    // double cbx=tri->points[1]->x-tri->points[2]->x;
    // double cax=tri->points[0]->x-tri->points[2]->x;
    // double cby=tri->points[1]->y-tri->points[2]->y;
    // double cay=tri->points[0]->y-tri->points[2]->y;
    val+=(abx*acy)-(aby*acx);
    return 0.5*abs(val);
    // val+=(bax*bcy)-(bay*bcy);
    // val+=(cax*)
}
int main(){

    //Reading the image
    cv::Mat3b img = imread("../src6.png");
    Mat image = imread("../src6.png");
    cv::Mat3b tarimg = imread("../tar6.png");
    Mat imageTar = imread("../tar6.png");
    //Preprocessing steps for point extarction
    imagePreProcessing(img,5,4);
    imagePreProcessing(tarimg,5,4);
    cout<<"displaying delaunay triangulation"<<endl;
    //Delanauy Triangulation
    std::vector<Point2dim*> points;
    std::vector<Point2dim*> Tarpoints;
    //Extracting Points from image
    getPointform(img.data,img.cols, img.rows,points);
    getPointform(tarimg.data,tarimg.cols, tarimg.rows,Tarpoints);
    // for(auto point:points){
    //     cout<<point->x<<" "<<point->y<<endl;
        
    // }
    // for(int i=0;i<100;i++){
    //     cout<<"Src"<<points[i]->x<<" "<<points[i]->y<<endl;
    //     cout<<"Tar"<<Tarpoints[i]->x<<" "<<Tarpoints[i]->y<<endl;
    // }
    
    //time_t starting,ending;
    std::vector<Trianglee*> Srctriangulation;
    std::vector<Trianglee*> Tartriangulation;
    clock_t start= clock();
    //time(&starting);
    cout<<"Bowser Watson in Progress..."<<endl;
    bowyer_watson(points,Srctriangulation,img);
    for(int i=0;i<10;i++){
        cout<<Srctriangulation[i]->points[0]->x<<endl;
    }
    bowyer_watson(Tarpoints,Tartriangulation,img);
    //time(&ending);
    
    //Does Euclidean Mapping
    map<pair<Trianglee*,Trianglee*>,double> SrcTarmapping=mapping(Srctriangulation,Tartriangulation);
    
    double endTime = (double)(clock()-start)/CLOCKS_PER_SEC;
    //cout<<"Bowser Watson Time Elapsed"<<difftime(ending,starting)<<endl;
    cout<<"Bowser Watson Time Elapsed"<<endTime<<endl;
    // tryDrawingTriangles(image,  triangulation,5);
    // tryDrawingTriangles(image,  triangulation,10);
    // tryDrawingTriangles(image,  triangulation,50);
    // tryDrawingTriangles(image,  triangulation,100);
    // tryDrawingTriangles(image,  triangulation,200);
    // tryDrawingTriangles(image,  triangulation,500);
    // tryDrawingTriangles(image,  triangulation,1000);
    tryDrawingTrianglesV(image.clone(),  Srctriangulation,10);
    tryDrawingTrianglesV(imageTar.clone(),  Tartriangulation,10);
    tryDrawingTriangles(image.clone(),  SrcTarmapping,12);
    tryDrawingTrianglesV(image.clone(),  Srctriangulation,10000);
    tryDrawingTrianglesV(imageTar.clone(),  Tartriangulation,10000);
    tryDrawingTriangles(image.clone(),  SrcTarmapping,10000);
    double SrcXMin = 0;
    double TarXMin = 0;
    double SrcYMin = 0;
    double TarYMin = 0;

    //Does Centroid Mapping
    map<pair<Trianglee*,Trianglee*>,double> CentSrcTarmapping=centroidMapping(Srctriangulation,Tartriangulation,10000, SrcXMin, SrcYMin, TarXMin,TarYMin);
    tryDrawingTriangles(image.clone(),  CentSrcTarmapping,10000);
    // tryDrawingTrianglesV(image,  Srctriangulation ,10000);
    
    
    //Eigen::Matrix3d rot = computeRotationMatrix(Srctriangulation, Tartriangulation, CentSrcTarmapping);
    std::vector<Trianglee*> intermediate_shapes ;
    //tryDrawingTriangles(image,  CentSrcTarmapping,10000);
    
    // tryDrawingTrianglesV(image,  intermediate_shapes ,10000);
    
    int count=0;
    cout<<"Check2"<<endl;
    for(int i=0;i<Srctriangulation.size();i++){
        if(Srctriangulation[i]->points[0]->x>10){
            count++;
            cout<<Srctriangulation[i]->points[0]->x<<endl;
        }
            
    }
    cout<<"src"<<SrcXMin<<" ";
    cout<<"tar"<<TarXMin<<" ";
    for(int iter=0;iter<5;iter++){
        std::set<Point2dim*> intPoints;
        std::vector<Point2dim*> resPoints;
        //Carries out interpolation
        intermediate_shapes = interpolateShapes(CentSrcTarmapping,0.5,SrcXMin,SrcYMin,TarXMin,TarYMin);
        // cv::imshow("seemeYou", img);
        // cv::waitKey(0);
        // cv::imshow("seemeYou2", image.clone());
        // cv::waitKey(0);
        // cv::imshow("blank", out);
        
        // tryDrawingTrianglesV(image.clone(), ftri ,360);
        // tryDrawingTrianglesV(image.clone(), Srctriangulation ,360);

        // for(auto tri1:intermediate_shapes){
        //     intPoints.insert(tri1->points[0]);
        //     intPoints.insert(tri1->points[1]);
        //     intPoints.insert(tri1->points[2]);
        // }

        for(auto tri1: CentSrcTarmapping){
            intPoints.insert(tri1.first.first->points[0]);
            intPoints.insert(tri1.first.first->points[1]);
            intPoints.insert(tri1.first.first->points[2]);
        }
        intPoints.insert(new Point2dim(SrcXMin+(iter/4.0),TarXMin));
        cout<<"size array"<<intermediate_shapes.size()<<endl;
        for(auto tri1:intPoints){
            resPoints.push_back(tri1);
            
        }
        Srctriangulation = intermediate_shapes;
        //tryDrawingTrianglesV(image.clone(),  Srctriangulation,1000);
        //cout<<"hi1";
        bowyer_watson(resPoints,Srctriangulation,img);
        //tryDrawingTrianglesV(image.clone(),  Srctriangulation,1000);
        //cout<<"hi2";
        map<pair<Trianglee*,Trianglee*>,double> CentSrcTarmapping=centroidMapping(Srctriangulation,Tartriangulation,1000, SrcXMin, SrcYMin, TarXMin,TarYMin);
        //cout<<"hi3";
        //cout<<"size array"<<intermediate_shapes.size()<<endl;
        cout<<"size"<<CentSrcTarmapping.size()<<endl;
        tryDrawingTriangles(image.clone(),  CentSrcTarmapping,iter+1,10000);
        tryDrawingTriangles(imageTar.clone(), CentSrcTarmapping,iter+1,10000);
    }
        
    
    
    // cout<<count<<endl;
    
    return 0;
}

