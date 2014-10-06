//
//  RobustTextDetection.h
//  RobustTextDetection
//
//  Created by Saburo Okita on 08/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __RobustTextDetection__RobustTextDetection__
#define __RobustTextDetection__RobustTextDetection__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * Parameters for robust text detection, quite a handful
 */
struct RobustTextParam {
    int minMSERArea         = 10;
    int maxMSERArea         = 2000;
    int cannyThresh1        = 20;
    int cannyThresh2        = 100;
    
    int maxConnCompCount     = 3000;
    int minConnCompArea      = 75;
    int maxConnCompArea      = 600;
    
    float minEccentricity    = 0.1;
    float maxEccentricity    = 0.995;
    float minSolidity        = 0.4;
    float maxStdDevMeanRatio = 0.5;
};


/**
 * Implementation of Chen, Huizhong, et al. "Robust Text Detection in Natural Images with Edge-Enhanced Maximally Stable Extremal
 * Regions." Image Processing (ICIP), 2011 18th IEEE International Conference on. IEEE, 2011.
 * 
 * http://www.stanford.edu/~hchen2/papers/ICIP2011_RobustTextDetection.pdf
 * http://www.mathworks.de/de/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html#zmw57dd0e829
 */
class RobustTextDetection {
public:
    RobustTextDetection( string temp_img_directory = "" );
    RobustTextDetection( RobustTextParam& param, string temp_img_directory = "" );
    
    pair<Mat, Rect> apply( Mat& image );
    
protected:
    Mat preprocessImage( Mat& image );
    Mat computeStrokeWidth( Mat& dist ) ;
    Mat createMSERMask( Mat& grey );
    
    static int toBin( const float angle, const int neighbors = 8 );
    Mat growEdges(Mat& image, Mat& edges );
    
    vector<Point> convertToCoords( int x, int y, bitset<8> neighbors ) ;
    vector<Point> convertToCoords( Point& coord, bitset<8> neighbors ) ;
    vector<Point> convertToCoords( Point& coord, uchar neighbors ) ;
    bitset<8> getNeighborsLessThan( int * curr_ptr, int x, int * prev_ptr, int * next_ptr ) ;
    
    Rect clamp( Rect& rect, Size size );
    
private:
    string tempImageDirectory;
    RobustTextParam param;
};

#endif /* defined(__RobustTextDetection__RobustTextDetection__) */
