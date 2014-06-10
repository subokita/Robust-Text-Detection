//
//  main.cpp
//  RobustTextDetection
//
//  Created by Saburo Okita on 05/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[])
{

    namedWindow( "" );
    moveWindow("", 0, 0);
    
    Mat image = imread( "/Users/saburookita/Personal Projects/RobustTextDetection/TestText.png" );
    
    RobustTextParam param;
    param.minMSERArea        = 10;
    param.maxMSERArea        = 2000;
    param.cannyThresh1       = 20;
    param.cannyThresh2       = 100;
    
    param.maxConnCompCount   = 3000;
    param.minConnCompArea    = 75;
    param.maxConnCompArea    = 600;
    
    param.minEccentricity    = 0.1;
    param.maxEccentricity    = 0.995;
    param.minSolidity        = 0.4;
    param.maxStdDevMeanRatio = 0.5;
    
    RobustTextDetection detector(param);
    pair<Mat, Rect> result = detector.apply( image );
    
    /* Get the region where the candidate text is */
    Mat stroke_width( result.second.height, result.second.width, CV_8UC1, Scalar(0) );
    Mat(result.first, result.second).copyTo( stroke_width);
    
    imshow("", stroke_width );
    waitKey();
    
    /* Use Tesseract to try to decipher our image */
    tesseract::TessBaseAPI tesseract_api;
    tesseract_api.Init(NULL, "eng"  );
    tesseract_api.SetImage((uchar*) stroke_width.data, stroke_width.cols, stroke_width.rows, 1, stroke_width.cols);
    
    char * out = tesseract_api.GetUTF8Text();
    cout << out << endl;
    
    return 0;
}
