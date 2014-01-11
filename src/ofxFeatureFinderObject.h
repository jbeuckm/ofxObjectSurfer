//
//  ofxFeatureFinderObject.h
//
//  Created by joe on 12/21/13.
//
//

#ifndef OFXFEATUREFINDER_OBJECT
#define OFXFEATUREFINDER_OBJECT

#include "ofMain.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d.hpp> // for homography


class ofxFeatureFinderObject {

public:
    
    ofxFeatureFinderObject();
    ofxFeatureFinderObject(std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors);
    
    std::vector<ofPolyline> outlines;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    bool save(string filepath);
    bool load(string filepath);

};

#endif /* defined(__ofxKinectOsc__ObjectFeatureDescription__) */
