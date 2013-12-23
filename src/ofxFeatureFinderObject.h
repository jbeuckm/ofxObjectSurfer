//
//  ObjectFeatureDescription.h
//  ofxKinectOsc
//
//  Created by joe on 12/21/13.
//
//

#ifndef __ofxKinectOsc__ObjectFeatureDescription__
#define __ofxKinectOsc__ObjectFeatureDescription__

#include "ofMain.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d.hpp> // for homography

#include <iostream>


class ofxFeatureFinderObject {

public:
    
    ofxFeatureFinderObject(std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors);
    
    std::vector<ofPolyline> outlines;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    void save(std::ostream & streamPtr) const;
    void load(std::istream & streamPtr) const;

};

#endif /* defined(__ofxKinectOsc__ObjectFeatureDescription__) */
