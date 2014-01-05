//
//  ObjectFeatureDescription.cpp
//  ofxKinectOsc
//
//  Created by joe on 12/21/13.
//
//

#include "ofxFeatureFinderObject.h"

ofxFeatureFinderObject::ofxFeatureFinderObject() {//std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors);
    outlines = std::vector<ofPolyline>();
    keypoints = std::vector<cv::KeyPoint>();
    descriptors = cv::Mat();
}

ofxFeatureFinderObject::ofxFeatureFinderObject(std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors) {

    outlines = _outlines;
    keypoints = _keypoints;
    descriptors = _descriptors;
    
}


void ofxFeatureFinderObject::save(string filepath)
{
    cv::FileStorage fs;
    fs.open(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        cout << "ERROR OPENING FILE TO WRITE" << endl;
        return;
    }
    cout << "now saving " << keypoints.size() << " keypoints..." << endl;

    fs << "keypoints" << "[";
    
    for(unsigned int j=0; j<keypoints.size(); ++j)
    {
        cv::KeyPoint kp = keypoints.at(j);
        
        fs << "{" <<
        "angle" << kp.angle <<
        "class_id" << kp.class_id <<
        "octave" << kp.octave <<
        "x" << kp.pt.x <<
        "y" << kp.pt.y <<
        "response" << kp.response <<
        "size" << kp.size <<
        "}";
    }
    
    fs << "]";

    cout << "now saving descriptors..." << endl;
    fs << "descriptors" << descriptors;

    fs.release();
    cout << "saving complete" << endl;
    
}


void ofxFeatureFinderObject::load(string filepath)
{
    cv::FileStorage fs;
    fs.open(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "ERROR OPENING FILE TO READ" << endl;
        return;
    }

    cv::FileNode keypointsList = fs["keypoints"];
    cv::FileNodeIterator it = keypointsList.begin(), it_end = keypointsList.end();
    int idx = 0;
    std::vector<uchar> lbpval;
    
    for( ; it != it_end; ++it, idx++ )
    {
        cv::KeyPoint kp;

        kp.angle = (*it)["angle"];
        kp.class_id = (*it)["class_id"];
        kp.octave = (*it)["octave"];
        kp.pt.x = (*it)["x"];
        kp.pt.y = (*it)["y"];
        kp.response = (*it)["response"];
        kp.size = (*it)["size"];
        
        keypoints.push_back(kp);
    }
    cout << "loaded " << keypoints.size() << " keypoints." << endl;
    
    cv::FileNode descriptorsNode = fs["descriptors"];
    descriptorsNode >> descriptors;

    cout << "loaded descriptors." << endl;

    fs.release();
    cout << "ofxFeatureFinderObject::load complete" << endl;
}


