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


void ofxFeatureFinderObject::save(string filename)
{
    cv::FileStorage fs;
    fs.open(ofToDataPath(filename), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        cout << "ERROR OPENING FILE ";
        return;
    }
    cout << "now saving " << keypoints.size() << " keypoints..." << endl;

    fs << "keypoints" << "[";
    
    for(unsigned int j=0; j<keypoints.size(); ++j)
    {
        cv::KeyPoint kp = keypoints.at(j);
        
        fs << "[" <<
        kp.angle <<
        kp.class_id <<
        kp.octave <<
        kp.pt.x <<
        kp.pt.y <<
        kp.response <<
        kp.size <<
        "]";
    }
    
    fs << "]";

    cout << "now saving descriptors..." << endl;
    fs << "descriptors" << descriptors;

    fs.release();
    cout << "saving complete" << endl;
    
}


void ofxFeatureFinderObject::load(string filename)
{
/*
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptors;
    
    int nKpts;

    streamPtr >> nKpts;
    for(int i=0;i<nKpts;++i)
    {
        cv::KeyPoint kpt;
        streamPtr >>
        kpt.angle >>
        kpt.class_id >>
        kpt.octave >>
        kpt.pt.x >>
        kpt.pt.y >>
        kpt.response >>
        kpt.size;
        kpts.push_back(kpt);
    }
    
    int rows, cols, type;
    int dataSize;

    streamPtr >> rows >> cols >> type >> dataSize;
*/
//    streamPtr >> data;
//    descriptors = cv::Mat(rows, cols, type, data.data()).clone();
    
//    this->setData(kpts, descriptors, cv::Mat(), detectorType, descriptorType);

    //    cvImage_ = cvtQImage2CvMat(pixmap_.toImage());
    //this->setMinimumSize(image_.size());
}
