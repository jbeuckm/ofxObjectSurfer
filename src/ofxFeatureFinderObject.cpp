//
//  ObjectFeatureDescription.cpp
//  ofxKinectOsc
//
//  Created by joe on 12/21/13.
//
//

#include "ofxFeatureFinderObject.h"


ofxFeatureFinderObject::ofxFeatureFinderObject(std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors) {

    outlines = _outlines;
    keypoints = _keypoints;
    descriptors = _descriptors;
    
}


void ofxFeatureFinderObject::save(std::ostream & streamPtr) const
{
    /*
    streamPtr << id_ << detectorType_ << descriptorType_;
    streamPtr << (int)keypoints_.size();
    for(unsigned int j=0; j<keypoints_.size(); ++j)
    {
        streamPtr << keypoints_.at(j).angle <<
        keypoints_.at(j).class_id <<
        keypoints_.at(j).octave <<
        keypoints_.at(j).pt.x <<
        keypoints_.at(j).pt.y <<
        keypoints_.at(j).response <<
        keypoints_.at(j).size;
    }
    
    qint64 dataSize = descriptors_.elemSize()*descriptors_.cols*descriptors_.rows;
    streamPtr << descriptors_.rows <<
    descriptors_.cols <<
    descriptors_.type() <<
    dataSize;
    streamPtr << QByteArray((char*)descriptors_.data, dataSize);
    streamPtr << pixmap_;
     */
}

void ofxFeatureFinderObject::load(std::istream & streamPtr) const
{
    /*
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptors;
    
    int nKpts;
    QString detectorType, descriptorType;
    streamPtr >> id_ >> detectorType >> descriptorType >> nKpts;
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
    
    int rows,cols,type;
    qint64 dataSize;
    streamPtr >> rows >> cols >> type >> dataSize;
    QByteArray data;
    streamPtr >> data;
    descriptors = cv::Mat(rows, cols, type, data.data()).clone();
    streamPtr >> pixmap_;
    this->setData(kpts, descriptors, cv::Mat(), detectorType, descriptorType);
    cvImage_ = cvtQImage2CvMat(pixmap_.toImage());
    //this->setMinimumSize(image_.size());
     */
}
