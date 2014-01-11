//
//  ofxFeatureFinderObject.cpp
//
//  Created by joe on 12/21/13.
//
//

#include "ofxFeatureFinderObject.h"

ofxFeatureFinderObject::ofxFeatureFinderObject() {
    outlines = std::vector<ofPolyline>();
    keypoints = std::vector<cv::KeyPoint>();
    descriptors = cv::Mat();
}

ofxFeatureFinderObject::ofxFeatureFinderObject(std::vector<ofPolyline> _outlines, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors) {
    outlines = _outlines;
    keypoints = _keypoints;
    descriptors = _descriptors;
}


bool ofxFeatureFinderObject::save(string filepath)
{
    cv::FileStorage fs;
    fs.open(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        cout << "ERROR OPENING FILE TO WRITE" << endl;
        return false;
    }

    cout << "now saving " << outlines.size() << " outlines..." << endl;
    
    fs << "outlines" << "[";
    for(unsigned int j=0; j<outlines.size(); ++j)
    {
        vector<ofPoint> vertices = outlines.at(j).getVertices();
        
        fs << "[";
        for(unsigned int i=0; i<vertices.size(); ++i)
        {
            fs << "{" <<
            "x" << vertices.at(i).x <<
            "y" << vertices.at(i).y <<
            "}";
        }
        fs << "]";
        
    }
    fs << "]";
    
    
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
    cout << "ofxFeatureFinderObject::save complete" << endl;
    
    return true;
}


bool ofxFeatureFinderObject::load(string filepath)
{
    cv::FileStorage fs;
    fs.open(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "ERROR OPENING FILE TO READ" << endl;
        return false;
    }
    
    cv::FileNode outlinesList = fs["outlines"];
    cv::FileNodeIterator it = outlinesList.begin(), it_end = outlinesList.end();
    int idx = 0;
    
    for( ; it != it_end; ++it, idx++ )
    {
        ofPolyline line;
        
        for (int k=0; k<(*it).size(); k++) {
            ofPoint p = ofPoint((*it)[k]["x"], (*it)[k]["y"]);
            line.addVertex(p);
        }
        
        line.close();
        
        outlines.push_back(line);
    }
    cout << "loaded " << outlines.size() << " outlines." << endl;
    
    
    cv::FileNode keypointsList = fs["keypoints"];
    cv::FileNodeIterator it2 = keypointsList.begin(), it2_end = keypointsList.end();
    idx = 0;

    for( ; it2 != it2_end; ++it2, idx++ )
    {
        cv::KeyPoint kp;
            
        kp.angle = (*it2)["angle"];
        kp.class_id = (*it2)["class_id"];
        kp.octave = (*it2)["octave"];
        kp.pt.x = (*it2)["x"];
        kp.pt.y = (*it2)["y"];
        kp.response = (*it2)["response"];
        kp.size = (*it2)["size"];
        
        keypoints.push_back(kp);
    }
    cout << "loaded " << keypoints.size() << " keypoints." << endl;
    
    cv::FileNode descriptorsNode = fs["descriptors"];
    descriptorsNode >> descriptors;

    cout << "loaded descriptors." << endl;

    fs.release();
    cout << "ofxFeatureFinderObject::load complete" << endl;
    
    return true;
}


