//
//  featureManager.cpp
//  ofxKinectOsc
//
//  Created by joe on 12/18/13.
//
//

#include "ofxFeatureFinder.h"

ofxFeatureFinder::ofxFeatureFinder() {
    
    storage = cvCreateMemStorage(0);
    
    ofRegisterMouseEvents(this);
    
    bDrawingRegion = false;
}

void ofxFeatureFinder::setFrame(int x, int y, int width, int height) {
    
    rect = ofRectangle(x, y, width, height);
    
}



void ofxFeatureFinder::clearRegions() {
    regions.clear();
}


void ofxFeatureFinder::findKeypoints(IplImage *_image) {
    
    image = _image;
    
    cv::Mat mat = cv::cvarrToMat(image);
    
    imageKeypoints.clear();
   
    cv::FeatureDetector * detector = new cv::SurfFeatureDetector();
    detector->detect(mat, imageKeypoints);
    delete detector;

    cv::DescriptorExtractor * extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(mat, imageKeypoints, imageDescriptors);
    delete extractor;
}

void ofxFeatureFinder::createObject() {
    
    std::vector<cv::KeyPoint> selectedKeypoints;
    cv::Mat selectedDescriptors;
    
    vector<ofPolyline>::iterator it;
    int i;
    // collect the selected keypoints
    for(i = 0; i < imageKeypoints.size(); i++ )
    {
        cv::KeyPoint r = imageKeypoints.at(i);
        
        for(it = regions.begin(); it != regions.end(); ++it){
            if ((*it).inside(r.pt.x, r.pt.y)) {
                
                selectedKeypoints.push_back(r);
                
                break;
            }
        }
    }
    
    if (selectedKeypoints.size() == 0) {
        return;
    }

    cv::Mat mat = cv::cvarrToMat(image);
    
    cv::DescriptorExtractor * extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(mat, selectedKeypoints, selectedDescriptors);
    delete extractor;
    
    ofxFeatureFinderObject object = ofxFeatureFinderObject(selectedKeypoints, selectedDescriptors);
    objects.push_back(object);
    
    cout << "added object with " << selectedKeypoints.size() << " keypoints.";
    
    this->clearRegions();
}



void ofxFeatureFinder::detectObjects() {

    int i = 0;
    for(; i<objects.size(); i++){
        ofxFeatureFinderObject object = objects.at(i);
        cv::Mat homography;
        if (this->detectObject(object, homography)) {
            cout << "detected object " << i << endl;
        }
    }
    
}

bool ofxFeatureFinder::detectObject(ofxFeatureFinderObject object, cv::Mat &homography) {
    ////////////////////////////
    // NEAREST NEIGHBOR MATCHING USING FLANN LIBRARY (included in OpenCV)
    //////////////////////////
    cv::Mat results;
    cv::Mat dists;
    int k=2; // find the 2 nearest neighbors

    // assume it is CV_32F
    // Create Flann KDTree index
    cv::flann::Index flannIndex(imageDescriptors, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);
    results = cv::Mat(object.descriptors.rows, k, CV_32SC1); // Results index
    dists = cv::Mat(object.descriptors.rows, k, CV_32FC1); // Distance results are CV_32FC1
        
    // search (nearest neighbor)
    flannIndex.knnSearch(object.descriptors, results, dists, k, cv::flann::SearchParams() );

    
    ////////////////////////////
    // PROCESS NEAREST NEIGHBOR RESULTS
    ////////////////////////////
    // Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
    float nndrRatio = 0.6;
    std::vector<cv::Point2f> mpts_1, mpts_2; // Used for homography
    std::vector<int> indexes_1, indexes_2; // Used for homography
    std::vector<uchar> outlier_mask;  // Used for homography
    for(unsigned int i=0; i<object.descriptors.rows; ++i)
    {
        // Check if this descriptor matches with those of the objects
        // Apply NNDR
        if(dists.at<float>(i,0) <= nndrRatio * dists.at<float>(i,1))
        {
            mpts_1.push_back(object.keypoints.at(i).pt);
            indexes_1.push_back(i);
            
            mpts_2.push_back(imageKeypoints.at(results.at<int>(i,0)).pt);
            indexes_2.push_back(results.at<int>(i,0));
        }
    }
    
    // FIND HOMOGRAPHY
    int nbMatches = 8;

    if(mpts_1.size() >= nbMatches)
    {
        homography = findHomography(mpts_1,
                                   mpts_2,
                                   cv::RANSAC,
                                   1.0,
                                   outlier_mask);

        uint inliers=0, outliers=0;
        for(unsigned int k=0; k<mpts_1.size();++k)
        {
            if(outlier_mask.at(k))
            {
                ++inliers;
            }
            else
            {
                ++outliers;
            }
        }

        cout << inliers << " in / " << outliers << " out" << endl;
        
        // Do what you want with the homography (like showing a rectangle)
        // The "outlier_mask" contains a mask representing the inliers and outliers.
        // ...
        return true;
    }
    
    return false;
}



void ofxFeatureFinder::draw() {
    this->drawFeatures();
    this->drawRegions();
}


void ofxFeatureFinder::drawRegions() {
    
    ofSetColor(0, 255, 0);
    
    vector<ofPolyline>::iterator it = regions.begin();
    
    ofPushMatrix();
    ofTranslate(rect.x, rect.y);
    for(; it != regions.end(); ++it){
        
        (*it).draw();
        
    }
    ofPopMatrix();
}


void ofxFeatureFinder::drawFeatures() {
    if (!imageKeypoints.size()) return;
    if (!image) return;
    
    ofEnableAlphaBlending();
    ofFill();
    
    float cx = (float)rect.width / (float)image->width;
    float cy = (float)rect.height / (float)image->height;
    
    
    ofPushMatrix();
    ofTranslate(rect.x, rect.y);
    ofScale(cx, cy);
    
    vector<ofPolyline>::iterator it;
    int i;
    //draw the keypoints on the captured frame
    for(i = 0; i < imageKeypoints.size(); i++ )
    {
        cv::KeyPoint r = imageKeypoints.at(i);
        
        ofSetColor(255, 255, 0, 127);

        for(it = regions.begin(); it != regions.end(); ++it){
            if ((*it).inside(r.pt.x, r.pt.y)) {
                ofSetColor(255, 0, 255, 127);
                break;
            }
        }

        ofCircle(cvRound(r.pt.x), cvRound(r.pt.y), 2);
    }
    
    ofPopMatrix();

    ofDisableAlphaBlending();
}





void ofxFeatureFinder::mouseMoved(ofMouseEventArgs &args){
    
}
void ofxFeatureFinder::mousePressed(ofMouseEventArgs &args){
    
    if (!rect.inside(args.x, args.y)) {
        return;
    }
    
    bDrawingRegion = true;
    
    ofPolyline line = ofPolyline();
    line.addVertex(ofVec2f(args.x - rect.x, args.y - rect.y));
    regions.push_back(line);
}
void ofxFeatureFinder::mouseDragged(ofMouseEventArgs &args){
    
    if (!bDrawingRegion) return;
    
    ofPolyline line = regions.back();
    line.addVertex(ofVec2f(args.x - rect.x, args.y - rect.y));
    regions.pop_back();
    regions.push_back(line);
}
void ofxFeatureFinder::mouseReleased(ofMouseEventArgs &args){
    
    if (!bDrawingRegion) return;
    
    ofPolyline line = regions.back();
    line.addVertex(ofVec2f(args.x - rect.x, args.y - rect.y));
    line.close();
    regions.pop_back();
    regions.push_back(line);

    bDrawingRegion = false;
}

