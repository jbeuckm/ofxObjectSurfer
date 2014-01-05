//
//  featureManager.cpp
//  ofxKinectOsc
//
//  Created by joe on 12/18/13.
//
//

#include "ofxFeatureFinder.h"

ofxFeatureFinder::ofxFeatureFinder() {
    
    ofRegisterMouseEvents(this);
    
    bDrawingRegion = false;
}

void ofxFeatureFinder::setFrame(int x, int y, int width, int height) {
    
    rect = ofRectangle(x, y, width, height);
    
}



void ofxFeatureFinder::clearRegions() {
    regions.clear();
}


void ofxFeatureFinder::findKeypoints(ofxCvGrayscaleImage _image) {
    
    image = _image;
    
    cv::Mat mat = cv::cvarrToMat(image.getCvImage());
    
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
        cv::KeyPoint keypt = imageKeypoints.at(i);
        
        for(it = regions.begin(); it != regions.end(); ++it){
            if ((*it).inside(keypt.pt.x, keypt.pt.y)) {
                
                selectedKeypoints.push_back(keypt);
                
                break;
            }
        }
    }
    
    if (selectedKeypoints.size() == 0) {
        return;
    }

    cv::Mat mat = cv::cvarrToMat(image.getCvImage());
    
    cv::DescriptorExtractor * extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(mat, selectedKeypoints, selectedDescriptors);
    delete extractor;
    
    ofxFeatureFinderObject object = ofxFeatureFinderObject(regions, selectedKeypoints, selectedDescriptors);
    objects.push_back(object);
    
    cout << "added object with " << selectedKeypoints.size() << " keypoints." << endl;
    
    object.save("object.yml");
    
    this->clearRegions();
}



void ofxFeatureFinder::detectObjects() {
    
    detectedObjects.clear();
    detectedHomographies.clear();

    int i = 0;
    for(; i<objects.size(); i++){
        ofxFeatureFinderObject object = objects.at(i);
        cv::Mat homography;
        if (this->detectObject(object, homography)) {
            cout << "detected object " << i << endl;
            detectedObjects.push_back(object);
            detectedHomographies.push_back(homography);
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
        homography = findHomography(mpts_1, mpts_2, cv::RANSAC, 1.0, outlier_mask);

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
        
        return true;
    }
    
    return false;
}



void ofxFeatureFinder::draw() {
    this->drawImage();
    
    this->drawFeatures();
    this->drawRegions();
    
    this->drawDetected();
}


void ofxFeatureFinder::drawImage() {
    image.draw(rect.x, rect.y, rect.width, rect.height);
}


void ofxFeatureFinder::drawDetected() {
    vector<ofxFeatureFinderObject>::iterator it = detectedObjects.begin();
    
    ofSetColor(0, 0, 255);

    ofPushMatrix();
    ofTranslate(rect.x, rect.y);

    for(int i=0; i < detectedObjects.size(); i++){
        
        ofxFeatureFinderObject object = detectedObjects.at(i);
        
        cv::Mat H = detectedHomographies.at(i);
        
        ofPushMatrix();
        ofMatrix4x4 transform = ofMatrix4x4(
                                            H.at<double>(0,0), H.at<double>(1,0), 0, H.at<double>(2,0),
                                            H.at<double>(0,1), H.at<double>(1,1), 0, H.at<double>(2,1),
                                            0, 0, 1, 0,
                                            H.at<double>(0,2), H.at<double>(1,2), 0, H.at<double>(2,2)
        );

        ofMultMatrix(transform);
        vector<ofPolyline>::iterator line = object.outlines.begin();
        for(; line != object.outlines.end(); ++line){
            (*line).draw();
        }
        
        ofPopMatrix();
    }
    
    ofPopMatrix();
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
    
    ofEnableAlphaBlending();
    ofFill();
    
    float cx = (float)rect.width / (float)image.width;
    float cy = (float)rect.height / (float)image.height;
    
    
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

