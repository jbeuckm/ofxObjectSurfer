//
//  ofxFeatureFinder.cpp
//
//  Created by joe on 12/18/13.
//
//

#include "ofxFeatureFinder.h"

ofxFeatureFinder::ofxFeatureFinder() {
    
    ofRegisterMouseEvents(this);
    
    bDrawingRegion = false;

    hessianThreshold = 800;
    octaves = 3;
    octaveLayers = 4;

    bBlur = true;
    blurLevel = 3;
}


void ofxFeatureFinder::toggleBlur(bool _blur) {
    bBlur = _blur;
}

void ofxFeatureFinder::setBlurLevel(int _blurLevel) {
    blurLevel = _blurLevel;
}

void ofxFeatureFinder::setHessianThreshold(double _hessian) {
    hessianThreshold = _hessian;
}


ofxFeatureFinder::~ofxFeatureFinder() {
}

void ofxFeatureFinder::setDisplayRect(int x, int y, int width, int height) {
    
    displayRect = ofRectangle(x, y, width, height);
    
}

void ofxFeatureFinder::setCropRect(int x, int y, int width, int height) {
    
    cropRect = ofRectangle(x, y, width, height);

    processImage.allocate(width, height);
    rawImageCropped.allocate(width, height);
}



void ofxFeatureFinder::clearRegions() {
    regions.clear();
}


void ofxFeatureFinder::updateSourceImage(ofxCvGrayscaleImage image) {
    updateSourceImage(image);
}
void ofxFeatureFinder::updateSourceImage(ofxCvColorImage image) {

    rawImage = image;
    rawImage.setROI(cropRect);
    
    rawImageCropped.setFromPixels(rawImage.getRoiPixels(), cropRect.width, cropRect.height);
    
    rawImage.setROI(0, 0, rawImage.width, rawImage.height);

    processImage = rawImageCropped;
    
    processImage.contrastStretch();
    
    if (bBlur) {
        processImage.blurGaussian(blurLevel);
    }
    
    processImageMat = cv::cvarrToMat(processImage.getCvImage());
    
}



void ofxFeatureFinder::findKeypoints() {
    
    detector = new cv::SurfFeatureDetector(hessianThreshold, octaves, octaveLayers);
    detector->detect(processImageMat, imageKeypoints);
    delete detector;

    extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(processImageMat, imageKeypoints, imageDescriptors);
    delete extractor;
}


ofxFeatureFinderObject ofxFeatureFinder::createObject() {
    
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

    cv::Mat mat = cv::cvarrToMat(processImage.getCvImage());
    
    extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(mat, selectedKeypoints, selectedDescriptors);
    delete extractor;
    
    ofxFeatureFinderObject object = ofxFeatureFinderObject(regions, selectedKeypoints, selectedDescriptors);
    objects.push_back(object);
    
    cout << "added object with " << selectedKeypoints.size() << " keypoints." << endl;

    this->clearRegions();
    
    return object;
}

void ofxFeatureFinder::createAndSaveObject(string filepath) {
    ofxFeatureFinderObject object = createObject();
    saveObject(object, filepath);
}

void ofxFeatureFinder::saveObject(ofxFeatureFinderObject object, string filepath) {
    object.save(filepath);
}


void ofxFeatureFinder::loadObject(string filepath) {
    ofLogNotice("loadObject() "+filepath);

    ofxFeatureFinderObject object;
    if (object.load(filepath)) {
        objects.push_back(object);
    }
}


void ofxFeatureFinder::loadObjectsInFolder(string folder) {

    ofDirectory dir(folder);
    dir.allowExt("yml");
    dir.listDir();

    for(int i = 0; i < dir.numFiles(); i++){
        string fullPath = dir.getAbsolutePath() + "/" + dir.getName(i);
        cout << "loading object description " << fullPath << endl;
        loadObject(fullPath);
    }

}



vector<string> ofxFeatureFinder::detectObjects() {
    
    detectedObjects.clear();
    detectedHomographies.clear();
    
    vector<string> foundObjectLabels;

    int i = 0;
    for(; i<objects.size(); i++){
        
        ofxFeatureFinderObject object = objects.at(i);
        cv::Mat homography;

        if (this->detectObject(object, homography)) {
            cout << "detected object " << i << endl;
            detectedObjects.push_back(object);
            detectedHomographies.push_back(homography);

            foundObjectLabels.push_back(object.label);
        }
    }
    
    return foundObjectLabels;
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
        for(unsigned int k=0; k<mpts_1.size(); ++k)
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

    ofPushMatrix();
    ofTranslate(displayRect.x, displayRect.y);

    rawImage.draw(0, 0, rawImage.width, rawImage.height);
    

    ofPushMatrix();
    ofTranslate(cropRect.x, cropRect.y);

    processImage.draw(0, 0, cropRect.width, cropRect.height);
    
    this->drawFeatures();
    this->drawRegions();
    
    this->drawDetected();
    
    ofPopMatrix();

    ofPopMatrix();
}


void ofxFeatureFinder::drawDetected() {
    vector<ofxFeatureFinderObject>::iterator it = detectedObjects.begin();
    
    for(int i=0; i < detectedObjects.size(); i++){
        
        ofxFeatureFinderObject object = detectedObjects.at(i);
        
        cv::Mat H = detectedHomographies.at(i);
        

        ofSetColor(0, 255, 255);
        
        vector<ofPolyline>::iterator outline = object.outlines.begin();
        for(; outline != object.outlines.end(); ++outline){

            vector<cv::Point2f> objectPoints((*outline).size());
            vector<cv::Point2f> scenePoints((*outline).size());
        
            for (int i=0, l=(*outline).size(); i<l; i++) {
                ofPoint p = (*outline)[i];
                objectPoints[i] = cv::Point2f(p.x, p.y);
            }

            perspectiveTransform( objectPoints, scenePoints, H);

            ofPolyline sceneOutlines;
            for (int i=0, l=(*outline).size(); i<l; i++) {
                cv::Point2f p = scenePoints[i];
                sceneOutlines.addVertex(p.x, p.y);
            }
            sceneOutlines.close();
            sceneOutlines.draw();
        }

    }
    
}


void ofxFeatureFinder::drawRegions() {
    
    ofSetColor(0, 255, 0);
    
    vector<ofPolyline>::iterator it = regions.begin();
    
    for(; it != regions.end(); ++it){
        
        (*it).draw();
        
    }
}


void ofxFeatureFinder::drawFeatures() {
    if (!imageKeypoints.size()) return;
    
    ofEnableAlphaBlending();
    ofFill();
    
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

    ofDisableAlphaBlending();
}





void ofxFeatureFinder::mouseMoved(ofMouseEventArgs &args){
    
}
void ofxFeatureFinder::mousePressed(ofMouseEventArgs &args){
    
    if (!displayRect.inside(args.x, args.y)) {
        return;
    }
    
    bDrawingRegion = true;
    
    ofPolyline line = ofPolyline();
    line.addVertex(ofVec2f(args.x - displayRect.x - cropRect.x, args.y - displayRect.y - cropRect.y));
    regions.push_back(line);
}
void ofxFeatureFinder::mouseDragged(ofMouseEventArgs &args){
    
    if (!bDrawingRegion) return;
    
    ofPolyline line = regions.back();
    line.addVertex(ofVec2f(args.x - displayRect.x - cropRect.x, args.y - displayRect.y - cropRect.y));
    regions.pop_back();
    regions.push_back(line);
}
void ofxFeatureFinder::mouseReleased(ofMouseEventArgs &args){
    
    if (!bDrawingRegion) return;
    
    ofPolyline line = regions.back();
    line.addVertex(ofVec2f(args.x - displayRect.x - cropRect.x, args.y - displayRect.y - cropRect.y));
    line.close();
    regions.pop_back();
    regions.push_back(line);

    bDrawingRegion = false;
}

