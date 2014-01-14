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
    
    bStretchContrast = true;
    bEqualizeHistogram = true;
    
    hessianThreshold = 800;
    octaves = 3;
    octaveLayers = 4;

    bBlur = false;
    blurLevel = 3;
    
    minMatchCount = 8;
    
    bDrawCircles = false;
    
    palette.push_back(ofColor(127, 127, 127));
    palette.push_back(ofColor(0, 0, 255));
    palette.push_back(ofColor(0, 255, 0));
    palette.push_back(ofColor(255, 255, 0));
    palette.push_back(ofColor(255, 127, 0));
    palette.push_back(ofColor(255, 0, 0));
    palette.push_back(ofColor(255, 255, 255));
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
    
    if (bStretchContrast) {
        processImage.contrastStretch();
    }

    processImageMat = cv::cvarrToMat(processImage.getCvImage());

    if (bEqualizeHistogram) {
        cv::Mat temp;
        equalizeHist( processImageMat, temp );
        processImageMat = temp;
    }

    if (bBlur) {
        cv::Mat temp;
        cv::GaussianBlur(processImageMat, temp, cv::Size(blurLevel, blurLevel), 0);
        processImageMat = temp;
    }
    
    processImage.setFromPixels(processImageMat.data, processImage.width, processImage.height);
}


void ofxFeatureFinder::setBlurLevel(unsigned int level) {
    if (level % 2 != 1) {
        level += 1;
    }
    blurLevel = level;
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

    extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(processImageMat, selectedKeypoints, selectedDescriptors);
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

    if(mpts_1.size() >= minMatchCount)
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
    this->drawLegend();

    ofPushMatrix();
    ofTranslate(cropRect.x, cropRect.y);

    processImage.draw(0, 0, cropRect.width, cropRect.height);
    
    this->drawFeatures();
    this->drawRegions();
    
    this->drawDetected();
    
    ofPopMatrix();

    ofPopMatrix();
}


void ofxFeatureFinder::drawLegend() {
    float yDiv = displayRect.height / 7.0;
    
    ofFill();
    for (int i=0; i<7; i++) {
        ofSetColor(palette.at(6-i));
        ofRect(displayRect.width - 10, i*yDiv, 0, 10, yDiv);
    }
    ofNoFill();
}


void ofxFeatureFinder::drawDetected() {
    vector<ofxFeatureFinderObject>::iterator it = detectedObjects.begin();
    
    for(int i=0; i < detectedObjects.size(); i++){
        
        ofxFeatureFinderObject object = detectedObjects.at(i);
        
        cv::Mat H = detectedHomographies.at(i);
        
        ofSetLineWidth(2);
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
    
    ofSetLineWidth(.5);
    ofEnableAlphaBlending();
    ofNoFill();
    
    vector<ofPolyline>::iterator it;
    
    float minResponse = FLT_MAX;
    float maxResponse = 0.0;

    for(int i = 0; i < imageKeypoints.size(); i++ )
    {
        cv::KeyPoint r = imageKeypoints.at(i);
        
        if (r.response > maxResponse) {
            maxResponse = r.response;
        }
        else if (r.response < minResponse) {
            minResponse = r.response;
        }
    }
    
    float responseRange = maxResponse - minResponse;
    
    for(int i = 0; i < imageKeypoints.size(); i++ )
    {
        cv::KeyPoint r = imageKeypoints.at(i);
        
        float responseRank = (r.response - minResponse) / responseRange;
        
        int colorIndex = floor(responseRank * 7.0);
        if (colorIndex > 6) colorIndex = 6;
        else if (colorIndex < 0) colorIndex = 0;
        ofSetColor(palette.at(colorIndex));

        for(it = regions.begin(); it != regions.end(); ++it){
            if ((*it).inside(r.pt.x, r.pt.y)) {
                ofSetColor(255, 0, 255, 127);
                break;
            }
        }
        
        if (bDrawCircles) {
            float radius = r.size/2.0;
            
            ofCircle(r.pt.x, r.pt.y, radius);
            ofLine(r.pt.x, r.pt.y, r.pt.x + radius*cos(r.angle), r.pt.y + radius*sin(r.angle));
        }
        
        ofCircle(r.pt.x, r.pt.y, 1.5);
        
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



void ofxFeatureFinder::colorReduce(cv::Mat &image, int div) {
    
    int nl= image.rows; // number of lines
    int nc= image.cols * image.channels(); // total number of elements per line
    
    for (int j=0; j<nl; j++) {
        
        uchar* data= image.ptr<uchar>(j);
        
        for (int i=0; i<nc; i++) {
            
            // process each pixel ---------------------
            
            data[i]= data[i]/div*div + div/2;
            
            // end of pixel processing ----------------
            
        } // end of line
    }
}