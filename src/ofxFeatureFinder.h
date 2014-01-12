//
//  ofxFeatureFinder.h
//
//  Created by joe on 12/18/13.
//
//

#ifndef OFXFEATUREFINDER
#define OFXFEATUREFINDER

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d.hpp> // for homography

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxFeatureFinderObject.h"


class ofxFeatureFinder {

private:
    cv::FeatureDetector * detector;
    cv::DescriptorExtractor * extractor;

    
    ofRectangle displayRect;
    ofRectangle cropRect;

	ofxCvColorImage rawImage;

    ofxCvGrayscaleImage processImage;
    cv::Mat processImageMat;
    
    std::vector<ofPolyline> regions;
    bool bDrawingRegion;

    std::vector<cv::KeyPoint> imageKeypoints;
    cv::Mat imageDescriptors;
    
    std::vector<ofxFeatureFinderObject> objects;
    std::vector<ofxFeatureFinderObject> detectedObjects;
    std::vector<cv::Mat> detectedHomographies;
    
    bool detectObject(ofxFeatureFinderObject object, cv::Mat &homography);
    
    double 	hessianThreshold;
    int 	octaves;
    int 	octaveLayers;

    bool bBlur;
    int blurLevel;

public:
    ofxFeatureFinder();
    ~ofxFeatureFinder();

    CvSeq *getImageKeypoints();
    
    void setDisplayRect(int x, int y, int width, int height);
    void setCropRect(int x, int y, int width, int height);
    
    void updateSourceImage(ofxCvColorImage image);
    
    void findKeypoints();
    
    void draw();
    void drawImage();
    void drawFeatures();
    void drawRegions();
    void drawDetected();
    
    void setBlurLevel(int _blurLevel);
    
    ofxFeatureFinderObject createObject();
    void saveObject(ofxFeatureFinderObject object, string filepath);
    void createAndSaveObject(string filepath);

    void loadObject(string filepath);
    void loadObjectsInFolder(string folder);
    
    void clearRegions();

    vector<string> detectObjects();

    void mouseMoved(ofMouseEventArgs &args);
    void mousePressed(ofMouseEventArgs &args);
    void mouseDragged(ofMouseEventArgs &args);
    void mouseReleased(ofMouseEventArgs &args);
    

};

#endif /* defined(OFXFEATUREFINDER) */
