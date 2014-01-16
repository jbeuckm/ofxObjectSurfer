#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxUI.h"


#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d.hpp> // for homography

#include "ofxFeatureFinder.h"


#define HOST "localhost"
#define PORT 12345

#define PROCESS_WIDTH 640
#define PROCESS_HEIGHT 480

#define DISPLAY_WIDTH 640
#define DISPLAY_HEIGHT 480

using namespace cv;

class testApp : public ofBaseApp {
public:
	
	void setup();
	void update();
	void draw();
	void exit();
	
	void keyPressed(int key);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void windowResized(int w, int h);

    
private:

    bool bBlur;
    int blurLevel;
    
    ofVideoGrabber vidGrabber;
    
    vector<string> foundObjectLabels;

	ofxCvColorImage rawImage;

    void openSampleImage();
    void processRawImage();

    ofxFeatureFinder featureFinder;
    bool bDetectObjects;
    
    bool bPaused;
	
    ofxUICanvas *gui;

    void setupGui();
    
    void guiEvent(ofxUIEventArgs &e);
    

    int snapCounter;
    
};
