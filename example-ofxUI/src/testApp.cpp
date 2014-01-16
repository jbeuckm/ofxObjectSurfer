
#include "testApp.h"

//--------------------------------------------------------------
void testApp::setup() {
	ofSetLogLevel(OF_LOG_VERBOSE);
    
    vidGrabber.setVerbose(true);
    vidGrabber.initGrabber(PROCESS_WIDTH,PROCESS_HEIGHT);

    rawImage.allocate(PROCESS_WIDTH, PROCESS_HEIGHT);
	
	ofSetFrameRate(30);
	
    snapCounter = 0;
    
    bPaused = false;
    
    featureFinder.setDisplayRect(320, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT);
    featureFinder.setCropRect(160, 120, 320, 240);
    bDetectObjects = false;
    
    featureFinder.loadObjectsInFolder("objects/active");
    
    this->setupGui();
}


void testApp::setupGui() {

    gui = new ofxUICanvas(0,0,320,700);
    
    gui->addWidgetDown(new ofxUILabel("testApp Workflow", OFX_UI_FONT_LARGE));
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "FULLSCREEN"));

    gui->addLabelButton("LOAD IMAGE", false);

    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "STRETCH CONTRAST"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "EQUALIZE HISTOGRAM"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "BLUR"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUISlider(304, 16, 0.0, 16.0, 3.0, "BLUR LEVEL"));
    gui->addSpacer(320, 1);

    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "DRAW CIRCLES"));
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "DETECT OBJECTS"));
    
    gui->addWidgetDown(new ofxUISlider(304, 16, 1.0, 20000.0, 800.0, "MIN HESSIAN"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUISlider(304, 16, 1.0, 8.0, 3.0, "OCTAVES"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUISlider(304, 16, 4.0, 20.0, 8.0, "MIN MATCHES"));
    gui->addSpacer(320, 1);
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "PAUSE"));

    gui->addLabelButton("CLEAR REGIONS", false);
    
    gui->addLabelButton("SAVE DESCRIPTORS", false);
    gui->addLabelButton("LOAD DESCRIPTORS", false);
    
    ofAddListener(gui->newGUIEvent, this, &testApp::guiEvent);
    
    gui->loadSettings("GUI/guiSettings.xml");
    
}




//--------------------------------------------------------------
void testApp::update() {
    
    if (bPaused) return;
    
    vidGrabber.update();
        
    if(vidGrabber.isFrameNew()) {
        rawImage.setFromPixels(vidGrabber.getPixels(), PROCESS_WIDTH, PROCESS_HEIGHT);
        this->processRawImage();
    }

}


void testApp::processRawImage() {

    featureFinder.updateSourceImage(rawImage);

    featureFinder.findKeypoints();

    if (bDetectObjects) {
        foundObjectLabels = featureFinder.detectObjects();
    }

    // update the cv images
    rawImage.flagImageChanged();
    
}



//--------------------------------------------------------------
void testApp::draw() {
	
	ofBackground(100, 100, 100);
	ofSetColor(255, 255, 255);

	// draw instructions
	stringstream reportStream;
    
	ofSetColor(255, 255, 255);
	ofDrawBitmapString(reportStream.str(), 10, 610);

    featureFinder.draw();

    
	stringstream foundLabelsStream;
    for (int i=0; i<foundObjectLabels.size(); i++) {
        cout << "found label " << foundObjectLabels.at(i) << endl;
        
        foundLabelsStream << foundObjectLabels.at(i) << endl;
    }
    ofDrawBitmapString(foundLabelsStream.str(), 965, 10);
}


void testApp::guiEvent(ofxUIEventArgs &e) {
    
    string name = e.widget->getName();
    int kind = e.widget->getKind();
    
    if(kind == OFX_UI_WIDGET_LABELBUTTON)
    {
        ofxUILabelButton *button = (ofxUILabelButton *) e.widget;

    }
    
    if (e.widget->getName() == "PAUSE") {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        bPaused = toggle->getValue();
    }

    if (e.widget->getName() == "CLEAR REGIONS") {
        ofxUILabelButton *button = (ofxUILabelButton *) e.widget;
        
        if (button->getValue()) {
            featureFinder.clearRegions();
        }
    }
    else if(e.widget->getName() == "BLUR")
    {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        featureFinder.bBlur = toggle->getValue();
    }
    else if(e.widget->getName() == "STRETCH CONTRAST")
    {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        featureFinder.bStretchContrast = toggle->getValue();
    }
    else if(e.widget->getName() == "EQUALIZE HISTOGRAM")
    {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        featureFinder.bEqualizeHistogram = toggle->getValue();
    }
    
    else if(e.widget->getName() == "BLUR LEVEL")
    {
        ofxUISlider *slider = (ofxUISlider *) e.widget;
        
        featureFinder.setBlurLevel( slider->getScaledValue() );
        
    }
    else if(e.widget->getName() == "OCTAVES")
    {
        ofxUISlider *slider = (ofxUISlider *) e.widget;
        
        featureFinder.octaves = slider->getScaledValue();
        
    }
    else if(e.widget->getName() == "MIN HESSIAN")
    {
        ofxUISlider *slider = (ofxUISlider *) e.widget;
        
        featureFinder.hessianThreshold = slider->getScaledValue();
        
    }
    else if(e.widget->getName() == "MIN MATCHES")
    {
        ofxUISlider *slider = (ofxUISlider *) e.widget;
        
        featureFinder.minMatchCount = slider->getScaledValue();
        
    }
    
    
    else if(e.widget->getName() == "FULLSCREEN")
    {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        ofSetFullscreen(toggle->getValue());
    }
    else if (e.widget->getName() == "LOAD IMAGE") {
        
        ofxUILabelButton *button = (ofxUILabelButton *) e.widget;
        
        if (button->getValue()) {
            bPaused = true;
            
            this->openSampleImage();
        }
        
    }
    else if (e.widget->getName() == "SAVE DESCRIPTORS") {
        ofxUILabelButton *button = (ofxUILabelButton *) e.widget;
        
        if (button->getValue()) {

            ofFileDialogResult res = ofSystemSaveDialog("object.yml", "Save Object Description");
            
            if (res.bSuccess) {
                featureFinder.createAndSaveObject(res.getPath());
            }
        }
    }
    else if (e.widget->getName() == "LOAD DESCRIPTORS") {
        ofxUILabelButton *button = (ofxUILabelButton *) e.widget;
        
        if (button->getValue()) {

            ofFileDialogResult res = ofSystemLoadDialog("Load Object Description");
            
            if (res.bSuccess) {
                featureFinder.loadObject(res.getPath());
            }
            
        }
    }
    else if (e.widget->getName() == "DRAW CIRCLES") {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        featureFinder.bDrawCircles = toggle->getValue();
    }
    else if (e.widget->getName() == "DETECT OBJECTS") {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        bDetectObjects = toggle->getValue();
    }
    
    processRawImage();
}

void testApp::openSampleImage() {
    ofFileDialogResult res = ofSystemLoadDialog("Open Sample Image");
    
    if (!res.bSuccess) return;
    
    string filepath = res.getPath();
    cout << "Will load sample image " << filepath << endl;
    ofImage loaded;
    if (loaded.loadImage(filepath)) {
        
        rawImage.setFromPixels(loaded.getPixels(), loaded.width, loaded.height);
        this->processRawImage();
        
    }
    else {
        ofSystemAlertDialog("ERROR LOADING IMAGE " + filepath);
    }
}



//--------------------------------------------------------------
void testApp::keyPressed (int key) {
	switch (key) {
			
        case 's':
            ofImage sampleImage;
            
            sampleImage.setFromPixels((unsigned char *)rawImage.getPixels() , PROCESS_WIDTH, PROCESS_HEIGHT, OF_IMAGE_COLOR);

            sampleImage.saveImage("sample-image.png");
            snapCounter++;
	}
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button) {
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button) {
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button) {
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h)
{
}


//--------------------------------------------------------------
void testApp::exit() {
    gui->saveSettings("GUI/guiSettings.xml");
    delete gui;
}



