/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"

#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

using namespace dso;

int mode = 0;
int preset = 0;
int start = 0;
int end = 100000;
bool rec =1;
bool noros = 0;
bool nolog = 0;
bool nogui = 0;
bool nomt = 0;
bool save = 1;
bool useSampleOutput = false;
bool prefetch = false;
bool preload = false;
bool disableROS = false;
bool firstRosSpin = false;
float playbackSpeed = 0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
double rescale = 1;
std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
std::string saveresultto = "";


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}

void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;

        setting_desiredImmatureDensity = 1500;    //original 1500. set higher
        setting_desiredPointDensity = 2000;       //original 2000
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations=6;
        setting_minOptIterations=1;

        setting_kfGlobalWeight=0.3;   // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set this para higher
        setting_maxShiftWeightT= 0.04f * (640 + 128);   // original is 0.04f * (640+480); this para is depend on the crop size.
        setting_maxShiftWeightR= 0.04f * (640 + 128);    // original is 0.0f * (640+480);
        setting_maxShiftWeightRT= 0.02f * (640 + 128);  // original is 0.02f * (640+480);

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}

void parseAguement()
{
  	//Check settings file
	std::string settings_file_path = "config.yaml";
	cv::FileStorage settings_file(settings_file_path, cv::FileStorage::READ);
	if(!settings_file.isOpened())
	{
	  std::cerr << "Failed to open settings file at: " << settings_file_path << std::endl;
	  exit(-1);
	}

 	settings_file["useSampleOutput"] >> useSampleOutput;
	if(useSampleOutput) 
	  printf("USING SAMPLE OUTPUT WRAPPER!\n");
	settings_file["quiet"] >> setting_debugout_runquiet;
	if(setting_debugout_runquiet)
	  printf("QUIET MODE, I'll shut up!\n");
	settings_file["preset"] >> preset;
	settingsDefault(preset);
	settings_file["rec"] >> rec;
	if(!rec)
	{
	  disableReconfigure = true;
	  printf("DISABLE RECONFIGURE!\n");
	}
	settings_file["noros"] >> noros;
	if(noros)
	{
	  disableROS = true;
	  disableReconfigure = true;
	  printf("DISABLE zROS (AND RECONFIGURE)!\n");
	}
	settings_file["nolog"] >> nolog;
	if(nolog)
	{
	  setting_logStuff = false;
	  printf("DISABLE LOGGING!\n");
	}
	settings_file["nogui"] >> nogui;
	if(nogui)
	{
	  disableAllDisplay = true;
	  printf("NO GUI!\n");
	}
	settings_file["nomt"] >> nomt;
	if(nomt)
	{
	  multiThreading = false;
	  printf("NO MultiThreading!\n");
	}
	settings_file["prefetch"] >> prefetch;
	if(prefetch) printf("PREFETCH!\n");
	settings_file["start"] >> start;
	printf("START AT %d!\n",start);
	settings_file["end"] >> end;
	printf("END AT %d!\n",end);
	settings_file["DataSetRootDirectory"] >> source;
	printf("loading data from %s!\n", source.c_str());
	settings_file["calib"] >> calib;
	printf("loading calibration from %s!\n", calib.c_str());
	settings_file["vignette"] >> vignette;
	printf("loading vignette from %s!\n", vignette.c_str());
	settings_file["gammaCalib"] >> gammaCalib;
	printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
	settings_file["rescale"] >> rescale;
	printf("RESCALE %f!\n", rescale);
	settings_file["playbackSpeed"] >> playbackSpeed;
	printf("PLAYBACK SPEED %f!\n", playbackSpeed);
	settings_file["save"] >> save;
	if(save)
	{
	  debugSaveImages = true;
	  if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
	  if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
	  if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
	  if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
	  printf("SAVE IMAGES!\n");
	}
	settings_file["SaveResultTo"] >> saveresultto;
	
	settings_file["mode"] >> mode;
	switch(mode)
	{
	  case 0: printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		  break;
	  case 1: printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
		  setting_photometricCalibration = 0;
		  setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		  setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		  break;
	  case 2: printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
		  setting_photometricCalibration = 0;
		  setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
		  setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
		  setting_minGradHistAdd = 3;
		  break;
	  default: printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
	}
}

int main( int argc, char** argv )
{

	parseAguement();
	
	
	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);

	ImageFolderReader* reader_left = new ImageFolderReader(source+"/left", calib, gammaCalib, vignette);
	ImageFolderReader* reader_right = new ImageFolderReader(source+"/right", calib, gammaCalib, vignette);
	reader_left->setGlobalCalibration();
// 	reader_right->setGlobalCalibration();
	
	if(setting_photometricCalibration > 0 && reader_left->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}
	
	int lstart = start;
	int lend = end;
	
	// build system
	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader_left->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);
	
	
	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;				// left images
        std::vector<double> timesToPlayAt;

        std::vector<int> idsToPlayRight;		// right images
        std::vector<double> timesToPlayAtRight;

        int linc = 1;

        for(int i=lstart;i>= 0 && i< reader_left->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader_left->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader_left->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }

        for(int i=lstart;i>= 0 && i< reader_right->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlayRight.push_back(i);
            if(timesToPlayAtRight.size() == 0)
            {
                timesToPlayAtRight.push_back((double)0);
            }
            else
            {
                double tsThis = reader_right->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader_right->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAtRight.push_back(timesToPlayAtRight.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }



        std::vector<ImageAndExposure*> preloadedImagesLeft;
        std::vector<ImageAndExposure*> preloadedImagesRight;
		if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
			  int i = idsToPlay[ii];
			  preloadedImagesLeft.push_back(reader_left->getImage(i));
			  preloadedImagesRight.push_back(reader_right->getImage(i));
            }
        }

        // timing
        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        for(int ii=0; ii<(int)idsToPlay.size(); ii++)
        {
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img_left;
			ImageAndExposure* img_right;
			if(preload){
			  img_left = preloadedImagesLeft[ii];
			  img_right = preloadedImagesRight[ii];
			}
			else{
			  img_left = reader_left->getImage(i);
			  img_right = reader_right->getImage(i);
			}

            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }

            // if MODE_SLAM is true, it runs slam.
            bool MODE_SLAM = true;
            // if MODE_STEREOMATCH is true, it does stereo matching and output idepth image.
            bool MODE_STEREOMATCH = true;

            if(MODE_SLAM)
            {
                if(!skipFrame) fullSystem->addActiveFrame(img_left, img_right, i);
            }

            if(MODE_STEREOMATCH)
            {
                std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

                cv::Mat idepthMap(img_left->h, img_left->w, CV_32FC3, cv::Scalar(0,0,0));
                cv::Mat &idepth_temp = idepthMap;
                fullSystem->stereoMatch(img_left, img_right, i, idepth_temp);

                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                double ttStereoMatch = std::chrono::duration_cast<std::chrono::duration<double>>(t1 -t0).count();
                std::cout << " casting time " << ttStereoMatch << std::endl;
            }

            delete img_left;
	        delete img_right;

			// initializer fail
            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader_left->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed==0);


                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested=false;
                }
            }

            if(fullSystem->isLost)
            {
                    printf("LOST!!\n");
                    break;
            }

        }


        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("saveresultto + result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader_left->getTimestamp(idsToPlay[0])-reader_left->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader_left->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader_left->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader_left;

	printf("EXIT NOW!\n");
	return 0;
}
