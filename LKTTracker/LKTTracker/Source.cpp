
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/utility.hpp>

#include <omp.h>
#include "SIFT.h"
#include "Akaze.h"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace std;

int vmin = 10, vmax = 256, smin = 30;

struct rectImage {
	Rect rect;
	Mat image;

};

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

Mat getFrame(VideoCapture cap);
Mat fillHoles(Mat image);
rectImage findBiggestBlob(cv::Mat  matImage, Mat image);
Rect getBackprojROI(Mat hue, Mat histObj, Mat mask, Mat frame, Mat image);

int kn = 1; //Number of clusters Default
float siftR = 0.45;
bool useSIFT = true;
bool useGFTT = false;
bool useBackproj = false;
bool useKMeans = false;

bool bpInit = false;
bool showPoints = false;





Mat getHSVHistogram(Mat imgROI, Mat maskROI, int colorModel)
{
	//Mat hsvObj;
	//MatND histObj;

	Mat hist;
	int hsize = 150;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	if (colorModel == 1)
	{

		//Mat roi(hue, selection), maskroi(mask, selection);
		calcHist(&imgROI, 1, 0, maskROI, hist, 1, &hsize, &phranges);
		normalize(hist, hist, 0, 255, NORM_MINMAX);
		/*
		cvtColor(img1, hsvObj, COLOR_BGR2HSV);
		vector<Mat> channelsHSV;
		//histogram equalization ..just in H
		//split(hsvObj, channelsHSV); //split the image into channels
		//equalizeHist(channelsHSV[0], channelsHSV[0]); //equalize histogram on the 1st channel H
		//merge(channelsHSV, hsvObj); //merge 3 channels including the modified 1st channel into one image



		int SAT_MAX = 200;
		int SAT_MIN = 20;
		Mat mask;
		inRange(hsvObj, Scalar(0, 30, 32), Scalar(180, 255, 255), mask);


		int h_bins = 180; int s_bins = 32;
		int histSize[] = { h_bins };

		float h_range[] = { 0, 180 };
		float s_range[] = { 0, 255 };
		const float* ranges[] = { h_range };

		int channels[] = { 0 };

		/// Get the Histogram and normalize it
		calcHist(&hsvObj, 1, channels,mask, histObj, 1, histSize, ranges, true, false);
		normalize(histObj, histObj, 0, 255, NORM_MINMAX, -1, Mat());

		}
		else if (colorModel == 2)
		{

		int dims = 3;
		const int sizes[] = { 256, 256, 256 };
		const int channels[] = { 0, 1, 2 };
		float rRange[] = { 0, 256 };
		float gRange[] = { 0, 256 };
		float bRange[] = { 0, 256 };
		const float *ranges[] = { rRange, gRange, bRange };
		Mat mask = Mat();
		calcHist(&img1, 1, channels, mask, histObj, dims, sizes, ranges, true, false);
		//normalize(histObj, histObj, 0, 255, NORM_MINMAX, -1, Mat());
		}
		*/
	}


	return hist;
}



int main()
{
	//HSV trackbars
	namedWindow("S-V Trackbar", 0);
	setMouseCallback("S-V Trackbar", onMouse, 0);
	createTrackbar("Vmin", "S-V Trackbar", &vmin, 256, 0);
	createTrackbar("Vmax", "S-V Trackbar", &vmax, 256, 0);
	createTrackbar("Smin", "S-V Trackbar", &smin, 256, 0);


	//Get SIFT Keypoints for object
	Mat img1 = imread("mcd_logo1.jpg");
	SIFT sift1;

	KeyPointsDesc kpdObject, kpdFrame;
	Mat gray1;
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	kpdObject = sift1.getSIFTKeyPointsDesc(gray1);

	KeyPointsDesc akazeObject, akazeFrame;
	Akaze ak1;
	akazeObject = ak1.getAkazeKeyPointsDesc(gray1);



	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.0001);
	Size subPixWinSize(15, 15), winSize(31, 31);

	//cap.open("utrgvFlag.mp4");
	//capWebcam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//capWebcam.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	help();



	cap.open(0);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}
	cout << "Video Resolution : \n";
	cout << "Camera Frame width \n" << cap.get(CV_CAP_PROP_FRAME_WIDTH);
	cout << "Camera Frame height" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";

	namedWindow("LK Demo", 0);
	//setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image, frame, hsvFrame;
	vector<Point2f> points[2];

	Mat histObj;

	RotatedRect minRect;

	//Rect rectScene;

	int frameCnt = 0;
	omp_set_num_threads(2);
	Mat grayROI;
	vector<Point2f> gfttROI;
	Rect bufRect, globalRect;
	Mat mask, hue;
	Rect recInner;

	//WE initialze the frame size over here 
	// 780p resolution : 1280x720
	//1080p resoltuion : 1920 x 1080
	Size videoSize = Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	VideoWriter trackVideo("tracked.wmv", CV_FOURCC('W', 'M', 'V', '1'), 10, videoSize, true);
	



	for (;;)
	{

		frame = getFrame(cap);
		if (!cap.read(frame))
			break;

		resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);

		frame.copyTo(image);

		Mat RGBimage = image;

		cvtColor(image, gray, COLOR_BGR2GRAY);
		cvtColor(image, hsvFrame, COLOR_BGR2HSV);
		int _vmin = vmin, _vmax = vmax;

		inRange(hsvFrame, Scalar(0, smin, MIN(_vmin, _vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		int ch[] = { 0, 0 };
		hue.create(hsvFrame.size(), hsvFrame.depth());
		mixChannels(&hsvFrame, 1, &hue, 1, ch, 1);
		//GaussianBlur(image, image, Size(5, 5), 1.5, 1.5);
		//bilateralFilter(image, image, 5, 5, 5);
		Rect bpRect;
		if (!histObj.empty())
		{
			bpRect = getBackprojROI(hue, histObj, mask, frame, image);
		}

		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{

			if (useSIFT)
			{

				kpdFrame = sift1.getSIFTKeyPointsDesc(image);
				points[1] = sift1.FLANNMatchKeyPDesc(img1, image, kpdObject, kpdFrame, siftR);

				//akazeFrame = ak1.getAkazeKeyPointsDesc(gray);
				//points[1] = ak1.BFMatchKeyPDesc(img1, image, akazeObject, akazeFrame);
				//cout << "matched_points1 = " << endl << " " << points[1] << endl << endl;

				if (!points[1].empty() && points[1].size()>3)
				{
					// minRect = minAreaRect(points[1]);
					//Rect rectScene = boundingRect(points[1]);
					//rectScene = Rect(rectScene.x, rectScene.y, rectScene.width , rectScene.height );
					RotatedRect minR = minAreaRect(points[1]);
					//cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

					// Set Region of Interest to the area defined by the box
					cv::Rect rectScene;
					rectScene.x = minR.center.x - (minR.size.width / 2);
					rectScene.y = minR.center.y - (minR.size.height / 2);
					rectScene.width = minR.size.width;
					rectScene.height = minR.size.height;

					globalRect = rectScene;

					if (rectScene.x >= 0 && rectScene.y >= 0 && rectScene.width + rectScene.x < image.cols && rectScene.height + rectScene.y < image.rows)
					{
						// your code


						Mat ROI = hue(rectScene);
						Mat maskROI = mask(rectScene);
						//resizeWindow("rect scene", ROI.rows, ROI.cols);
						imwrite("roi.jpg", ROI);
						imshow("rect scene", ROI);
						bufRect = rectScene;
						waitKey(1);
						histObj = getHSVHistogram(ROI, maskROI, 1);
						ROI.release();

						//cvtColor(image(rectScene), grayROI, COLOR_BGR2GRAY);


						//Added to find GFTT corners in the ROI selected after SIFT
						grayROI = gray(rectScene);
						goodFeaturesToTrack(grayROI, gfttROI, MAX_COUNT, 0.01, 5, Mat(), 3, true, 0.04);
						Point offset;
						Size wholesize;
						grayROI.locateROI(wholesize, offset);
						for (int t = 0; t < gfttROI.size(); t++)
						{
							gfttROI[t] = Point2f(gfttROI[t].x + offset.x, gfttROI[t].y + offset.y);
						}

						std::copy(gfttROI.begin(), gfttROI.end(), std::back_inserter(points[1]));
						gfttROI.clear();

					}
					else
						continue;
				}

				int emptyCnt = 0;

				while (points[1].empty())
				{
					frame = getFrame(cap);
					frame.copyTo(image);
					cvtColor(image, gray, COLOR_BGR2GRAY);
					putText(image, "NO OBJECT DETECTED BY SIFT !", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 255), 1, CV_AA);
					imshow("LK Demo", image);
					waitKey(1);


					//#pragma omp parallel 
					if (emptyCnt % 30 == 0)
					{
						emptyCnt = 0;
						kpdFrame = sift1.getSIFTKeyPointsDesc(image);
						points[1] = sift1.FLANNMatchKeyPDesc(img1, image, kpdObject, kpdFrame, siftR);

						//akazeFrame = ak1.getAkazeKeyPointsDesc(gray);
						//points[1] = ak1.BFMatchKeyPDesc(img1, image, akazeObject, akazeFrame);

						if (!points[1].empty() && points[1].size() > 3)
						{
							RotatedRect minR = minAreaRect(points[1]);
							//cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

							// Set Region of Interest to the area defined by the box
							cv::Rect rectScene;
							rectScene.x = minR.center.x - (minR.size.width / 2);
							rectScene.y = minR.center.y - (minR.size.height / 2);
							rectScene.width = minR.size.width;
							rectScene.height = minR.size.height;

							globalRect = rectScene;

							if (rectScene.x >= 0 && rectScene.y >= 0 && rectScene.width + rectScene.x < image.cols && rectScene.height + rectScene.y < image.rows)
							{
								// your code

								Mat ROI1 = hue(rectScene);
								Mat maskROI1 = mask(rectScene);
								//resizeWindow("Gray rect scne", ROI1.rows, ROI1.cols);
								bufRect = rectScene;
								imshow("Gray rect scne", ROI1);

								waitKey(1);

								histObj = getHSVHistogram(ROI1, maskROI1, 1);
								ROI1.release();


								//Added to find GFTT corners in the ROI selected after SIFT
								grayROI = gray(rectScene);
								goodFeaturesToTrack(grayROI, gfttROI, MAX_COUNT, 0.01, 3, Mat(), 7, true, 0.04);
								Point offset;
								Size wholesize;
								grayROI.locateROI(wholesize, offset);
								for (int t = 0; t < gfttROI.size(); t++)
								{
									gfttROI[t] = Point2f(gfttROI[t].x + offset.x, gfttROI[t].y + offset.y);
								}
								std::copy(gfttROI.begin(), gfttROI.end(), std::back_inserter(points[1]));
								gfttROI.clear();


							}
							else
								continue;
						}
					}

					emptyCnt++;
					//imshow("LK Demo1", image);

				}
			}

			// automatic initialization
			//goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 7, true, 0.04);

			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;
		}
		else if (bpInit && frameCnt % 5 == 0)
		{
			frameCnt = 0;
			Rect rect1;

			//Add GFTT corners to the ROI of the K-means Rect
			vector<Point2f> KMPts;
			rect1 = bpRect;
			//rectangle(image, Point(rect1.x, rect1.y), Point(rect1.x + rect1.width, rect1.y + rect1.height), Scalar(0, 255, 0), 2, 8, 0);
			Mat ROI = hue(rect1);
			Mat maskROI = mask(rect1);
			histObj = getHSVHistogram(ROI, maskROI, 1);

			if (rect1.x >= 0 && rect1.y >= 0 && rect1.width + rect1.x < image.cols && rect1.height + rect1.y < image.rows)
			{
				goodFeaturesToTrack(gray(rect1), KMPts, MAX_COUNT, 0.01, 10, Mat(), 7, true, 0.04);

				Point offset;
				Size wholesize;
				gray(rect1).locateROI(wholesize, offset);
				for (int t = 0; t < KMPts.size(); t++)
				{
					KMPts[t] = Point2f(KMPts[t].x + offset.x, KMPts[t].y + offset.y);
					//circle(image, KMPts[t], 3, CV_RGB(255, 0, 0), -1, 8);
				}
				points[1] = KMPts;
				//cornerSubPix(gray, points[1], winSize, Size(-1, -1), termcrit);
				//std::copy(KMPts.begin(), KMPts.end(), std::back_inserter(points[1]));
				KMPts.clear();
			}

		}

		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);


			//calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);



			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 11, termcrit, 0, 0.001);


		
			//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
			int indexCorrection = 0;
			for (int i = 0; i<status.size(); i++)
			{
			Point2f pt = points[1].at(i - indexCorrection);
			if ((status.at(i) == 0) || (pt.x<0) || (pt.y<0))	{
			if ((pt.x<0) || (pt.y<0))	{
			status.at(i) = 0;
			}
			points[0].erase(points[0].begin() + i - indexCorrection);
			points[1].erase(points[1].begin() + i - indexCorrection);
			indexCorrection++;
			}

			}
			

			size_t i, k;

			for (i = k = 0; i < points[1].size(); i++)
			{
				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				if(showPoints)
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);




			if (points[1].size() > kn + 10 && useKMeans)
			{

				kn = 3;
				vector<int> bestLables;

				Mat centers;
				kmeans(points[1], kn, bestLables, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.0001), 20, cv::KMEANS_PP_CENTERS, centers);

				// Find the largest cluster
				int max = 0, indx = 0, id = 0;
				vector<int> clusters(kn, 0);

				for (size_t i = 0; i < bestLables.size(); i++)
				{
					id = bestLables[i];
					clusters[id]++;

					if (clusters[id] > max)
					{
						max = clusters[id];
						indx = id;
					}
				}

				// save largest cluster
				int cluster = indx;

				vector<Point2f> shape;
				//shape.reserve(2000);

				for (int y = 0; y < points[1].size(); y++)
				{
					if (bestLables[y] == cluster)
					{
						shape.push_back(points[1][y]);
					}

				}


				//minRect = minAreaRect(shape);

				//Rect rect1 = boundingRect(shape);
				recInner = boundingRect(shape);
				if (recInner.area() > (image.rows*image.cols) / 3 || recInner.area() <100)
				{
					if (!points[1].empty())
						points[1].clear();
					putText(image, "OBJECT NOT VISIBLE !", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 255), 2, CV_AA);
				}
				else{
				rectangle(image, Point(recInner.x, recInner.y), Point(recInner.x + recInner.width, recInner.y + recInner.height), Scalar(0, 255, 0), 2, 8, 0);
				}

				bestLables.clear();
				clusters.clear();
				shape.clear();

				/*
				//Add GFTT corners to the ROI of the K-means Rect
				vector<Point2f> KMPts;
				if (rect1.x >= 0 && rect1.y >= 0 && rect1.width + rect1.x < image.cols && rect1.height + rect1.y < image.rows)
				{
				goodFeaturesToTrack(gray(rect1), KMPts, MAX_COUNT, 0.01, 10, Mat(), 7, true, 0.04);
				Point offset;
				Size wholesize;
				gray(rect1).locateROI(wholesize, offset);
				for (int t = 0; t < KMPts.size(); t++)
				{
				KMPts[t] = Point2f(KMPts[t].x + offset.x, KMPts[t].y + offset.y);
				circle(image, KMPts[t], 3, CV_RGB(255, 0, 0), -1, 8);
				}
				points[1] = KMPts;
				//std::copy(KMPts.begin(), KMPts.end(), std::back_inserter(points[1]));
				KMPts.clear();
				}
				*/

			}
			else
			{
				recInner = boundingRect(points[1]);
				if (recInner.area() > (image.rows*image.cols) / 3 || recInner.area() <100)
				{
					if (!points[1].empty())
						points[1].clear();
					putText(image, "OBJECT NOT VISIBLE !", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 255), 2, CV_AA);
				}
				else {
					rectangle(image, Point(recInner.x, recInner.y), Point(recInner.x + recInner.width, recInner.y + recInner.height), Scalar(0, 255, 0), 2, 8, 0);
				}
			}




			//vec.copyTo(rectPoints);
			//points[1].clear();
			//points[1] = rectPoints;
			// rotated rectangle
			// Point2f rect_points[4]; minRect.points(rect_points);
			//for (int j = 0; j < 4; j++)
			//	line(image, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(255, 255, 255), 3, 8);




			if (useBackproj)
			{
				/// Get Backprojection
				float hranges[] = { 0, 180 };
				const float* phranges = hranges;
				Mat backproj;
				calcBackProject(&hue, 1, 0, histObj, backproj, &phranges);
				backproj &= mask;

				//calcBackProject(&hsvFrame, 1, channels, histObj, backproj, ranges, 1, true);
				Mat struct_element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
				filter2D(backproj, backproj, -1, struct_element);
				//Mat bilFil;
				//	bilateralFilter(backproj, bilFil, 5, 7, 7);
				//	GaussianBlur(backproj, backproj, Size(9, 9), 1.5, 1.5);
				namedWindow("BackProjection", 0);
				//backproj = bilFil;
				imshow("BackProjection", backproj);
				//equalizeHist(backproj, backproj);
				//Ptr<CLAHE> cl = createCLAHE();
				//cl->apply(backproj, backproj);
				//imshow("backpp-clahe", backproj);

				//term_crit1 = TermCriteria(TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 80, 1);
				TermCriteria termcrit1(TermCriteria::COUNT | TermCriteria::EPS, 80, 1);
				Rect msRect = bufRect;

				//meanShift(backproj, msRect, termcrit1);
				//rectangle(image, Point(msRect.x,msRect.y), Point(msRect.x +msRect.height, msRect.y+msRect.width), CV_RGB(255, 255, 255), 3, 8);
				//(image, "Mean-Shift Tracked", Point(msRect.x -10, msRect.y -10), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 1, CV_AA);

				Mat threholdBP;
				threshold(backproj, threholdBP, 240, 255, 0);

				vector<Mat> threshMerge;
				threshMerge.push_back(threholdBP); threshMerge.push_back(threholdBP); threshMerge.push_back(threholdBP);
				Mat threshMergeI;
				merge(threshMerge, threshMergeI);
				Mat res, res1;
				bitwise_and(frame, threshMergeI, res);

				// Create a structuring element (SE)
				int morph_size = 2;
				Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
				morphologyEx(threholdBP, threholdBP, MORPH_DILATE, element, Point(-1, -1), 3);
				morphologyEx(threholdBP, threholdBP, MORPH_ERODE, element, Point(-1, -1), 2);

				cv::Mat holes = threholdBP.clone();
				cv::floodFill(holes, cv::Point2i(0, 0), cv::Scalar(1));
				bitwise_and(holes, holes, res1);


				//Mat res2 = findBiggestBlob(res1,image);

				//res1=fillHoles(backproj);
				namedWindow("Backprojection Thresholded", 0);
				//imshow("Backprojection Thresholded", res2);
			}



		}


		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);

			cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		if (bpInit)
		{
			putText(image, "NOW USING BACK-PROJECTION !", cvPoint( 30, image.rows - 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, CV_RGB(0, 0, 255), 2, CV_AA);
		}
		else
		{
			putText(image, "NO BACK-PROJECTION !", cvPoint(30,image.rows-30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, CV_RGB(0, 0, 255), 2, CV_AA);
		}


		needToInit = false;
		//bpInit = false;

		imshow("LK Demo", image);
		trackVideo.write(image);
		waitKey(20);

		char c = (char)waitKey(1);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		case 'b':
			bpInit = !bpInit;
			break;
		case 'p':
			showPoints = !showPoints;
			break;
		case 'k':
			useKMeans = !useKMeans;
			break;

		}


		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
		frameCnt++;

	}

	return 0;
}




Mat getFrame(VideoCapture cap)
{
	Mat frame;
	cap >> frame;
	if (frame.empty())
		exit;

	return frame;
}

Mat fillHoles(Mat image)
{
	Mat image_thresh;
	cv::threshold(image, image_thresh, 50, 255, cv::THRESH_BINARY);
	// Loop through the border pixels and if they're black, floodFill from there
	cv::Mat mask;
	image_thresh.copyTo(mask);
	for (int i = 0; i < mask.cols; i++) {
		if (mask.at<char>(0, i) == 0) {
			cv::floodFill(mask, cv::Point(i, 0), 255, 0, 10, 10);
		}
		if (mask.at<char>(mask.rows - 1, i) == 0) {
			cv::floodFill(mask, cv::Point(i, mask.rows - 1), 255, 0, 10, 10);
		}
	}
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<char>(i, 0) == 0) {
			cv::floodFill(mask, cv::Point(0, i), 255, 0, 10, 10);
		}
		if (mask.at<char>(i, mask.cols - 1) == 0) {
			cv::floodFill(mask, cv::Point(mask.cols - 1, i), 255, 0, 10, 10);
		}
	}


	// Compare mask with original.
	cv::Mat newImage;
	image.copyTo(newImage);
	for (int row = 0; row < mask.rows; ++row) {
		for (int col = 0; col < mask.cols; ++col) {
			if (mask.at<char>(row, col) == 0) {
				newImage.at<char>(row, col) = 255;
			}
		}
	}

	return newImage;
}

rectImage findBiggestBlob(cv::Mat src, Mat image) {
	//int largest_area = 0;
	//int largest_contour_index = 0;
	//Mat temp(src.rows, src.cols, CV_8UC1);
	Mat dst;
	cvtColor(src, dst, CV_GRAY2BGR);
	Rect rect1;

	//Canny(src, src, 50, 255, 3);

	//Mat dst=Mat(src.rows,src.cols,CV_8UC3);
	//src.copyTo(temp);

	vector<vector<Point>> contours, contoursOuter; // storing contour
	vector<Vec4i> hierarchy;
	bitwise_not(src, src);
	//invert(src, src);

	Mat outerCntr = Mat(src.rows, src.cols, src.type());
	findContours(src.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	drawContours(outerCntr, contours, hierarchy[i][0],CV_RGB(255,255,255),CV_FILLED);
	//}
	//namedWindow("OC", 0);
	//imshow("OC", outerCntr);



	double maxArea = 0, secMaxArea;
	int maxAreaIndex = 0, secondMaxAreaInd = 0;
	vector<double> areas(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		areas[i] = contourArea(Mat(contours[i]));
		if (areas[i] > maxArea)
		{
			maxArea = areas[i];
			secondMaxAreaInd = maxAreaIndex;
			maxAreaIndex = i;
		}
		//	cout << areas[i] << "\n";
	}
	//double max, min;
	//Point maxPosition, minPosition;
	//minMaxLoc(Mat(areas), &min, &max, &minPosition, &maxPosition);
	//drawContours(dst, contours, minPosition.y, CV_FILLED, 1);
	if (contourArea(contours[secondMaxAreaInd])>500)
	{
		rect1 = boundingRect(contours[secondMaxAreaInd]);
		rectangle(dst, rect1, CV_RGB(0, 255, 0), 2, 8);
	}
	//	rectangle(dst, Point(rect1.x, rect1.y), Point(rect1.x + rect1.width, rect1.y + rect1.height), Scalar(0, 255, 0), 2, 8, 0);


	//imshow("outermost contour", heirarch1_outermost);

	//drawContours(dst, contours, secondMaxAreaInd, CV_RGB(0,255,255), 3, 8, hierarchy,0);
	// Draw the largest contour
	rectImage rectIO;
	rectIO.image = dst;
	rectIO.rect = rect1;

	return rectIO;
}

Rect getBackprojROI(Mat hue, Mat histObj, Mat mask, Mat frame, Mat image)
{
	/// Get Backprojection
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	Mat backproj;
	calcBackProject(&hue, 1, 0, histObj, backproj, &phranges);
	backproj &= mask;

	//calcBackProject(&hsvFrame, 1, channels, histObj, backproj, ranges, 1, true);
	Mat struct_element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	filter2D(backproj, backproj, -1, struct_element);
	//Mat bilFil;
	//	bilateralFilter(backproj, bilFil, 5, 7, 7);
	//	GaussianBlur(backproj, backproj, Size(9, 9), 1.5, 1.5);
	namedWindow("BackProjection", 0);
	//backproj = bilFil;
	imshow("BackProjection", backproj);
	//equalizeHist(backproj, backproj);
	//Ptr<CLAHE> cl = createCLAHE();
	//cl->apply(backproj, backproj);
	//imshow("backpp-clahe", backproj);

	//term_crit1 = TermCriteria(TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 80, 1);
	//TermCriteria termcrit1(TermCriteria::COUNT | TermCriteria::EPS, 80, 1);
	//Rect msRect = bufRect;

	//meanShift(backproj, msRect, termcrit1);
	//rectangle(image, Point(msRect.x,msRect.y), Point(msRect.x +msRect.height, msRect.y+msRect.width), CV_RGB(255, 255, 255), 3, 8);
	//(image, "Mean-Shift Tracked", Point(msRect.x -10, msRect.y -10), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 1, CV_AA);

	Mat threholdBP;
	threshold(backproj, threholdBP, 240, 255, 0);

	vector<Mat> threshMerge;
	threshMerge.push_back(threholdBP); threshMerge.push_back(threholdBP); threshMerge.push_back(threholdBP);
	Mat threshMergeI;
	merge(threshMerge, threshMergeI);
	Mat res, res1;
	bitwise_and(frame, threshMergeI, res);

	// Create a structuring element (SE)
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(threholdBP, threholdBP, MORPH_DILATE, element, Point(-1, -1), 3);
	morphologyEx(threholdBP, threholdBP, MORPH_ERODE, element, Point(-1, -1), 2);

	cv::Mat holes = threholdBP.clone();
	cv::floodFill(holes, cv::Point2i(0, 0), cv::Scalar(1));
	bitwise_and(holes, holes, res1);

	rectImage rectIO;
	Mat res2;
	rectIO = findBiggestBlob(res1, image);
	res2 = rectIO.image;
	Rect r1 = rectIO.rect;
	//res1=fillHoles(backproj);
	namedWindow("Backprojection Thresholded", 0);
	imshow("Backprojection Thresholded", res2);

	return r1;
}
