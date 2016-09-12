
/////////////////////

/*
Bassam Arshad
0259149
Project-04 SIFT Detector & Descriptors
*/

#include <opencv2/core/core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include<iostream>

using namespace cv;
using namespace std;

void   BFMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2, float ratio);
void   FLANNMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2, float ratio);
void objectBoundingBox(vector<Point2f> obj, vector<Point2f> scene, Mat img_object, Mat img_matches);

struct KeyPointsDesc
{
	std::vector<cv::KeyPoint> keypoints;
	Mat descriptors;
};

struct StatusPoints
{
	vector<Point2f> points;
	int status;
};

class SIFT
{
public:

	KeyPointsDesc getSIFTKeyPointsDesc(Mat img1)
	{
		//Mat img1 = imread("school.jpg");
		//Mat img2 = imread("sign1.png");

		//Mat gray1;

		//Convert to grayscale
		//cvtColor(img1, gray1, COLOR_BGR2GRAY);
		//cvtColor(img2, gray2, COLOR_BGR2GRAY);


		Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10, 1.0);


		std::vector<cv::KeyPoint> keypoints1;
		Mat descriptors_1;

		//Detect the KeyPoints
		sift->detect(img1, keypoints1);
		//sift->detect(img2, keypoints2);

		//Compute the Feature Descriptors for the KeyPoints
		sift->compute(img1, keypoints1, descriptors_1);
		//sift->compute(img2, keypoints2, descriptors_2);


		//Use Brute Force Matcher
		//BFMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2,0.6);
		//Use FLANN Matcher
		//FLANNMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2, 0.7);


		// Add results to image and save.
		//cv::Mat output;
		//cv::drawKeypoints(img1, keypoints1, output, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//cv::imshow("sift_result.jpg", output);

		//	waitKey();

		KeyPointsDesc local;
		local.keypoints = keypoints1;
		local.descriptors = descriptors_1;

		return local;
		//	return 0;
	}


	KeyPointsDesc getSURFKeyPointsDesc(Mat img1)
	{


		Ptr<xfeatures2d::SURF>surf = xfeatures2d::SURF::create();

		surf->setHessianThreshold(400);
		std::vector<cv::KeyPoint> keypoints1;
		Mat descriptors_1;

		//Detect the KeyPoints
		surf->detect(img1, keypoints1);
		//sift->detect(img2, keypoints2);

		//Compute the Feature Descriptors for the KeyPoints
		surf->compute(img1, keypoints1, descriptors_1);
		//sift->compute(img2, keypoints2, descriptors_2);


		//Use Brute Force Matcher
		//BFMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2,0.6);
		//Use FLANN Matcher
		//FLANNMatching(img1, img2, keypoints1, keypoints2, descriptors_1, descriptors_2, 0.7);


		// Add results to image and save.
		//cv::Mat output;
		//cv::drawKeypoints(img1, keypoints1, output, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//cv::imshow("sift_result.jpg", output);

		//	waitKey();

		KeyPointsDesc local;
		local.keypoints = keypoints1;
		local.descriptors = descriptors_1;

		return local;
		//	return 0;
	}













	void   BFMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2, float r)
	{
		//Using the below .match for the matcher gives a lot of results --> implemented the knn one , for better results.
		/*
		//Using Brute Force Matcher - To match the descriptors
		BFMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(descriptors_1, descriptors_2, matches);
		//matcher.radiusMatch(descriptors_1, descriptors_2, matches,2);
		*/

		std::vector<std::vector<cv::DMatch>> matches;
		cv::BFMatcher matcher;
		//k-nearest neighbor matcher
		matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

		vector<cv::DMatch> good_matches;
		for (int i = 0; i < matches.size(); ++i)
		{
			const float ratio = r; // As in Lowe's paper; can be tuned
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}

		Mat img_matches;
		//drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, CV_RGB(0, 255, 0), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Show detected matches
		namedWindow("BF-Matcher SIFT Matches", 0);
		imshow("BF-Matcher SIFT Matches", img_matches);

	}

	vector<Point2f> FLANNMatchKeyPDesc(Mat img1, Mat img2, KeyPointsDesc obj, KeyPointsDesc frame, float r)
	{

		Mat descriptors_1 = obj.descriptors;
		Mat descriptors_2 = frame.descriptors;
		vector<cv::KeyPoint> keypoints1 = obj.keypoints;
		vector<cv::KeyPoint> keypoints2 = frame.keypoints;
		vector<Point2f> empty;


		FlannBasedMatcher matcher;
		std::vector<std::vector<cv::DMatch>> matches;
		//k-nearest neighbor matcher
		matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches


		vector<cv::DMatch> good_matches;
		for (int i = 0; i < matches.size(); ++i)
		{
			const float ratio = r; // As in Lowe's paper; can be tuned
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}



		if (good_matches.empty() || good_matches.size() < 4)
		{
			//std::cout << "\n No Object in Frame !!";
			//exit;

			//empty.clear();
			return  empty;
			//throw 20;
			//exit;
		}
		else
		{

			long num_matches = good_matches.size();
			vector<Point2f> matched_points1;
			vector<Point2f> matched_points2;

			for (int i1 = 0; i1 < num_matches; i1++)
			{
				Point2f point1 = keypoints1[good_matches[i1].queryIdx].pt;
				Point2f point2 = keypoints2[good_matches[i1].trainIdx].pt;
				matched_points1.push_back(point1);
				matched_points2.push_back(point2);
			}


			//vector<Point2f> scene_corners;
			//scene_corners = objectBoundingBox(matched_points1, matched_points2, img1, img2);
			//cout << "matched_points1 = " << endl << " " << matched_points1 << endl << endl;
			//cout << "matched_points2 = " << endl << " " << matched_points2 << endl << endl;


			Mat H = findHomography(matched_points1, matched_points2, CV_RANSAC);
			//cout << "H = " << endl << " " << H << endl << endl;
			//-- Get the corners from the image_1 ( the object to be "detected" )
			//Mat obj_corners =  Mat(4, 1, CV_32FC2);
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img1.cols, 0);
			obj_corners[2] = Point2f(img1.cols, img1.rows); obj_corners[3] = Point2f(0, img1.rows);
			std::vector<Point2f> scene_corners(4);
			//Mat scene_corners = Mat(4, 1, CV_32FC2);

			if (!H.empty())
			{
				perspectiveTransform(obj_corners, scene_corners, H);
				//	getAffineTransform(matched_points1, scene_corners);
				//std::vector<Point2f> allScenePoints;
				//std::copy(scene_corners.begin(), scene_corners.end(), std::back_inserter(allScenePoints));
				//std::copy(matched_points2.begin(), matched_points2.end(), std::back_inserter(allScenePoints));
				return scene_corners;
			}
			else
				return empty;



		}
	}


	void   FLANNMatching(Mat img1, Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, Mat descriptors_1, Mat descriptors_2, float r)
	{


		FlannBasedMatcher matcher;
		std::vector<std::vector<cv::DMatch>> matches;
		//k-nearest neighbor matcher
		matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

		vector<cv::DMatch> good_matches;
		for (int i = 0; i < matches.size(); ++i)
		{
			const float ratio = r; // As in Lowe's paper; can be tuned
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}

		long num_matches = good_matches.size();
		vector<Point2f> matched_points1;
		vector<Point2f> matched_points2;

		for (int i1 = 0; i1<num_matches; i1++)
		{
			//int idx1 = good_matches[i1].trainIdx;
			//int idx2 = good_matches[i1].queryIdx;
			//matched_points1.push_back(keypoints1[idx1].pt);
			//matched_points2.push_back(keypoints2[idx2].pt);
			Point2f point1 = keypoints1[good_matches[i1].queryIdx].pt;
			Point2f point2 = keypoints2[good_matches[i1].trainIdx].pt;
			matched_points1.push_back(point1);
			matched_points2.push_back(point2);
		}


		Mat img11 = img1.clone();
		for (int i = 0; i < matched_points1.size(); i++)
		{
			circle(img11, matched_points1[i], 3, Scalar(0, 255, 0), -1, 8);

		}

		imshow("Image 1 MAtches", img11);

		Mat img12 = img2.clone();
		for (int i = 0; i < matched_points2.size(); i++)
		{
			circle(img12, matched_points2[i], 3, Scalar(0, 255, 0), -1, 8);

		}
		imshow("Image 2 MAtches", img12);

		//-- Draw only "good" matches
		Mat img_matches;
		//drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, CV_RGB(0, 255, 0), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Show detected matches
		namedWindow("FLANN-Matcher SIFT Matches", 0);
		imshow("FLANN-Matcher SIFT Matches", img_matches);

		objectBoundingBox(matched_points1, matched_points2, img1, img2);

	}


	vector<Point2f> objectBoundingBox(vector<Point2f> obj, vector<Point2f> scene, Mat img_object, Mat img_matches)
	{

		/*
		Mat contour_poly;
		Rect boundRect;
		approxPolyDP(Mat(scene), contour_poly, 3, true);
		boundRect = boundingRect(Mat(contour_poly));
		rectangle(img_matches, Point(boundRect.x, boundRect.y), Point(boundRect.x + boundRect.height, boundRect.y + boundRect.width), CV_RGB(0, 0, 255), 1, 8, 0);
		*/

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners;
		//obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
		//obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
		std::vector<Point2f> scene_corners;

		perspectiveTransform(obj, scene_corners, H);

		/*
		Rect boundRect;
		boundRect = boundingRect(Mat(scene_corners));
		rectangle(img_matches, Point(boundRect.x, boundRect.y), Point(boundRect.x + boundRect.height, boundRect.y + boundRect.width), CV_RGB(0, 0, 255), 1, 8, 0);
		*/

		/*
		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);


		line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
		*/
		//Rect newTemp(scene_corners[0], y1pt, (x2pt - x1pt), (y2pt - y1pt)) ;

		//Mat imgROI = img2(newTemp);

		return scene_corners;




		//-- Show detected matches
		//	imshow("Good Matches & Object detection", img_matches);
	}


	vector<Point2f> FLANNMatchKeyPDescSURF(Mat img1, Mat img2, KeyPointsDesc obj, KeyPointsDesc frame, float r)
	{

		Mat descriptors_1 = obj.descriptors;
		Mat descriptors_2 = frame.descriptors;
		vector<cv::KeyPoint> keypoints1 = obj.keypoints;
		vector<cv::KeyPoint> keypoints2 = frame.keypoints;
		vector<Point2f> empty;




		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector<std::vector<cv::DMatch>> matches;
		//k-nearest neighbor matcher
		matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_1.rows; i++)
		{
			double dist = matches[i][0].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_1.rows; i++)
		{
			if (matches[i][0].distance < 2 * min_dist)
			{
				good_matches.push_back(matches[i][0]);
			}
		}


		if (good_matches.empty() || good_matches.size() < 5)
		{
			//std::cout << "\n No Object in Frame !!";
			//exit;

			//empty.clear();
			return  empty;
			//throw 20;
			//exit;
		}
		else
		{

			long num_matches = good_matches.size();
			vector<Point2f> matched_points1;
			vector<Point2f> matched_points2;

			for (int i1 = 0; i1 < num_matches; i1++)
			{
				Point2f point1 = keypoints1[good_matches[i1].queryIdx].pt;
				Point2f point2 = keypoints2[good_matches[i1].trainIdx].pt;
				matched_points1.push_back(point1);
				matched_points2.push_back(point2);
			}


			//vector<Point2f> scene_corners;
			//scene_corners = objectBoundingBox(matched_points1, matched_points2, img1, img2);
			cout << "matched_points1 = " << endl << " " << matched_points1 << endl << endl;
			cout << "matched_points2 = " << endl << " " << matched_points2 << endl << endl;


			Mat H = findHomography(matched_points1, matched_points2, CV_RANSAC);
			cout << "H = " << endl << " " << H << endl << endl;
			//-- Get the corners from the image_1 ( the object to be "detected" )
			//Mat obj_corners =  Mat(4, 1, CV_32FC2);
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img1.cols, 0);
			obj_corners[2] = Point2f(img1.cols, img1.rows); obj_corners[3] = Point2f(0, img1.rows);
			std::vector<Point2f> scene_corners(4);
			//Mat scene_corners = Mat(4, 1, CV_32FC2);

			if (!H.empty())
			{
				perspectiveTransform(obj_corners, scene_corners, H);
				//	getAffineTransform(matched_points1, scene_corners);
				//std::vector<Point2f> allScenePoints;
				//std::copy(scene_corners.begin(), scene_corners.end(), std::back_inserter(allScenePoints));
				//std::copy(matched_points2.begin(), matched_points2.end(), std::back_inserter(allScenePoints));
				return scene_corners;
			}
			else
				return empty;



		}
	}



















};

