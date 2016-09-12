#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;




class Akaze{
	public:

		const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
		const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
		const double ransac_thresh = 2.5f;

		KeyPointsDesc getAkazeKeyPointsDesc(Mat img1)
		{
			vector<KeyPoint> kpts1;
			Mat desc1;

			Ptr<AKAZE> akaze = AKAZE::create();
			akaze->detectAndCompute(img1, noArray(), kpts1, desc1);

			KeyPointsDesc local;
			local.keypoints = kpts1;
			local.descriptors = desc1;

			return local;
			//	return 0;
		}


		vector<Point2f> BFMatchKeyPDesc(Mat img1, Mat img2, KeyPointsDesc obj, KeyPointsDesc frame)
		{

			Mat first_desc = obj.descriptors;
			Mat desc = frame.descriptors;
			vector<cv::KeyPoint> first_kp = obj.keypoints;
			vector<cv::KeyPoint> kp = frame.keypoints;
			vector<Point2f> empty;
			BFMatcher matcher(NORM_HAMMING);

			/*
			BFMatcher matcher(NORM_HAMMING);
			vector< vector<DMatch> > nn_matches;
			matcher.knnMatch(descriptors_1, descriptors_2, nn_matches, 2);

			vector<KeyPoint> matched1, matched2;
			vector<DMatch> good_matches;
			for (size_t i = 0; i < nn_matches.size(); i++) {
				DMatch first = nn_matches[i][0];
				float dist1 = nn_matches[i][0].distance;
				float dist2 = nn_matches[i][1].distance;

				if (dist1 < nn_match_ratio * dist2) {
					matched1.push_back(keypoints1[first.queryIdx]);
					matched2.push_back(keypoints2[first.trainIdx]);
				}
			}

			Mat inlier_mask, homography;
			vector<KeyPoint> inliers1, inliers2;
			vector<DMatch> inlier_matches;
			if (matched1.size() >= 4) {
				homography = findHomography(Points(matched1), Points(matched2),
					RANSAC, ransac_thresh, inlier_mask);


				for (unsigned i = 0; i < matched1.size(); i++) {
					if (inlier_mask.at<uchar>(i)) {
						int new_i = static_cast<int>(inliers1.size());
						inliers1.push_back(matched1[i]);
						inliers2.push_back(matched2[i]);
						inlier_matches.push_back(DMatch(new_i, new_i, 0));
					}
				}


				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img1.cols, 0);
				obj_corners[2] = Point2f(img1.cols, img1.rows); obj_corners[3] = Point2f(0, img1.rows);
				vector<Point2f> new_bb(4);
				perspectiveTransform(obj_corners, new_bb, homography);
				return new_bb;
			}
			else
				return empty;
				*/


	
			vector< vector<DMatch> > matches;
			vector<KeyPoint> matched1, matched2;
			matcher.knnMatch(first_desc, desc, matches, 2);
			for (unsigned i = 0; i < matches.size(); i++) {
				if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
					matched1.push_back(first_kp[matches[i][0].queryIdx]);
					matched2.push_back(kp[matches[i][0].trainIdx]);
				}
			}
			//stats.matches = (int)matched1.size();

			Mat inlier_mask, homography;
			vector<KeyPoint> inliers1, inliers2;
			vector<DMatch> inlier_matches;
			if (matched1.size() >= 4) {
				homography = findHomography(Points(matched1), Points(matched2),
					RANSAC, ransac_thresh, inlier_mask);
			}

			if (matched1.size() < 4 || homography.empty()) {
				//Mat res;
				//hconcat(first_frame, frame, res);
				//stats.inliers = 0;
				//stats.ratio = 0;
				return empty;
			}

			for (unsigned i = 0; i < matched1.size(); i++) {
				if (inlier_mask.at<uchar>(i)) {
					int new_i = static_cast<int>(inliers1.size());
					inliers1.push_back(matched1[i]);
					inliers2.push_back(matched2[i]);
					inlier_matches.push_back(DMatch(new_i, new_i, 0));
				}
			}
			//stats.inliers = (int)inliers1.size();
			//stats.ratio = stats.inliers * 1.0 / stats.matches;
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img1.cols, 0);
			obj_corners[2] = Point2f(img1.cols, img1.rows); obj_corners[3] = Point2f(0, img1.rows);
			vector<Point2f> new_bb;
			perspectiveTransform(obj_corners, new_bb, homography);

			Mat res;
			drawMatches(img1, inliers1, img2, inliers2,
				inlier_matches, res,
				Scalar(255, 0, 0), Scalar(255, 0, 0));
			namedWindow("AKAZE Matches", 0);
			imshow("AKAZE Matches", res);
			
			//Mat frame_with_bb = frame.clone();
			return new_bb;

			//if (stats.inliers >= bb_min_inliers) {
			//	drawBoundingBox(frame_with_bb, new_bb);
			//}


		}


		vector<Point2f> Points(vector<KeyPoint> keypoints)
		{
			vector<Point2f> res;
			for (unsigned i = 0; i < keypoints.size(); i++) {
				res.push_back(keypoints[i].pt);
			}
			return res;
		}

};