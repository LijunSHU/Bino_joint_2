#pragma once
#include<opencv2\opencv.hpp>
class calibrate {
public:
	calibrate();
	~calibrate();
	void cameraCalibrate_(std::vector<cv::Mat>chessboards, cv::Size imgSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int boardSize,
		std::vector<std::vector<cv::Point3f>>& objectPoints, std::vector<std::vector<cv::Point2f>>& imagePoints);
	void stereoCalibrate(std::vector<cv::Mat>leftChessboards, std::vector<cv::Mat>rightChessboards,
		cv::Mat leftCameraMatrix, cv::Mat leftdistCoeffs, cv::Mat rightCameraMatrix, cv::Mat rightdistCoeffs, cv::Size imgSize);
	void stereoCalibrate2(std::vector<std::vector<cv::Point2f>> leftimagePoints, std::vector<std::vector<cv::Point2f>> rightimagePoints, std::vector<std::vector<
		cv::Point3f>> objectPoints, cv::Mat leftCameraMatrix, cv::Mat leftdistCoeffs, cv::Mat rightCameraMatrix, cv::Mat rightdistCoeffs, cv::Size imgSize);
	float calError(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints, cv::Mat cameraMatrix, cv::Mat distCoeffs,
		std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs);
};