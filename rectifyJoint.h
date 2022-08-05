#pragma once
#include <opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
typedef struct
{
	cv::Point2f left_top;
	cv::Point2f left_bottom;
	cv::Point2f right_top;
	cv::Point2f right_bottom;
}four_corners_t;

class rectifyJoint {
public:
	rectifyJoint();
	~rectifyJoint();
	cv::Point2i getOffset(cv::Mat img, cv::Mat img1);
	cv::Mat cylinder(cv::Mat imgIn, int f);
	cv::Mat linearFusion(cv::Mat img, cv::Mat img1, cv::Point2i a);
	cv::Mat linearFusion2(cv::Mat img, cv::Mat img1, cv::Point2i a);

	void imageOverlap(cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, cv::Mat h1, cv::Mat h2);
	void cylinderOn(cv::Mat src);
	cv::Mat CylindricalWarp(cv::Mat imgMat);
	cv::Mat CylindricalWarp2(cv::Mat imgMat, float f);
	void CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t& corners);
	//cv::Mat cylinder(cv::Mat imgIn, int f);
};