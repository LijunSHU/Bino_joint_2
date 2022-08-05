#pragma once
#include<opencv2\opencv.hpp>
#include"HeaderCB.h"
#define mtype CV_32F
#define dtype float
class CornerDetAC {
public:

	CornerDetAC();
	~CornerDetAC();
	bool detectCorners(cv::Mat& src, Corners& mcorners, dtype scoreThreshold, bool isrefine, cv::Size boardSize, int check);
private:
	void secondDerivCornerMetric(cv::Mat I, int sigma, cv::Mat* cxy, cv::Mat* c45, cv::Mat* Ix, cv::Mat* Iy, cv::Mat* Ixy, cv::Mat* I_45_45);
	void nonMaximumSuppression(cv::Mat& inputCorners, std::vector<cv::Point2f>& outputCorners, int patchSize, dtype threshold, int margin);
	void getImageAngleAndWeight(cv::Mat img, cv::Mat& imgDu, cv::Mat& imgDv, cv::Mat& imgAngle, cv::Mat& imgWeight);
	void refineCorners(cv::Mat image, std::vector<cv::Point2f>& cornors, cv::Mat imgDu, cv::Mat imgDv, cv::Mat imgAngle, cv::Mat imgWeight, float radius);
	void edgeOrientations(cv::Mat imgAngle, cv::Mat imgWeight, int index);
	void findModesMeanShift(std::vector<dtype> hist, std::vector<dtype>& hist_smoothed, std::vector<std::pair<dtype, int>>& modes, dtype sigm);
	dtype normpdf(dtype dist, dtype mu, dtype sigma);
	float norm2d(cv::Point2f o);
	void scoreCorners(cv::Mat img, cv::Mat imgAngle, cv::Mat imgWeight, std::vector<cv::Point2f>& corners, std::vector<float>& score);
	void cornerCorrelationScore(cv::Mat img, cv::Mat imgWeight, std::vector<cv::Point2f> cornersEdge, float& score);
	void createkernel(float angle1, float angle2, int kernelSize, cv::Mat& kernelA, cv::Mat& kernelB, cv::Mat& kernelC, cv::Mat& kernelD);
	std::vector<std::vector<dtype>> cornersEdge1;
	std::vector<std::vector<dtype> > cornersEdge2;
	std::vector<int> radius;
};