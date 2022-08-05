#pragma once
#include<vector>
#include<opencv2\opencv.hpp>
struct Corners {
	std::vector<cv::Point2f> p;
	std::vector<cv::Vec2f> v1;
	std::vector<cv::Vec2f> v2;
	std::vector<float> score;
};