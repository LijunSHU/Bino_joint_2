#pragma once
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\xfeatures2d.hpp>
#include<opencv2\stitching.hpp>
class pictureJoint {
public:
	pictureJoint();
	~pictureJoint();
	bool Image_Stitching(std::vector<cv::Mat>&imgs, cv::Mat&pano);
	void imageJoint(std::vector<cv::Mat> images);
};