#include"pictureJoint.h"
#include<opencv2\imgcodecs\imgcodecs.hpp>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;
pictureJoint::pictureJoint() {

}
pictureJoint::~pictureJoint() {

}
bool pictureJoint::Image_Stitching(std::vector<cv::Mat>& imgs, cv::Mat& pano) {
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
    if (cv::Stitcher::OK != status) {
        std::cout << "failed to stitch images, err code:"<< status << std::endl;
        return false;
       // LOG(INFO) << "failed to stitch images, err code: " << (int)status;
    }
}

void pictureJoint::imageJoint(std::vector<cv::Mat> images) {

}
