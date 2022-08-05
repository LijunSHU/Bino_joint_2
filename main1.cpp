#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include<io.h>
#include<fstream>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
//#define ISH
//#define ISRECT
#define ISPANO
using namespace cv;
using namespace std;
using namespace cv::detail;
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

four_corners_t corners;


bool getFiles(std::string path, std::vector<std::string>& files) {
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					getFiles(p.assign(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	if (files.size() == 0)
		return false;
	return true;
}



void CalcCorners(const Mat& H, const Mat& src)
{
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

    V1 = H * V2;
    //左上角(0,0,1)
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

}

int main(int argc, char* argv[])
{
    cv::Mat leftCameraMatrix, rightCameraMatrix, leftDistcoeffs, rightDistcoeffs;
#ifdef ISRECT
    std::string path1 = "./data/7.20/mono/left/";
    std::string path2 = "./data/7.20/mono/right/";
    std::vector<std::string> files1, files2;
    getFiles(path1, files1);
    getFiles(path2, files2);
    std::vector<std::vector<cv::Point2f>> leftPoints, rightPoints;
    cv::Size imgSize;
    std::vector<cv::Mat> leftImages, rightImages;
    for (int i = 0; i < files1.size(); i++) {
        cv::Mat imgLeft = cv::imread(path1 + files1[i]);
        cv::Mat imgRight = cv::imread(path2 + files2[i]);

        std::vector<cv::Point2f> leftCorners, rightCorners;
        cv::findChessboardCorners(imgLeft, cv::Size(11, 8), leftCorners);
        cv::findChessboardCorners(imgRight, cv::Size(11, 8), rightCorners);
        leftPoints.push_back(leftCorners);
        rightPoints.push_back(rightCorners);
        imgSize = imgLeft.size();
        leftImages.push_back(imgLeft);
        rightImages.push_back(imgRight);
    }
    std::vector<std::vector<cv::Point3f>> objectPoints;
    for (int i = 0; i < files1.size(); i++) {
        std::vector<cv::Point3f> object;
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 11; col++) {
                object.push_back(cv::Point3f(row * 25, col * 25, 0));
            }
        }
        objectPoints.push_back(object);
    }
    std::vector<cv::Mat> rvecs1, rvecs2;
    std::vector<cv::Mat> tvecs1, tvecs2;
    cv::calibrateCamera(objectPoints, leftPoints, imgSize, leftCameraMatrix, leftDistcoeffs, rvecs1, tvecs1);
    std::cout << "left:" << leftCameraMatrix << "\n" << leftDistcoeffs << std::endl;
    cv::calibrateCamera(objectPoints, rightPoints, imgSize, rightCameraMatrix, rightDistcoeffs, rvecs2, tvecs2);
    std::cout << "right:" << rightCameraMatrix << "\n" << rightDistcoeffs << std::endl;
    cv::FileStorage fs1;
    fs1.open("cameraParam.xml", cv::FileStorage::WRITE);

    fs1 << "leftCameraMatrix" << leftCameraMatrix;
    fs1 << "leftDistcoeffs" << leftDistcoeffs;
    fs1 << "rightCameraMatrix" << rightCameraMatrix;
    fs1 << "rightDistcoeffs" << rightDistcoeffs;
    fs1.release();
#else
    cv::FileStorage fs1;
    fs1.open("cameraParam.xml", cv::FileStorage::READ);
    fs1["leftCameraMatrix"] >> leftCameraMatrix;
    fs1["leftDistcoeffs"] >> leftDistcoeffs;
    fs1["rightCameraMatrix"] >> rightCameraMatrix;
    fs1["rightDistcoeffs"] >> rightDistcoeffs;
    fs1.release();
#endif
//求单应矩阵
    Mat image01 = imread("./data/8.4/bino/1/30.bmp", 1);    //右图
    Mat image02 = imread("./data/8.4/bino/1/29.bmp", 1);    //左图
    cv::Mat unImg01, unImg02;
    cv::Size imgSize1(image01.cols, image01.rows);
    //const double alpha = 1;
    //cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(leftCameraMatrix, leftDistcoeffs, imgSize, alpha, imgSize, 0);
    unImg01 = image01.clone();
    unImg02 = image02.clone();
    //cv::undistort(image01, unImg01, leftCameraMatrix, leftDistcoeffs);
   //cv::undistort(image02, unImg02, rightCameraMatrix, rightDistcoeffs);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat leftMapx1, leftMapy1, rightMapx2, rightMapy2;
    cv::Mat newc = cv::getOptimalNewCameraMatrix(leftCameraMatrix, leftDistcoeffs, cv::Size(imgSize1.width * 0.7, imgSize1.height * 0.7), 0);
    cv::Mat newc2 = cv::getOptimalNewCameraMatrix(rightCameraMatrix, rightDistcoeffs, cv::Size(imgSize1.width * 0.7, imgSize1.height * 0.7), 0);

    cv::initUndistortRectifyMap(leftCameraMatrix, leftDistcoeffs, R, newc, imgSize1, CV_32FC1, leftMapx1, leftMapy1);
    cv::initUndistortRectifyMap(rightCameraMatrix, rightDistcoeffs, R, newc2, imgSize1, CV_32FC1, rightMapx2, rightMapy2);
    cv::remap(image01, unImg01, leftMapx1, leftMapy1, cv::INTER_LANCZOS4);
    cv::remap(image02, unImg02, rightMapx2, rightMapy2, cv::INTER_LANCZOS4);

   
    //cv::namedWindow("un", cv::WINDOW_NORMAL);
    //cv::namedWindow("un2", cv::WINDOW_NORMAL);

    //cv::imshow("un", unImg01);
    //cv::imshow("un2", unImag02);
    //cv::waitKey(0);
    cv::imwrite("unImg1.png", unImg01);
    cv::imwrite("unImg2.png", unImg02);
    //灰度图转换  
    Mat image1, image2;
    cvtColor(unImg01, image1, cv::COLOR_RGB2GRAY);
    cvtColor(unImg02, image2, cv::COLOR_RGB2GRAY);
    vector<Point2f> imagePoints1, imagePoints2;

#if 0
    //提取特征点    
    vector<KeyPoint> keypoints1, keypoints2;

    Mat descriptors_box, descriptors_sence;

    Ptr<ORB> detector = ORB::create();

    detector->detectAndCompute(image01, Mat(), keypoints1, descriptors_sence);

    detector->detectAndCompute(image02, Mat(), keypoints2, descriptors_box);


    vector<DMatch> matches;

    // 初始化flann匹配

    // Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create(); // default is bad, using local sensitive hash(LSH)

    Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));//12, 20, 2

    matcher->match(descriptors_box, descriptors_sence, matches);


    // 发现匹配

    vector<DMatch> goodMatches;

    printf("total match points : %d\n", matches.size());

    float maxdist = 0;

    for (unsigned int i = 0; i < matches.size(); ++i) {

        printf("dist : %.2f \n", matches[i].distance);

        maxdist = max(maxdist, matches[i].distance);

    }

    for (unsigned int i = 0; i < matches.size(); ++i) {

        if (matches[i].distance < maxdist * 0.4)

            goodMatches.push_back(matches[i]);

    }


    for (int i = 0; i < goodMatches.size(); i++)
    {
        imagePoints1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        imagePoints2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }
    std::cout << imagePoints2.size() << "\t" << imagePoints1.size() << std::endl;
#else
    cv::findChessboardCorners(image1, cv::Size(11, 8), imagePoints1);
    cv::findChessboardCorners(image2, cv::Size(11, 8), imagePoints2);
    std::vector<cv::Point2f> imagePoints;
    
    cv::drawChessboardCorners(image01, cv::Size(11, 8), imagePoints1, true);
    cv::drawChessboardCorners(image02, cv::Size(11, 8), imagePoints2, true);
    for (int i = 0; i < imagePoints1.size(); i++) {
       // cv::putText(unImg01, std::to_string(i), cv::Point(imagePoints1[i]), 0.3, 0.4, cv::Scalar(255, 0, 0));
        //cv::putText(unImg02, std::to_string(i), cv::Point(imagePoints2[i]), 0.3, 0.4, cv::Scalar(255, 0, 0));
        //cv::circle(unImg01, cv::Point(imagePoints1[i]), 1, cv::Scalar(0, 255, 0), 1);
        //cv::circle(unImg02, cv::Point(imagePoints2[i]), 1, cv::Scalar(0, 255, 0), 1);
    }
    //std::cout << imagePoints1.size() << "\t" << imagePoints2.size() << std::endl;
    //cv::imwrite("1.png", unImg01);
    //cv::imwrite("2.png", unImg02);
    //cv::namedWindow("1", cv::WINDOW_NORMAL);
    //cv::namedWindow("2", cv::WINDOW_NORMAL);

    //cv::imshow("1", unImg01);
    //cv::imshow("2", unImg02);
    //cv::waitKey(0);

#endif

#if 1
    //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
    Mat homo;
#ifdef ISH

    cv::FileStorage fs;
    fs.open("Homography.xml", cv::FileStorage::WRITE);
    homo = findHomography(imagePoints1, imagePoints2, cv::RANSAC);
    fs << "Homography" << homo;
    fs.release();
#else    
    cv::FileStorage fs;
    fs.open("Homography.xml", cv::FileStorage::READ);
    fs["Homography"] >> homo;
    fs.release();
#endif // ISH

    cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      

                                                //计算配准图的四个顶点坐标
    CalcCorners(homo, unImg01);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    //图像配准  
    Mat imageTransform1, imageTransform2;
#ifdef ISPANO
    warpPerspective(unImg01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), unImg02.rows));
    //warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
     //cv::imshow("直接经过透视矩阵变换", imageTransform1);
     //cv::imwrite("trans1.jpg", imageTransform1);


    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = unImg02.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);
    //cv::namedWindow("tras", cv::WINDOW_NORMAL);
    //cv::imshow("tras", imageTransform1);
    //cv::waitKey(0);
    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    unImg02.copyTo(dst(Rect(0, 0, unImg02.cols, unImg02.rows)));
    cv::imwrite("dst.png", dst);
   // cv::imshow("b_dst", dst);
    
#else
    warpPerspective(unImg01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), MAX(corners.right_top.y, corners.right_bottom.y)));
    int dst_width = imageTransform1.cols;
    int dst_height = imageTransform1.rows;
    cv::Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);
    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    unImg02.copyTo(dst(Rect(0, 0, unImg02.cols, unImg02.rows)));
#endif

    OptimizeSeam(unImg02, imageTransform1, dst);
    std::cout << dst.type() << std::endl;
    int c;
    bool is = true;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<cv::Vec3b>(i, j)[0] == 0 && dst.at<cv::Vec3b>(i, j)[1] == 0 && dst.at<cv::Vec3b>(i, j)[2] == 0) {
                c = i;
                is = false;
                break;
            }
        }
    }
    std::cout << "c:" << c<< std::endl;
    dst.rowRange(c, dst.rows);
    std::cout << "check:" << dst.rows << std::endl;
    cv::imwrite("dst.jpg", dst);

   // waitKey();
#else


#endif
    return 0;
}


//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
    int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
    std::cout << "dst:" << dst.size() << std::endl;
    double processWidth = img1.cols - start;//重叠区域的宽度  
    int rows = dst.rows;
    int cols = img1.cols; //注意，是列数*通道数
    double alpha = 1.0;//img1中像素的权重  
#ifdef ISPANO
    int n = 0;
    for (int i = 0; i < rows; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
        uchar* t = trans.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = start; j < cols; j++)
        {
            //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }
            else
            {
                //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
                alpha = (processWidth - (j - start)) / processWidth;
            }
            if (j > processWidth) {
                // std::cout << "j:" << j * 3 << std::endl;
                d[j * 3] = p[j * 3] * (alpha)+t[j * 3] * (1 - alpha);
                d[j * 3 + 1] = p[j * 3 + 1] * (alpha)+t[j * 3 + 1] * (1 - alpha);
                d[j * 3 + 2] = p[j * 3 + 2] * (alpha)+t[j * 3 + 2] * (1 - alpha);
            }
            else {
                d[j * 3] = p[j * 3] * (1) + t[j * 3] * (1 - 1);
                d[j * 3 + 1] = p[j * 3 + 1] * (1) + t[j * 3 + 1] * (1 - 1);
                d[j * 3 + 2] = p[j * 3 + 2] * (1) + t[j * 3 + 2] * (1 - 1);
            }

        }
        //std::cout << std::endl;
    }
#else
    //cols = dst.cols;
    for (int i = 0; i < rows; i++) {
        if (i < img1.rows) {
            for (int j = 0; j < cols; j++) {
                if (trans.at<cv::Vec3b>(i, j)[0] == 0 && trans.at<cv::Vec3b>(i, j)[1] == 0 && trans.at<cv::Vec3b>(i, j)[2] == 0) {
                    alpha = 1;
                }
                else {
                    alpha = (processWidth - (j - start)) / processWidth;
                }
                //std::cout << "alpha:" << i << j << std::endl;
                if (j > processWidth) {
                    dst.at<cv::Vec3b>(i, j)[0] = img1.at<cv::Vec3b>(i, j)[0] * alpha + trans.at<cv::Vec3b>(i, j)[0] * (1 - alpha);
                    dst.at<cv::Vec3b>(i, j)[1] = img1.at<cv::Vec3b>(i, j)[1] * alpha + trans.at<cv::Vec3b>(i, j)[1] * (1 - alpha);
                    dst.at<cv::Vec3b>(i, j)[2] = img1.at<cv::Vec3b>(i, j)[2] * alpha + trans.at<cv::Vec3b>(i, j)[2] * (1 - alpha);
                }
                else {
                    dst.at<cv::Vec3b>(i, j)[0] = img1.at<cv::Vec3b>(i, j)[0];
                    dst.at<cv::Vec3b>(i, j)[1] = img1.at<cv::Vec3b>(i, j)[1];
                    dst.at<cv::Vec3b>(i, j)[2] = img1.at<cv::Vec3b>(i, j)[2];

                }
            }
            //std::cout << "i:" << i << std::endl;
        }

    }
#endif // ISPANO


}
