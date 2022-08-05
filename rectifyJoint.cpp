#include"rectifyJoint.h"
#include<iostream>
#include<fstream>
using namespace cv;

rectifyJoint::rectifyJoint() {

}
rectifyJoint::~rectifyJoint() {

}
cv::Point2i rectifyJoint::getOffset(cv::Mat img, cv::Mat img1) {
	//cv::Mat temp1(img1, cv::Rect(0, 0.4 * img1.rows, 0.2 * img1.cols, 0.2 * img1.rows));
	//std::cout << "getoffset temp1 size:" << temp1.size() << std::endl;
	//cv::Mat result(img.cols - temp1.cols + 1, img.rows - temp1.rows + 1, CV_8UC1);
	////cv::namedWindow("img");
	////cv::imshow("img", img);
	////cv::imshow("temp", temp1);
	////cv::waitKey(0);
	//cv::matchTemplate(img, temp1, result, cv::TM_CCOEFF_NORMED);//模板匹配
	//cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);
	//double minVal, maxVal;
	//cv::Point minLoc, maxLoc, mathcLoc;
	//cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	//mathcLoc = maxLoc;
	//int dx = mathcLoc.x;
	//int dy = mathcLoc.y - 0.4 * img1.rows;
	//cv::Point2i a(dx, dy);

	//cv::Mat t(img1.rows, 100, img1.type());
	//int mincols;
	//std::vector<int> colsArray;
	//for (int i = 1; i < img1.rows; i++) {
	//	for (int j = 0; j < img1.cols; j++) {
	//		if (img1.at<uchar>(i, j) != 255) {
	//			//mincols = j;
	//			//break;
	//			colsArray.push_back(j);
	//		}
	//	}
	//}
	//std::sort(colsArray.begin(), colsArray.end());
	//mincols = colsArray[0];
	//std::cout << "mincols:" << mincols << std::endl;


	cv::Mat t = img1.colRange(864, 1064);//936, 1036
	cv::Mat ori;
	ori = t.clone();



	//cv::Mat ori = img1.colRange(0, 350);
	////ori = ori.rowRange(600, 1400);
	cv::imshow("ori", ori);
	cv::waitKey(0);
	cv::Mat result;
	cv::matchTemplate(img, ori, result, cv::TM_CCOEFF_NORMED);


	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);
	double minVal, maxVal;
	cv::Point minLoc, maxLoc, mathLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	mathLoc = maxLoc;
	int dx = mathLoc.x;
	int dy = mathLoc.y;
	std::cout << "dx dy:" << dx << "\t" << dy << std::endl;
	
	cv::Point2i a(dx, dy);
	//a.y = 110;
	return a;
}

//cv::Mat rectifyJoint::cylinder(cv::Mat imgIn, int f) {
//	int colNum, rowNum;
//	colNum = 2 * f * atan(0.5 * imgIn.cols / f); //柱面图像宽
//	rowNum = 0.5 * imgIn.rows * f / sqrt(pow(f, 2)) + 0.5 * imgIn.rows;
//	cv::Mat imgOut = cv::Mat::zeros(rowNum, colNum, CV_8UC1);
//	cv::Mat_<uchar>im1(imgIn);
//	cv::Mat_<uchar>im2(imgOut);
//	int x1(0), y1(0);
//	for (int i = 0; i < imgIn.rows; i++) {
//		for (int j = 0; j < imgIn.cols; j++) {
//			x1 = f * atan((j - 0.5 * imgIn.cols) / f) + f * atan(0.5 * imgIn.cols / f);
//			y1 = f * (i - 0.5 * imgIn.rows) / sqrt(pow(j - 0.5 * imgIn.cols, 2) + pow(f, 2)) + 0.5 * imgIn.rows;
//			if (x1 >= 0 && x1 < colNum && y1 >= 0 && y1 < rowNum)
//			{
//				im2(y1, x1) = im1(i, j);
//			}
//		}
//	}
//	return imgOut;
//}
/**柱面投影函数
 *参数列表中imgIn为输入图像，f为焦距
 *返回值为柱面投影后的图像
*/
cv::Mat rectifyJoint::cylinder(Mat imgIn, int f)
{
	int colNum, rowNum;
	colNum = 2 * f * atan(0.5 * imgIn.cols / f);//柱面图像宽
	rowNum = 0.5 * imgIn.rows * f / sqrt(pow(f, 2)) + 0.5 * imgIn.rows;//柱面图像高

	Mat imgOut = Mat::zeros(rowNum, colNum, CV_8UC1);
	Mat_<uchar> im1(imgIn);
	Mat_<uchar> im2(imgOut);

	//正向插值
	int x1(0), y1(0);
	for (int i = 0; i < imgIn.rows; i++)
		for (int j = 0; j < imgIn.cols; j++)
		{
			x1 = f * atan((j - 0.5 * imgIn.cols) / f) + f * atan(0.5 * imgIn.cols / f);
			y1 = f * (i - 0.5 * imgIn.rows) / sqrt(pow(j - 0.5 * imgIn.cols, 2) + pow(f, 2)) + 0.5 * imgIn.rows;
			if (x1 >= 0 && x1 < colNum && y1 >= 0 && y1 < rowNum)
			{
				im2(y1, x1) = im1(i, j);
			}
		}
	return imgOut;
}
cv::Mat rectifyJoint::CylindricalWarp(cv::Mat imgMat)
{
	//cv::Mat imgMat = cv::imread("./data/cy/im0.png");
	double r = imgMat.cols;
	int w = atan2(imgMat.cols / 2, r) * 2 * r;
	int h = imgMat.rows;
	cv::Mat destImgMat = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
	for (int y = 0; y < destImgMat.rows; y++)
	{
		for (int x = 0; x < destImgMat.cols; x++)
		{
			cv::Point2f current_pos(x, y);

			float point_x = r * tan(current_pos.x / r - atan2(w, 2 * r)) + w / 2;
			// float point_y = r * tan(current_pos.y / r - atan2(h, 2 * r)) + h / 2;
			float point_y = (current_pos.y - h / 2) * sqrt(r * r + (w / 2 - x) * (w / 2 - x)) / r + h / 2;
			cv::Point2f original_point(point_x, point_y);

			cv::Point2i top_left((int)(original_point.x), (int)(original_point.y)); //top left because of integer rounding

			//make sure the point is actually inside the original image
			if (top_left.x < 0 || top_left.x > imgMat.cols - 2 || top_left.y < 0 || top_left.y > imgMat.rows - 2)
			{
				continue;
			}

			//bilinear interpolation
			float dx = original_point.x - top_left.x;
			float dy = original_point.y - top_left.y;

			float weight_tl = (1.0 - dx) * (1.0 - dy);
			float weight_tr = (dx) * (1.0 - dy);
			float weight_bl = (1.0 - dx) * (dy);
			float weight_br = (dx) * (dy);
			for (int k = 0; k < 3; k++)
			{
				uchar value = weight_tl * imgMat.at<cv::Vec3b>(top_left)[k] +
					weight_tr * imgMat.at<cv::Vec3b>(top_left.y, top_left.x + 1)[k] +
					weight_bl * imgMat.at<cv::Vec3b>(top_left.y + 1, top_left.x)[k] +
					weight_br * imgMat.at<cv::Vec3b>(top_left.y + 1, top_left.x + 1)[k];

				destImgMat.at<cv::Vec3b>(y, x)[k] = value;
			}
		}
	}
	return destImgMat;
}

cv::Mat rectifyJoint::linearFusion(cv::Mat img, cv::Mat img1, cv::Point2i a) {
	std::vector<float> value;
#if 0
	int d = img.cols - a.x;
	int row = img.rows;
	int col = img1.cols - d;
	cv::Mat stitch = cv::Mat::zeros(row, a.x + img.cols, img.type());
	cv::Mat t1 = img.clone();
	cv::Mat t2 = img1.colRange(d, img1.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			stitch.at<cv::Vec3b>(i, j)[0] = t1.at<cv::Vec3b>(i, j)[0];
			stitch.at<cv::Vec3b>(i, j)[1] = t1.at<cv::Vec3b>(i, j)[1];
			stitch.at<cv::Vec3b>(i, j)[2] = t1.at<cv::Vec3b>(i, j)[2];
		}
	}
	std::cout << "stitch size:" << stitch.size() << std::endl;
	std::cout << "check:" << d << "\t" << img.cols << "\t" << img1.cols << std::endl;
	for (int i = 0; i < t2.rows; i++) {
		for (int j = 0; j < t2.cols; j++) {
			//.at<uchar>(i, j + img.cols - 1) = img1.at<uchar>(i, j);
			uchar tt1 = t2.at<cv::Vec3b>(i, j)[0];
			uchar tt2 = t2.at<cv::Vec3b>(i, j)[1];
			uchar tt3 = t2.at<cv::Vec3b>(i, j)[2];

			stitch.at<cv::Vec3b>(i, img.cols + j)[0] = tt1;
			stitch.at<cv::Vec3b>(i, img.cols + j)[1] = tt2;
			stitch.at<cv::Vec3b>(i, img.cols + j)[2] = tt3;

		}
	}
	cv::imwrite("stitch.png", stitch);

	return stitch;
#else
	float d = 1522 - a.x;//1602
	std::cout << "d:" << d << std::endl;
	cv::Mat dst = cv::Mat::zeros(img.rows, img.cols*2, img.type());
	double alpha1 = 1.0, alpha2= 1.0;
	int t1 = 0;
	int tt;
	for (int i = a.y; i < img.rows; i++) {
		int t = 0;
	
		for (int j = a.x; j < 1522; j++) {
			if ( img.at<cv::Vec3b>(i, j)[0] == 0 && img.at<cv::Vec3b>(i, j)[1] == 0 && img.at<cv::Vec3b>(i, j)[2] == 0) {
				alpha1 = 1;
				alpha2 = 1;
				//t++;
			}
			else {
				alpha1 = (d - float(j - a.x)) / d;
				alpha2 = 1 - alpha1;


				//alpha1 = 1 - (int(j - a.x) / d);
				//alpha2 = 1 - alpha1;

				//alpha1 = std::pow((img.cols  - j), 5) / (std::pow(j - a.x , 5) + std::pow( a.x - j,5));
				//alpha2 = 1 - alpha1;

				//alpha1 = 0.2;
				//alpha2 = 0.8;

			}
			dst.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0] * ( alpha1) + img1.at<cv::Vec3b>(t1, t)[0] * (alpha2);
			dst.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1] * ( alpha1) + img1.at<cv::Vec3b>(t1, t)[1] * (alpha2);
			dst.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2] * ( alpha1) + img1.at<cv::Vec3b>(t1, t)[2] * (alpha2);
			t++;
		}
		tt = t - 1;
		t1++;
	}
	std::cout <<"t:" <<  tt << std::endl;
	for (int i = a.y; i < img.rows; i++) {
		for (int j = 0; j < a.x; j++) {
			dst.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
			dst.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
			dst.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
		}
	}
	int r = 0;
	for (int i = 0; i < img1.rows- a.y; i++) {
		int c = 0;							
		for (int k = tt + 864; k < img1.cols; k++) {
			//dst.at<uchar>(i, k + a.x) = img1.at<uchar>(i, k);

			dst.at<cv::Vec3b>(a.y + r, c + 1522)[0] = img1.at<cv::Vec3b>(i, k)[0];
			dst.at<cv::Vec3b>(a.y + r, c + 1522)[1] = img1.at<cv::Vec3b>(i, k)[1];
			dst.at<cv::Vec3b>(a.y + r, c + 1522)[2] = img1.at<cv::Vec3b>(i, k)[2];
			c++;
		}
		r++;
	}
	img1.colRange(tt + 864, img1.cols);
	cv::Mat d1 = dst.clone();
	//cv::GaussianBlur(dst, d1, cv::Size(3, 3), 1.5);
	//cv::medianBlur(dst, d1, 7);
	cv::imwrite("stitch.png", d1);
	std::cout << "end" << std::endl;
	return dst;
#endif

}

//void get_pics_stitiching(int val_resize) {
//	int size_t = val_resize;
//	cv::Mat target = cv::Mat::zeros(size_t, size_t, CV_16UC1);
//	cv::Mat rect;
//	cv::cvtColor(target, rect, cv::COLOR_GRAY2BGR);
//	std::string imgs_index = { "left", "right" };
//	cv::Mat leftImg = cv::imread();
//	cv::Mat rightImg = cv::imread();
//	std::vector<cv::Mat> imgs_set;
//	imgs_set.push_back(leftImg);
//	imgs_set.push_back(rightImg);
//
//}
//
//void circle_stitch() {
//	int val_resize = 512;
//	get_pics_stitiching(val_resize);
//}

void rotateImage(Mat& dst)//旋转图像
{
	Point center(dst.cols / 2, dst.rows / 2);
	double angle = 180;//旋转180度
	double scale = 1.0;//不缩放
	Mat rotMat = getRotationMatrix2D(center, angle, scale);//计算旋转矩阵
	warpAffine(dst, dst, rotMat, dst.size());//生成图像
}

void rectifyJoint::cylinderOn(cv::Mat src) {
	int nbottom = 0;
	int ntop = 0;
	int nright = 0;
	int nleft = 0;
	nright = src.cols;
	//nleft = 0;
	nbottom = src.rows;
	int d = min(nright - nleft, nbottom - ntop);
	//cv::imshow("ori", src);
	std::cout << "1:" << nright - nleft << "\t" << nbottom - ntop << std::endl;
	cv::Mat imgRoi;
	imgRoi = src(cv::Rect(nleft, ntop, d, d));
	std::cout << "ROI:" << imgRoi.size() << std::endl;
	cv::imshow("ROI", imgRoi);
	//cv::waitKey(0);
	cv::Mat dst(imgRoi.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat map_x, map_y;
	map_x.create(imgRoi.size(), CV_32FC1);
	map_y.create(imgRoi.size(), CV_32FC1);
	for (int j = 0; j < d - 1; j++) {
		for (int i = 0; i < d - 1; i++) {
			map_x.at<float>(i, j) = static_cast<float>(d / 2.0 + i / 2.0 * cos(1.0 * j / d * 2.0 * CV_PI));
			map_y.at<float>(i, j) = static_cast<float>(d / 2.0 + i / 2.0 * sin(1.0 * j / d * 2.0 * CV_PI));
		}
	}
	std::cout << map_x.size() << "\tmap:" << map_y.size() << std::endl;
	cv::remap(imgRoi, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	std::cout << "before size:" << dst.size() << std::endl;
	cv::resize(dst, dst, Size(), 2.0, 1.0);
	std::cout << "after size:" << dst.size() << std::endl;
	rotateImage(dst);
	cv::imshow("result", dst);
	cv::waitKey(0);
}



cv::Mat rectifyJoint::linearFusion2(cv::Mat img, cv::Mat img1, cv::Point2i a) {
	std::vector<float> value;
	//std::fstream fout("alpha.txt", std::ios::out);
#if 0
	int d = img.cols - a.x;
	int row = img.rows;
	int col = img1.cols - d;
	cv::Mat stitch = cv::Mat::zeros(row, a.x + img.cols, img.type());
	cv::Mat t1 = img.clone();
	cv::Mat t2 = img1.colRange(d, img1.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			stitch.at<cv::Vec3b>(i, j)[0] = t1.at<cv::Vec3b>(i, j)[0];
			stitch.at<cv::Vec3b>(i, j)[1] = t1.at<cv::Vec3b>(i, j)[1];
			stitch.at<cv::Vec3b>(i, j)[2] = t1.at<cv::Vec3b>(i, j)[2];
		}
	}
	std::cout << "stitch size:" << stitch.size() << std::endl;
	std::cout << "check:" << d << "\t" << img.cols << "\t" << img1.cols << std::endl;
	for (int i = 0; i < t2.rows; i++) {
		for (int j = 0; j < t2.cols; j++) {
			//.at<uchar>(i, j + img.cols - 1) = img1.at<uchar>(i, j);
			uchar tt1 = t2.at<cv::Vec3b>(i, j)[0];
			uchar tt2 = t2.at<cv::Vec3b>(i, j)[1];
			uchar tt3 = t2.at<cv::Vec3b>(i, j)[2];

			stitch.at<cv::Vec3b>(i, img.cols + j)[0] = tt1;
			stitch.at<cv::Vec3b>(i, img.cols + j)[1] = tt2;
			stitch.at<cv::Vec3b>(i, img.cols + j)[2] = tt3;

		}
	}
	cv::imwrite("stitch.png", stitch);

	return stitch;
#else
	float d = img1.cols - a.x;
	std::cout << "d:" << d << std::endl;
	cv::Mat dst = cv::Mat::zeros(img1.rows, img1.cols + a.x, img.type());
	float alpha1, alpha2;
	int t;
	for (int i = 0; i < img.rows; i++) {
		//int t = d - 1;
		t = 530;
		for (int j = a.x; j < img.cols; j++) {
			if (img1.at<cv::Vec3b>(i, j)[0] == 0 && img1.at<cv::Vec3b>(i, j)[1] == 0 && img1.at<cv::Vec3b>(i, j)[2] == 0) {
				alpha1 = 1;
				alpha2 = 1;
				//t++;
			}
			else {
				//alpha = (d - float(j - a.x)) / d;
				alpha1 = 1 - (float(j - a.x) / d);
				alpha2 = 1 - alpha1;
			}
			dst.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0] * (alpha1)+img1.at<cv::Vec3b>(i, t)[0] * (alpha2);
			dst.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1] * (alpha1)+img1.at<cv::Vec3b>(i, t)[1] * (alpha2);
			dst.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2] * (alpha1)+img1.at<cv::Vec3b>(i, t)[2] * (alpha2);
			t++;
		}
	}
	//fout.close();
	//for (int i = 0; i < img1.rows; i++) {
	//	for (int j = 0; j < a.x; j++) {
	//		//dst.at<uchar>(i, j) = img1.at<uchar>(i, j);

	//		dst.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
	//		dst.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
	//		dst.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
	//	}
	//}

	//for (int i = 0; i < img1.rows; i++) {
	//	for (int k = d; k < img1.cols; k++) {
	//		//dst.at<uchar>(i, k + a.x) = img1.at<uchar>(i, k);

	//		dst.at<cv::Vec3b>(i, k + a.x)[0] = img1.at<cv::Vec3b>(i, k)[0];
	//		dst.at<cv::Vec3b>(i, k + a.x)[1] = img1.at<cv::Vec3b>(i, k)[1];
	//		dst.at<cv::Vec3b>(i, k + a.x)[2] = img1.at<cv::Vec3b>(i, k)[2];
	//	}
	//}
	cv::Mat d1 = dst.clone();
	cv::imwrite("stitch.png", d1);
	//cv::imshow("s", d1);
	//cv::waitKey(0);
	std::cout << "end" << std::endl;
	return dst;
#endif

}


cv::Mat rectifyJoint::CylindricalWarp2(cv::Mat imgMat, float f) {
	float w = imgMat.cols;
	float h = imgMat.rows;
	float w1 = 2 * f * atan(w / 2.0 * f);
	cv::Mat destImgMat = cv::Mat::zeros(cv::Size(w1, h), CV_8UC3);
	for (int y = 0; y< imgMat.rows; y++) {
		for (int x = 0; x < imgMat.cols; x++) {
			float theta = atan((x - w * 0.5) / f);
			float x1 = w1 * 0.5 + f * theta;
			float y1 = (f * (y - h / 2.0)) / (std::sqrt(std::pow(x - w / 2.0, 2) + f * f)) + h / 2.0;
			cv::Point2f original_point(x1, y1);

			cv::Point2i top_left((int)(original_point.x), (int)(original_point.y)); //top left because of integer rounding

			//make sure the point is actually inside the original image
			if (top_left.x < 0 || top_left.x > imgMat.cols - 2 || top_left.y < 0 || top_left.y > imgMat.rows - 2)
			{
				continue;
			}

			//bilinear interpolation
			float dx = original_point.x - top_left.x;
			float dy = original_point.y - top_left.y;

			float weight_tl = (1.0 - dx) * (1.0 - dy);
			float weight_tr = (dx) * (1.0 - dy);
			float weight_bl = (1.0 - dx) * (dy);
			float weight_br = (dx) * (dy);
			for (int k = 0; k < 3; k++)
			{
				uchar value = weight_tl * imgMat.at<cv::Vec3b>(top_left)[k] +
					weight_tr * imgMat.at<cv::Vec3b>(top_left.y, top_left.x + 1)[k] +
					weight_bl * imgMat.at<cv::Vec3b>(top_left.y + 1, top_left.x)[k] +
					weight_br * imgMat.at<cv::Vec3b>(top_left.y + 1, top_left.x + 1)[k];

				destImgMat.at<cv::Vec3b>(y, x)[k] = value;
			}

		}
	}
	return destImgMat;
}



void rectifyJoint::CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t& corners)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	//cout << "V2: " << V2 << endl;
	//cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}