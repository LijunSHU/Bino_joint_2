#include"CornerDetAC.h"
#include"ChessboradStruct.h"
#include<io.h>
#include"calibrate.h"
#include"pictureJoint.h"
#include"rectifyJoint.h"
//bool getFiles(std::string path, std::vector<std::string>& files) {
//	intptr_t hFile = 0;
//	struct _finddata_t fileinfo;
//	std::string p;
//	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1) {
//		do {
//			if ((fileinfo.attrib & _A_SUBDIR)) {
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
//					getFiles(p.assign(fileinfo.name), files);
//				}
//			}
//			else {
//				files.push_back(p.assign(fileinfo.name));
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//	if (files.size() == 0)
//		return false;
//	return true;
//}

void remap_2(cv::Mat& image, cv::Mat XYmap, cv::Mat Fxy_map, cv::Mat Wmap) {
	if (XYmap.empty() || Fxy_map.empty() || Wmap.empty()) {
		std::cout << "mat empty" << std::endl;
		cv::waitKey(0);
		return;
	}
	cv::Mat temp(image.size(), image.type(), cv::Scalar::all(0));
	for (int i = 0; i < temp.rows; i++) {
		uchar* d = temp.ptr<uchar>(i);
		cv::Vec2s* xy = XYmap.ptr<cv::Vec2s>(i);
		ushort* d_fxy = Fxy_map.ptr<ushort>(i);
		for (int j = 0; j < temp.cols - 1; j++) {
			int xx = xy[j][0], yy = xy[j][1];
			if (xx == 0 && yy == 0)
				continue;
			uchar* in = image.ptr<uchar>(yy);
			uchar* in1 = image.ptr<uchar>(yy + 1);
			float fx1 = in[xx], fx2 = in[xx + 1], fx3 = in1[xx], fx4 = in1[xx + 1];
			ushort fxy_L = d_fxy[j] * 4;
			ushort w1 = Wmap.ptr<ushort>(0)[fxy_L], w2 = Wmap.ptr<ushort>(0)[fxy_L + 1], w3 = Wmap.ptr<ushort>(0)[fxy_L + 2],
				w4 = Wmap.ptr<ushort>(0)[fxy_L + 3];
			d[j] = (uchar)((fx1 * w1 + fx2 * w2 + fx3 * w3 + fx4 * w4 + 16384) / 32768);
		}
	}
	image = temp;
}

bool chessboard2Points(std::vector<cv::Mat> chessboards, Corners corner, std::vector<cv::Mat>& result) {
	//if (chessboards.size() == 0 || chessboards.size() < 2) {
	//	std::cout << "棋盘格数量不足" << std::endl;
	//	return false;
	//}
	int width, hight;
	int index, count = 0;
	cv::Point2f point;
	for (int i = 0; i < chessboards.size(); i++) {
		width = chessboards[i].cols;
		hight = chessboards[i].rows;
		cv::Mat temp = cv::Mat::zeros(hight, width, CV_32FC2);
		if (width * hight < 4)
			continue;
		for (int row = 0; row < chessboards[i].rows; row++) {
			for (int col = 0; col < chessboards[i].cols; col++) {
				index = chessboards[i].at<int>(row, col);
				point = corner.p[index];
				temp.at<cv::Vec2f>(row, col)[0] = point.x;
				temp.at<cv::Vec2f>(row, col)[1] = point.y;
			}
		}
		result.push_back(temp);
		count++;
	}
	//if (count < 2) {
	//	std::cout << "棋盘格角点数量不足" << std::endl;
	//	return false;
	//}
	return true;
}
//标定与矫正
void main111() {
#if 0
	CornerDetAC detAC;
	ChessboradStruct chessStruct;
	std::string path1 = "./data/7.20/mono/left/";
	std::string path2 = "./data/7.20/mono/right/";
	std::vector<std::string>leftfiles, rightfiles;
	getFiles(path1, leftfiles);
	getFiles(path2, rightfiles);
	std::vector<cv::Mat> aLeftChessboards, aRightChessboards;
	std::cout << leftfiles.size() << std::endl;
	cv::Size cornerSize(11,8);
	std::vector<std::vector<cv::Point2f>> leftImagePoints_, rightImagePoints_;
	std::vector<std::vector<cv::Point3f>> leftObjectPoints, rightObjectPoints;
	cv::Size imgSize;
	imgSize.height = 1440;
	imgSize.width = 2560;
	for (int i = 0; i < leftfiles.size(); i++) {
		//std::cout << path1 << leftfiles[i] << std::endl;
		cv::Mat leftImage = cv::imread(path1 + leftfiles[i], -1);
		cv::Mat rightImage = cv::imread(path2 + rightfiles[i], -1);
#if 0 //分为利用生长棋盘格和opencv棋盘格对单张图片进行标定的问题
		Corners leftPoints, rightPoints;
		
		detAC.detectCorners(leftImage, leftPoints, 0.01, true, cornerSize, 1);
		detAC.detectCorners(rightImage, rightPoints, 0.01, true, cornerSize, 1);

		std::vector<cv::Mat>leftChessboards, rightChessboards, leftChessboard_, rightChessboard_;
		std::cout << "check:" << leftPoints.p.size() << "\t" << rightPoints.p.size() << std::endl;
		if (leftPoints.p.size() < 11 * 8 || rightPoints.p.size() < 11 * 8) {
			std::cout << "........." << i << "............." << std::endl;
			continue;

		}
		chessStruct.chessboardsFromCorners(leftPoints, leftChessboards, 0.6);
		chessStruct.chessboardsFromCorners(rightPoints, rightChessboards, 0.6);
		std::cout << leftChessboards[0].size() << "\t" << rightChessboards[0].size() << std::endl;
		//chessStruct.drawchessboard(leftImage, leftPoints, leftChessboards);

		if (leftChessboards.size() != rightChessboards.size()) {
			std::cout << "here 1" << std::endl;
			continue;
		}
		std::cout << "here 2" << std::endl;
		chessboard2Points(leftChessboards, leftPoints, leftChessboard_);
		chessboard2Points(rightChessboards, rightPoints, rightChessboard_);
		cv::Mat t1 = leftImage.clone();
		cv::Mat t2 = rightImage.clone();
		//chessStruct.drawchessboard_("left_b", t1, leftChessboard_);
		//chessStruct.drawchessboard_("right_b", t2, rightChessboard_);

		chessStruct.matchCorners(leftChessboard_);
		chessStruct.matchCorners(rightChessboard_);
		std::cout << "chessboards:" << leftChessboard_[0].size() << "\t" << rightChessboard_[0].size() << std::endl;

		aLeftChessboards.push_back(leftChessboard_[0]);
		aRightChessboards.push_back(rightChessboard_[0]);
		chessStruct.drawchessboard_("left", leftImage, leftChessboard_);
		chessStruct.drawchessboard_("right", rightImage, rightChessboard_);
#else  
		cv::GaussianBlur(leftImage, leftImage, cv::Size(3, 3), 1.5);
		cv::GaussianBlur(rightImage, rightImage, cv::Size(3, 3), 1.5);

		std::vector<cv::Point2f> leftImagePoints, rightImagePoints;
		cv::findChessboardCorners(leftImage, cornerSize, leftImagePoints);
		//cv::drawChessboardCorners(leftImage, cornerSize, leftImagePoints, true);

		//for (int k = 0; k < leftImagePoints.size(); k++) {
		//	cv::putText(leftImage, std::to_string(k), cv::Point(leftImagePoints[k]), 0, 0.5, cv::Scalar(255), 1);
		//}
		//cv::namedWindow("i", cv::WINDOW_NORMAL);
		//cv::imshow("i", leftImage);
		//cv::waitKey(0);

		cv::findChessboardCorners(rightImage, cornerSize, rightImagePoints);
		//cv::drawChessboardCorners(rightImage, cornerSize, rightImagePoints, true);
		if (leftImagePoints.size() == 0 || rightImagePoints.size() == 0)
			continue;
		leftImagePoints_.push_back(leftImagePoints);
		rightImagePoints_.push_back(rightImagePoints);

	/*	cv::namedWindow("i", cv::WINDOW_NORMAL);
		cv::imshow("i", leftImage);

		cv::namedWindow("j", cv::WINDOW_NORMAL);
		cv::imshow("j", rightImage);

		cv::waitKey(0);*/
#endif
	}


	calibrate cal;
	cv::Mat leftCameraMatrix, leftDistcoeffs, rightCameraMatrix, rightDistcoeffs;
#if 0
	//std::vector<std::vector<cv::Point3f>> leftObjectPoints, rightObjectPoints;
	std::vector<std::vector<cv::Point2f>> leftImagePoints, rightImagePoints;
	cal.cameraCalibrate_(aLeftChessboards, imgSize, leftCameraMatrix, leftDistcoeffs, 25, leftObjectPoints, leftImagePoints);
	cal.cameraCalibrate_(aRightChessboards, imgSize, rightCameraMatrix, rightDistcoeffs, 25, rightObjectPoints, rightImagePoints);

#else
	//std::cout << "size:" << rightImagePoints_.size() << std::endl;
	for (int i = 0; i < rightImagePoints_.size(); i++) {
		std::vector<cv::Point3f> objectTemp;
		for (int row = 0; row < cornerSize.height; row++) {
			for (int col = 0; col < cornerSize.width; col++) {
				objectTemp.push_back(cv::Point3f(row * 25, col * 25, 0));
			}
		}
		leftObjectPoints.push_back(objectTemp);
	}
	//std::cout << leftObjectPoints.size() << "\t" << leftImagePoints_.size() << "\t" << rightImagePoints_.size() << std::endl;
	std::vector<cv::Mat> rvecs, tvecs, rvecs2, tvecs2;
	cv::calibrateCamera(leftObjectPoints, leftImagePoints_, imgSize, leftCameraMatrix, leftDistcoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);
	//cv::fisheye::calibrate(leftObjectPoints, leftImagePoints_, imgSize, leftCameraMatrix, leftDistcoeffs, rvecs, tvecs);
	//float err = cal.calError(leftObjectPoints, leftImagePoints_, leftCameraMatrix, leftDistcoeffs, rvecs, tvecs);
	std::cout << leftCameraMatrix << "\n" << leftDistcoeffs << std::endl;
	//std::cout << "err:" << err << std::endl;
	
	cv::calibrateCamera(leftObjectPoints, rightImagePoints_, imgSize, rightCameraMatrix, rightDistcoeffs, rvecs2, tvecs2);

	//cv::fisheye::calibrate(leftObjectPoints, rightImagePoints_, imgSize, rightCameraMatrix, rightDistcoeffs, rvecs2, tvecs2);
	//err = cal.calError(leftObjectPoints, rightImagePoints_, rightCameraMatrix, rightDistcoeffs, rvecs2, tvecs2);
	std::cout << rightCameraMatrix << "\n" << rightDistcoeffs << std::endl;

	//std::cout << ".....err:" << err << std::endl;
	//cv::Mat R1, t1;
	//cv::Rodrigues(rvecs[0], R1);
	//cv::Rodrigues(tvecs[0], t1);

	//std::cout << R1.size() << std::endl;
	//cv::Mat h = leftCameraMatrix * R1 * t1;
	//cv::Mat h_inv;
	//cv::invert(h, h_inv);
	//std::cout << "h_inv:" << h_inv.rows << "\t" << h_inv.cols << std::endl;
	//cv::Mat img1 = cv::Mat::zeros(3, 1, CV_64FC1);
	//img1.at<double>(0, 0) = leftImagePoints_[0][0].x;
	//img1.at<double>(1, 0) = leftImagePoints_[0][0].y;
	//img1.at<double>(2, 0) = 1;
	//std::cout << img1.rows << "\t" << img1.cols << std::endl;
	//cv::Mat result = h_inv * img1;
	//std::cout << "result:" << result << std::endl;
	//cv::Mat img2 = cv::Mat::zeros(3, 1, CV_64FC1);
	//img2.at<double>(0, 0) = leftImagePoints_[1][0].x;
	//img2.at<double>(1, 0) = leftImagePoints_[1][0].y;
	//img2.at<double>(2, 0) = 1;
	//cv::Mat result2 = h_inv * img2;
	//double dis = std::sqrt(std::pow(result.at<double>(0, 0) - result2.at<double>(0, 0), 2) + std::pow(result.at<double>(1, 0) - result2.at<double>(1, 0), 2));
	//std::cout << "dis:" << dis << std::endl;
	



	//cv::Mat uLeftImg = cv::imread("./data/35.bmp", 0);
	//cv::Mat uRightImg = cv::imread("./data/35_1.bmp", 0);
	//cv::Mat resultL, resultR;
	//cv::undistort(uLeftImg, resultL, leftCameraMatrix, leftDistcoeffs);
	//cv::undistort(uRightImg, resultR, rightCameraMatrix, rightDistcoeffs);
	//cv::imwrite("ul.png", resultL);
	//cv::imwrite("ur.png", resultR);

	//std::cout << rightCameraMatrix << "\n" << rightDistcoeffs << std::endl;
	//cal.stereoCalibrate2(leftImagePoints_, rightImagePoints_, leftObjectPoints, leftCameraMatrix, leftDistcoeffs, rightCameraMatrix, rightDistcoeffs, imgSize);

#endif  //标定方法选择if 2
	std::string path3 = "./data/7.20/bino/1/left/";
	std::string path4 = "./data/7.20/bino/1/right/";
	std::vector<std::string> files3, files4;
	getFiles(path3, files3);
	getFiles(path4, files4);
	std::vector<cv::Mat> calLeftChessboards, calRightChessboards;
	std::cout << "files:" << files3.size() << std::endl;
	for (int i = 0; i < files3.size(); i++) {
		cv::Mat bLeft = cv::imread(path3 + files3[i], -1);
		cv::Mat bRight = cv::imread(path4 + files4[i], -1);
		std::cout << files3[i] << "\t" << files4[i] << std::endl;
		//cv::resize(bLeft, bLeft, cv::Size(1280, 720));
		//cv::resize(bRight, bRight, cv::Size(1280, 720));
		//cv::flip(bRight, bRight, 0);
		//cv::flip(bRight, bRight, 1);


		//Corners bLeftCorners, bRightCorners;
		//std::vector<cv::Mat>bLeftChessboards, bRightChessboards, bLeftChessboards_, bRightChessboards_;
		//bLeftChessboards_.clear();
		//bRightChessboards_.clear();
		//detAC.detectCorners(bLeft, bLeftCorners, 0.01, true, cornerSize, 0);
		//detAC.detectCorners(bRight, bRightCorners, 0.01, true, cornerSize,0);
		//std::cout << "corners:" << bLeftCorners.p.size() << "\t" << bRightCorners.p.size() << std::endl;
		//chessStruct.chessboardsFromCorners(bLeftCorners, bLeftChessboards, 0.6);
		//chessStruct.chessboardsFromCorners(bRightCorners, bRightChessboards, 0.6);
		//chessboard2Points(bLeftChessboards, bLeftCorners, bLeftChessboards_);
		//chessboard2Points(bRightChessboards, bRightCorners, bRightChessboards_);
		//std::cout << ".............left match............" << std::endl;
		//chessStruct.matchCorners(bLeftChessboards_);
		//std::cout << ".............right match............" << std::endl;
		//std::cout << "bLeftChessboards_:" << bLeftChessboards_[0].rows << "\t" << bLeftChessboards_[0].cols << std::endl;
		//std::cout << "bRightChessboards_:" << bRightChessboards_[0].rows << "\t" << bRightChessboards_[0].cols << std::endl;
		//chessStruct.matchCorners(bRightChessboards_);
		//if (bLeftChessboards_.size() != bRightChessboards_.size())
		//	continue;
		//if (bLeftChessboards_[0].rows != bRightChessboards_[0].rows || bLeftChessboards_[0].cols != bRightChessboards_[0].cols)
		//	continue;

		//calLeftChessboards.push_back(bLeftChessboards_[0]);
		//calRightChessboards.push_back(bRightChessboards_[0]);

		//chessStruct.drawchessboard_("l", bLeft, bLeftChessboards_);
		//chessStruct.drawchessboard_("2", bRight, bRightChessboards_);


		std::vector<cv::Point2f> leftImages, rightImages;
		bool isTrue1 = cv::findChessboardCorners(bLeft, cornerSize, leftImages);
		bool isTrue2 = cv::findChessboardCorners(bRight, cornerSize, rightImages);
		std::cout << "corners Size:" << leftImages.size() << "\t" << rightImages.size() << "\t" << isTrue1 << "\t" << isTrue2 << std::endl;
		cv::Mat t1 = cv::Mat::zeros(cornerSize, CV_32FC2);
		cv::Mat t2 = cv::Mat::zeros(cornerSize, CV_32FC2);
		int tt = 0;
		for (int row = 0; row < cornerSize.height; row++) {
			for (int col = 0; col < cornerSize.width; col++) {
				t1.at<cv::Vec2f>(row, col) = cv::Point2f(leftImages[tt].x, leftImages[tt].y);
				t2.at<cv::Vec2f>(row, col) = cv::Point2f(rightImages[tt].x, rightImages[tt].y);
				tt++;
			}
		}
		cv::drawChessboardCorners(bLeft, cornerSize, leftImages, isTrue1);
		cv::drawChessboardCorners(bRight, cornerSize, rightImages, isTrue2);
		cv::namedWindow("bL", cv::WINDOW_NORMAL);
		cv::namedWindow("bR", cv::WINDOW_NORMAL);

		cv::imshow("bL", bLeft);
		cv::imshow("bR", bRight);
		cv::waitKey(0);
		calLeftChessboards.push_back(t1);
		calRightChessboards.push_back(t2);

	}
	cv::Mat leftCameraMatrix2, leftDistcoeffs2, rightCameraMatrix2, rightDistcoeffs2;
	std::cout << calLeftChessboards.size() << "\t" << calRightChessboards.size() << std::endl;
	if(calLeftChessboards.size() != 0 && calRightChessboards.size() == calLeftChessboards.size())
		cal.stereoCalibrate(calLeftChessboards, calRightChessboards,  leftCameraMatrix, leftDistcoeffs, rightCameraMatrix, rightDistcoeffs, imgSize);

#else
	cv::Mat leftImage = cv::imread("./data/35.bmp", -1);
	cv::Mat rightImage = cv::imread("./data/36.bmp", -1);
	cv::flip(rightImage, rightImage, 0);
	cv::flip(rightImage, rightImage, 1);
	cv::FileStorage fs;
	fs.open("Bino_stitching.xml", cv::FileStorage::READ);
	int width = leftImage.cols;
	int height = leftImage.rows;
	cv::Mat mapXYL(cv::Size(width, height), CV_16SC2);
	cv::Mat mapXYR(cv::Size(width, height), CV_16SC2);
	cv::Mat Wmap(cv::Size(1, 4096), CV_16SC1);
	cv::Mat fMapXYL(cv::Size(width, height), CV_16SC1);
	cv::Mat fMapXYR(cv::Size(width, height), CV_16SC1);

	fs["remapX1"] >> mapXYL;
	fs["remapY1"] >> fMapXYL;
	fs["remapX2"] >> mapXYR;
	fs["remapY2"] >> fMapXYR;
	fs.open("Tab2D_data.xml", cv::FileStorage::READ);
	fs["Tab2D"] >> Wmap;
	fs.release();

	remap_2(leftImage, mapXYL, fMapXYL, Wmap);
	remap_2(rightImage, mapXYR, fMapXYR, Wmap);

	cv::Mat canvas;
	double sf;
	int w, h;
	//sf = 600. / imageSize.height;

	//cv::Mat L = cv::imread(path + "IR_1_640_360_145944.png", -1);
	//cv::Mat R = cv::imread(path + "IR_2_640_360_145944.png", -1);
	w = cvRound(leftImage.cols);
	h = cvRound(leftImage.rows);
	canvas.create(h, w * 2, CV_8U);
	canvas.setTo(cv::Scalar(255, 0, 0));
	/*左图像画到画布上*/
	cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	cv::resize(leftImage, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);        //把图像缩放到跟canvasPart一样大小  

	///*右图像画到画布上*/
	canvasPart = canvas(cv::Rect(w, 0, w, h));                                      //获得画布的另一部分  
	cv::resize(rightImage, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR);

	/*画上对应的线条*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, cv::Point(0, i), cv::Point(canvas.cols, i), cv::Scalar(0, 255, 0), 1, 8);
	//cv::namedWindow("rectified", cv::WINDOW_NORMAL);
	//cv::imshow("rectified", canvas);
	cv::imwrite("recit.png", canvas);
	cv::waitKey(0);

#endif
}

//利用特征点提取方法拼接及矫正后拼接


void main22() {
#if 1
	pictureJoint pJ;
	cv::Mat leftImage = cv::imread("./data/7.27/scene/101.bmp");
	cv::Mat rightImage = cv::imread("./data/7.27/scene/100.bmp");
	//cv::imwrite("./data/35_1.bmp", rightImage);
	/*cv::imshow("1", leftImage);
	cv::imshow("2", rightImage);
	cv::waitKey(0);*/
	std::vector<cv::Mat> imgs;
	imgs.push_back(rightImage);
	imgs.push_back(leftImage);
	
	cv::Mat pano;
	bool isStatus = pJ.Image_Stitching(imgs, pano);
	if (isStatus) {
		cv::imshow("1", pano);
		cv::waitKey(0);
	}
	else {
		return;
	}


	//cv::Mat img1 = cv::imread("./data/7.20/bino/1/1.bmp");
	//cv::Mat img2 = cv::imread("./data/7.20/bino/1/2.bmp");
	//cv::Mat dstImg1 = img1.clone();
	//cv::Mat dstImg2 = img2.clone();
	//std::vector<cv::Point2f> leftImagePoints, rightImagePoints;
	//cv::findChessboardCorners(img1, cv::Size(11, 8), leftImagePoints);
	//cv::findChessboardCorners(img2, cv::Size(11, 8), rightImagePoints);
	////cv::drawChessboardCorners(img1, cv::Size(11, 8), leftImagePoints, true);
	////cv::drawChessboardCorners(img2, cv::Size(11, 8), rightImagePoints, true);

	////for (int i = 0; i < leftImagePoints.size(); i++) {
	////	cv::putText(img1, std::to_string(i), leftImagePoints[i], 0.5, 0.4, cv::Scalar(0, 255, 0));
	////	cv::putText(img2, std::to_string(i), rightImagePoints[i], 0.5, 0.4, cv::Scalar(0, 255, 0));

	////}
	//////cv::namedWindow("1", cv::WINDOW_NORMAL);
	//////cv::namedWindow("2", cv::WINDOW_NORMAL);
	//////cv::imshow("1", img1);
	//////cv::imshow("2", img2);
	////cv::imwrite("1.png", img1);
	////cv::imwrite("2.png", img2);

	////cv::waitKey(0);	

	//rectifyJoint retJ;

	//cv::Mat h = cv::findHomography(leftImagePoints, rightImagePoints, cv::RANSAC);
	//cv::Mat h1 = cv::findHomography(rightImagePoints, leftImagePoints, cv::RANSAC);
	//four_corners_t corners_1, corners_2;
	//retJ.CalcCorners(h, img1, corners_1);
	//std::cout << corners_1.left_top << "\t" << corners_1.left_bottom << "\t" << corners_1.right_top << "\t" << corners_1.right_bottom << std::endl;
	//retJ.CalcCorners(h1, img2, corners_2);
	//std::cout << corners_2.left_top << "\t" << corners_2.left_bottom << "\t" << corners_2.right_top << "\t" << corners_2.right_bottom << std::endl;
	////for (int i = 0; i < 4; i++) {
	//	cv::circle(img2, corners_1.left_bottom, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img1, corners_2.left_bottom, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img2, corners_1.right_bottom, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img1, corners_2.right_bottom, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img2, corners_1.left_top, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img1, corners_2.left_top, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img2, corners_1.right_top, 10, cv::Scalar(0, 255, 0), 1);
	//	cv::circle(img1, corners_2.right_top, 10, cv::Scalar(0, 255, 0), 1);

	////}
	//cv::namedWindow("1", cv::WINDOW_NORMAL);
	//cv::namedWindow("2", cv::WINDOW_NORMAL);

	//cv::imshow("1", img1);
	//cv::imshow("2", img2);
	//cv::waitKey(0);



	//int mRows = img2.rows;
	//if (img1.rows > img2.rows)
	//	mRows = img1.rows;
	//std::vector<cv::Point> vPoint1, vPoint2;
	//cv::Mat dst = cv::Mat::zeros(mRows, MAX(corners_1.right_top.x, corners_1.right_bottom.x), CV_8UC3);
	//warpPerspective(img2, dst, h, cv::Size(MAX(corners_1.right_top.x, corners_1.right_bottom.x), mRows));
	////cv::Mat half(dst, cv::Rect(0, 0, dst.cols, dst.rows));
	////img1.copyTo(half);
	//cv::Mat half;
	//img1.copyTo(half);

	//cv::Mat result = cv::Mat::zeros(mRows, dst.cols + half.cols, half.type());
	//std::cout << result.type() << "\t" << dst.type() << std::endl;
	////half.copyTo(result);
	//for (int i = 0; i < dst.rows; i++) {
	//	for (int j = 0; j < dst.cols; j++) {
	//		//std::cout << i << "\t" << j << "\t" << j + half.cols << std::endl;
	//		result.at<cv::Vec3b>(i, j + half.cols)[0] = dst.at<cv::Vec3b>(i, j)[0];
	//		result.at<cv::Vec3b>(i, j + half.cols)[1] = dst.at<cv::Vec3b>(i, j)[1];
	//		result.at<cv::Vec3b>(i, j + half.cols)[2] = dst.at<cv::Vec3b>(i, j)[2];

	//	}
	//}
	//for (int i = 0; i < half.rows; i++) {
	//	for (int j = 0; j < half.cols; j++) {
	//		//std::cout << i << "\t" << j << "\t" << j + half.cols << std::endl;
	//		result.at<cv::Vec3b>(i, j)[0] = half.at<cv::Vec3b>(i, j)[0];
	//		result.at<cv::Vec3b>(i, j)[1] = half.at<cv::Vec3b>(i, j)[1];
	//		result.at<cv::Vec3b>(i, j)[2] = half.at<cv::Vec3b>(i, j)[2];

	//	}
	//}



	//vPoint1.push_back(cv::Point((int)corners_1.left_top.x, (int)corners_1.left_top.y));
	//vPoint1.push_back(cv::Point((int)corners_1.right_top.x, (int)corners_1.right_top.y));
	//vPoint1.push_back(cv::Point((int)corners_1.right_bottom.x, (int)corners_1.right_bottom.y));
	//vPoint1.push_back(cv::Point((int)corners_1.left_bottom.x, (int)corners_1.left_bottom.y));

	//vPoint2.push_back(cv::Point((int)corners_2.left_top.x, (int)corners_2.left_top.y));
	//vPoint2.push_back(cv::Point((int)corners_2.right_top.x, (int)corners_2.right_top.y));
	//vPoint2.push_back(cv::Point((int)corners_2.right_bottom.x, (int)corners_2.right_bottom.y));
	//
	////drawingLine(img1, vPoint1);

	////drawingLine(img2, vPoint2);
	//cv::namedWindow("1", cv::WINDOW_NORMAL);
	//cv::namedWindow("2", cv::WINDOW_NORMAL);

	//cv::imshow("1", img1);
	//cv::imshow("2", img2);
	//cv::waitKey(0);
	//cv::waitKey(0);


	//cv::imwrite("dst.png", dst);
	
	std::cout << "end" << std::endl;
#else





	rectifyJoint reJoint;
	cv::Mat img1 = cv::imread("./data/7.27/1-1.png");
	cv::Mat img2 = cv::imread("./data/7.27/1-2.png");
	cv::Mat dstImg1 = img1.clone();
	cv::Mat dstImg2 = img2.clone();


	dstImg1 = reJoint.CylindricalWarp(img1);
	dstImg2 = reJoint.CylindricalWarp(img2);
	cv::imwrite("cylinderLeft.png", dstImg1);
	cv::imwrite("cylinderRight.png", dstImg2);


	cv::Point a = reJoint.getOffset(dstImg1, dstImg2);
	//cv::Point b = reJoint.getOffset(dstImg2, dstImg1);
	std::cout << "a:" << a << std::endl;
	std::cout << "b:" << std::endl;
	a.x = a.x;

	reJoint.linearFusion(dstImg1, dstImg2, a);
	//reJoint.cylinderOn(img1);

#endif
}


