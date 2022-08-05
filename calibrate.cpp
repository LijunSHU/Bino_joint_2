#include"calibrate.h"
calibrate::calibrate() {

}
calibrate::~calibrate() {

}
float calibrate::calError(std::vector<std::vector<cv::Point3f>> objectPoints, std::vector<std::vector<cv::Point2f>> imagePoints, cv::Mat cameraMatrix, cv::Mat distCoeffs,
	std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs) {
	float total_error = 0.0f;
	for (int i = 0; i < objectPoints.size(); i++) {
		std::vector<cv::Point2f> imgPoint;
		cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imgPoint);
		float error = cv::norm(imagePoints[i], imgPoint, cv::NORM_L2) / imgPoint.size();
		//float error = 0.0f;
		//for (int j = 0; j < imgPoint.size(); j++) {
		//	float err = std::sqrt(std::pow(imagePoints[i][j].x - imgPoint[j].x, 2) + std::pow(imagePoints[i][j].y - imgPoint[j].y, 2));
		//	error += err;
		//}
		//error = error / imgPoint.size();
		total_error += error;
	}
	float mean_error = total_error / objectPoints.size();
	return mean_error;
}
void calibrate::cameraCalibrate_(std::vector<cv::Mat>chessboards, cv::Size imgSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int boardSize,
	std::vector<std::vector<cv::Point3f>>& objectPoints, std::vector<std::vector<cv::Point2f>>& imagePoints) {
	objectPoints.clear();
	imagePoints.clear();
	for (int i = 0; i < chessboards.size(); i++) {
		std::vector<cv::Point3f> tempObjectPoints;
		std::vector<cv::Point2f> tempImagePoints;
		for (int row = 0; row < chessboards[i].rows; row++) {
			for (int col = 0; col < chessboards[i].cols; col++) {
				cv::Point2f p = chessboards[i].at<cv::Vec2f>(row, col);
				tempImagePoints.push_back(cv::Point2f(p.x, p.y));
				//fout << p.x << "\t" << p.y << "\n";
				tempObjectPoints.push_back(cv::Point3f(row * boardSize, col * boardSize, 0));
			}
		}
		objectPoints.push_back(tempObjectPoints);
		imagePoints.push_back(tempImagePoints);
	}
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, DBL_EPSILON);
	cv::calibrateCamera(objectPoints, imagePoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
	std::cout << "calibrate:" << cameraMatrix << "\n" << distCoeffs << std::endl;
	//float error = calError(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvecs, tvecs);
	//std::cout << error << std::endl;
}

void saveMap(cv::Mat mapLx, cv::Mat mapLy, cv::Mat mapRx, cv::Mat mapRy) {
	cv::Mat RRP(cv::Size(mapLx.cols, mapLx.rows), 0, cv::Scalar::all(0));
	cv::Mat RRP1(cv::Size(mapLx.cols, mapLx.rows), 0, cv::Scalar::all(0));
	cv::Mat mapLXY(RRP.size(), CV_16SC2, cv::Scalar::all(0));
	cv::Mat FLxy(RRP.size(), CV_16UC1, cv::Scalar::all(0));
	cv::Mat mapRXY(RRP.size(), CV_16SC2, cv::Scalar::all(0));
	cv::Mat FRxy(RRP.size(), CV_16UC1, cv::Scalar::all(0));
	for (int r = 0; r < RRP.rows; r++) {
		uchar* d = RRP.ptr<uchar>(r);
		for (int c = 0; c < RRP.cols; c++) {

			float u = mapRx.at<float>(r, c);
			float v = mapRy.at<float>(r, c);
			//std::cout << u << "\t" << v << std::endl;
			int iu = u;
			int iv = v;
			if (iu >= 0 && iu < RRP.cols - 2 && iv >= 0 && iv <= RRP.rows - 2 && u >= 0 && u < RRP.cols - 2 &&
				v >= 0 && v < RRP.rows - 2) {
				cv::Vec2s* x = mapRXY.ptr<cv::Vec2s>(r);
				x[c][0] = iu;
				x[c][1] = iv;
				double deta_u = u - (double)iu;
				double deta_v = v - (double)iv;
				int u_r = deta_u / (1.0 / 32.0) + 0.5;
				int v_c = deta_v / (1.0 / 32.0) + 0.5;
				//std::cout << u_r << "\t" << v_c << "\t";
				u_r = u_r > 31 ? 31 : u_r;
				v_c = v_c > 31 ? 31 : v_c;
				int num = (v_c) * 32 + u_r;
				ushort* x1 = FRxy.ptr<ushort>(r);
				//std::cout << num << std::endl;
				x1[c] = num;
			}
			else {
				cv::Vec2s* x = mapRXY.ptr<cv::Vec2s>(r);
				x[c][0] = 0;

				x[c][1] = 0;
				ushort* x1 = FRxy.ptr<ushort>(r);
				x1[c] = 0;
			}
		}
	}
	cv::FileStorage fs;
	std::string name1 = "Bino_stitching.xml";
	fs.open(name1, cv::FileStorage::WRITE);
	fs << "remapX2" << mapRXY;
	fs << "remapY2" << FRxy;
	if (FRxy.empty()) {
		std::cout << "empty" << std::endl;
	}
	for (int r = 0; r < RRP1.rows; r++) {
		uchar* d = RRP1.ptr<uchar>(r);
		for (int c = 0; c < RRP1.cols; c++) {
			float u = mapLx.at<float>(r, c);
			float v = mapLy.at<float>(r, c);
			int iu = u;
			int iv = v;
			if (iu >= 0 && iu < RRP1.cols - 2 && iv >= 0 && iv <= RRP1.rows - 2 && u >= 0 && u < RRP1.cols - 2 &&
				v >= 0 && v < RRP1.rows - 2) {
				cv::Vec2s* x = mapLXY.ptr<cv::Vec2s>(r);
				x[c][0] = iu;
				x[c][1] = iv;
				double deta_u = u - (double)iu;
				double deta_v = v - (double)iv;
				int u_r = deta_u / (1.0 / 32.0) + 0.5;
				int v_c = deta_v / (1.0 / 32.0) + 0.5;
				u_r = u_r > 31 ? 31 : u_r;
				v_c = v_c > 31 ? 31 : v_c;
				int num = (v_c) * 32 + u_r;
				ushort* x1 = FLxy.ptr<ushort>(r);
				x1[c] = num;
			}
			else {
				cv::Vec2s* x = mapLXY.ptr<cv::Vec2s>(r);
				x[c][0] = 0;
				x[c][1] = 0;
				ushort* x1 = FLxy.ptr<ushort>(r);
				x1[c] = 0;
			}
		}
	}
	fs << "remapX1" << mapLXY;
	fs << "remapY1" << FLxy;
	fs.release();
}

void calibrate::stereoCalibrate(std::vector<cv::Mat>leftChessboards, std::vector<cv::Mat>rightChessboards,
	cv::Mat leftCameraMatrix, cv::Mat leftdistCoeffs, cv::Mat rightCameraMatrix, cv::Mat rightdistCoeffs, cv::Size imgSize) {
	cv::Mat R, T, E, F, Rl, Rr, Pl, Pr, Q;
	cv::Rect validROIL, validROIR;
	int boardSize = 25;
	std::cout << "imgsize:" << imgSize << std::endl;
	//std::cout << "cameraMatrix:" << leftCameraMatrix << "\n" << rightCameraMatrix << std::endl;
	//std::cout << "distcoeffs:" << leftdistCoeffs << "\n" << rightdistCoeffs << std::endl;
	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<std::vector<cv::Point2f>> leftimagePoints, rightimagePoints;
	for (int i = 0; i < leftChessboards.size(); i++) {
		std::vector<cv::Point3f> tempObjectPoints;
		std::vector<cv::Point2f> tempImagePoints, tempImagePoints2;
		for (int row = 0; row < leftChessboards[i].rows; row++) {
			for (int col = 0; col < leftChessboards[i].cols; col++) {
				cv::Point2f p = leftChessboards[i].at<cv::Vec2f>(row, col);
				tempImagePoints.push_back(cv::Point2f(p.x, p.y));
				tempObjectPoints.push_back(cv::Point3f(row * boardSize, col * boardSize, 0));

				cv::Point2f p2 = rightChessboards[i].at<cv::Vec2f>(row, col);
				tempImagePoints2.push_back(cv::Point2f(p2.x, p2.y));
			}
		}
		objectPoints.push_back(tempObjectPoints);
		leftimagePoints.push_back(tempImagePoints);
		rightimagePoints.push_back(tempImagePoints2);
	}
	//std::cout << "...leftcameraMatrix:" << leftCameraMatrix << std::endl;
	//std::cout << "...rightcameraMatrix:" << rightCameraMatrix << std::endl;
	leftdistCoeffs.at<double>(0, 0) = 0.0; leftdistCoeffs.at<double>(0, 1) = 0.0;
	leftdistCoeffs.at<double>(0, 4) = 0.0;
	rightdistCoeffs.at<double>(0, 0) = 0.0; rightdistCoeffs.at<double>(0, 1) = 0.0;
	rightdistCoeffs.at<double>(0, 4) = 0.0;

	cv::stereoCalibrate(objectPoints, leftimagePoints, rightimagePoints, leftCameraMatrix, leftdistCoeffs, rightCameraMatrix, rightdistCoeffs, imgSize,
		R, T, E, F, cv::CALIB_FIX_INTRINSIC /*+  cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2 + cv::CALIB_FIX_K3 + cv::CALIB_FIX_S1_S2_S3_S4
           cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_SAME_FOCAL_LENGTH*/,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 1e-5));
	//cv::fisheye::stereoCalibrate(objectPoints, leftimagePoints, rightimagePoints, leftCameraMatrix, leftdistCoeffs, rightCameraMatrix,
	//	rightdistCoeffs, imgSize, R, T, cv::fisheye::CALIB_FIX_INTRINSIC);
	cv::FileStorage fs;
	std::string name1 = "extrinsics_parameters.xml";
	fs.open(name1, cv::FileStorage::WRITE);
	fs << "R" << R;
	fs << "T" << T;
	std::cout << "R" << R << std::endl;
	std::cout << "T" << T << std::endl;
	std::cout << R.size() << "\t" << T.size() << std::endl;

	std::cout << "after...leftcameraMatrix:" << leftCameraMatrix << std::endl;
	std::cout << "after...rightcameraMatrix:" << rightCameraMatrix << std::endl;
	std::cout << "distcoeffs:" << leftdistCoeffs << "\n" << rightdistCoeffs << std::endl;

	cv::stereoRectify(leftCameraMatrix, leftdistCoeffs, rightCameraMatrix, rightdistCoeffs, imgSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, -1, imgSize, &validROIL, &validROIR);
	//cv::fisheye::stereoRectify(leftCameraMatrix, leftdistCoeffs, rightCameraMatrix, rightdistCoeffs, imgSize, R, T, Rl, Rr, Pl, Pr,
	//	Q, cv::CALIB_ZERO_DISPARITY, imgSize);
	std::cout << "Rl:" << Rl << std::endl;
	std::cout << "Rr:" << Rr << std::endl;
	std::cout << "Pl:" << Pl << std::endl;
	std::cout << "Pr:" << Pr << std::endl;
	std::cout << "Q:" << Q << std::endl;
	fs << "Rl" << Rl;
	fs << "Rr" << Rr;
	fs << "Pl" << Pl;
	fs << "Pr" << Pr;
	fs << "Q" << Q;
	cv::Mat mapLx, mapLy, mapRx, mapRy;
	cv::Mat zero = cv::Mat::zeros(leftdistCoeffs.rows, leftdistCoeffs.cols, leftdistCoeffs.type());

	fs.release();

	name1 = "intrinsics_parameters.xml";
	fs.open(name1, cv::FileStorage::WRITE);
	fs << "rightCameraMatrix" << rightCameraMatrix;
	fs << "rightdistCoeffs" << rightdistCoeffs;
	fs << "leftCameraMatrix" << leftCameraMatrix;
	fs << "leftdistCoeffs" << leftdistCoeffs;
	fs.release();
	cv::initUndistortRectifyMap(leftCameraMatrix, leftdistCoeffs, Rl, Pl, imgSize, CV_32FC1, mapLx, mapLy);
	cv::initUndistortRectifyMap(rightCameraMatrix, rightdistCoeffs, Rr, Pr, imgSize, CV_32FC1, mapRx, mapRy);
	//cv::fisheye::initUndistortRectifyMap(leftCameraMatrix, leftdistCoeffs, Rl, Pl, imgSize, CV_32FC1, mapLx, mapLy);
	//cv::fisheye::initUndistortRectifyMap(rightCameraMatrix, rightdistCoeffs, Rr, Pr, imgSize, CV_32FC1, mapRx, mapRy);
	std::cout << mapLx.size() << "\t" << mapLy.size() << "\t" << mapRx.size() << std::endl;

	cv::Mat leftImage = cv::imread("./data/7.20/bino/1/left/3.bmp", -1);
	cv::Mat rightImage = cv::imread("./data/7.20/bino/1/right/3.bmp", -1);
	//cv::resize(leftImage, leftImage, cv::Size(1280, 720));
	//cv::resize(rightImage, rightImage, cv::Size(1280, 720));
	//cv::flip(rightImage, rightImage, 0);
	//cv::flip(rightImage, rightImage, 1);
	cv::remap(leftImage, leftImage, mapLx, mapLy, cv::INTER_LINEAR);
	cv::remap(rightImage, rightImage, mapRx, mapRy, cv::INTER_LINEAR);
	cv::imwrite("./data/cy/remapleft_3.png", leftImage);
	cv::imwrite("./data/cy/remapright_3.png", rightImage);
	//cv::Mat uLeft, uRight;
	//cv::undistort(leftImage, uLeft, leftCameraMatrix, leftdistCoeffs);
	//cv::undistort(rightImage, uRight, rightCameraMatrix, rightdistCoeffs);

	//cv::imwrite("./data/cy/undistortleft.png", uLeft);
	//cv::imwrite("./data/cy/undistortright.png", uRight);


	float f = Q.at<double>(2, 3);
	//std::cout << "..........f:" << f << std::endl;
	//std::cout << "left matrix:\n" << leftCameraMatrix << std::endl;
	//std::cout << "left distcoeffs:\n" << leftdistCoeffs << std::endl;
	//std::cout << "right matirx:\n" << rightCameraMatrix << std::endl;
	//std::cout << "right distcoeffs:\n" << rightdistCoeffs << std::endl;
	std::cout << "Q type:" << Q.type() << std::endl;
	//saveMap(mapLx, mapLy, mapRx, mapRy);
	//std::vector<cv::Point2f> keyPoints;
	//cv::Mat img_1 = cv::imread("IR_RAW8_1_768_1024_172412.png", 0);
	//cv::resize(img_1, img_1, cv::Size(384, 512));
	//depth2ori(img_1, leftCameraMatrix, leftdistCoeffs, Rl, Pl, 100, 200, 5, 5, keyPoints);
	std::vector<cv::Point2f> featrue1, feature2;
	for (int i = 0; i < leftimagePoints.size(); i++) {
		for (int j = 0; j < leftimagePoints[i].size(); j++) {
			featrue1.push_back(cv::Point2f(leftimagePoints[i][j].x, leftimagePoints[i][j].y));
			feature2.push_back(cv::Point2f(rightimagePoints[i][j].x, rightimagePoints[i][j].y));
		}
	}

	cv::Mat h = cv::findHomography(featrue1, feature2, cv::RANSAC);
	cv::Mat h2 = cv::findHomography(feature2, featrue1, cv::RANSAC);
	std::cout << "h:" << h.type();
	cv::FileStorage f2;
	f2.open("homography.xml", cv::FileStorage::WRITE);
	f2 << "H1" << h;
	f2 << "H2" << h2;
	f2.release();

}


void calibrate::stereoCalibrate2(std::vector<std::vector<cv::Point2f>> leftimagePoints, std::vector<std::vector<cv::Point2f>> rightimagePoints, std::vector<std::vector<
	cv::Point3f>> objectPoints,cv::Mat leftCameraMatrix, cv::Mat leftdistCoeffs, cv::Mat rightCameraMatrix, cv::Mat rightdistCoeffs, cv::Size imgSize) {
	cv::Mat R, T, E, F, Rl, Rr, Pl, Pr, Q;
	cv::Rect validROIL, validROIR;
	cv::stereoCalibrate(objectPoints, leftimagePoints, rightimagePoints, leftCameraMatrix, leftdistCoeffs, rightCameraMatrix, rightdistCoeffs, imgSize,
		R, T, E, F, cv::CALIB_FIX_INTRINSIC /*+  cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2 + cv::CALIB_FIX_K3 + cv::CALIB_FIX_S1_S2_S3_S4*/,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 1e-5));
	cv::FileStorage fs;
	std::string name1 = "extrinsics_parameters.xml";
	fs.open(name1, cv::FileStorage::WRITE);
	fs << "R" << R;
	fs << "T" << T;
	std::cout << "R" << R << std::endl;
	std::cout << "T" << T << std::endl;
	std::cout << R.size() << "\t" << T.size() << std::endl;

	std::cout << "after...leftcameraMatrix:" << leftCameraMatrix << std::endl;
	std::cout << "after...rightcameraMatrix:" << rightCameraMatrix << std::endl;
	std::cout << "distcoeffs:" << leftdistCoeffs << "\n" << rightdistCoeffs << std::endl;

	cv::stereoRectify(leftCameraMatrix, leftdistCoeffs, rightCameraMatrix, rightdistCoeffs, imgSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, -1, imgSize, &validROIL, &validROIR);
	std::cout << "Rl:" << Rl << std::endl;
	std::cout << "Rr:" << Rr << std::endl;
	std::cout << "Pl:" << Pl << std::endl;
	std::cout << "Pr:" << Pr << std::endl;
	std::cout << "Q:" << Q << std::endl;
	fs << "Rl" << Rl;
	fs << "Rr" << Rr;
	fs << "Pl" << Pl;
	fs << "Pr" << Pr;
	fs << "Q" << Q;
	cv::Mat mapLx, mapLy, mapRx, mapRy;
	cv::Mat zero = cv::Mat::zeros(leftdistCoeffs.rows, leftdistCoeffs.cols, leftdistCoeffs.type());

	fs.release();

	name1 = "intrinsics_parameters.xml";
	fs.open(name1, cv::FileStorage::WRITE);
	fs << "rightCameraMatrix" << rightCameraMatrix;
	fs << "rightdistCoeffs" << rightdistCoeffs;
	fs << "leftCameraMatrix" << leftCameraMatrix;
	fs << "leftdistCoeffs" << leftdistCoeffs;
	fs.release();
	cv::initUndistortRectifyMap(leftCameraMatrix, zero, Rl, Pl, imgSize, CV_32FC1, mapLx, mapLy);
	cv::initUndistortRectifyMap(rightCameraMatrix, zero, Rr, Pr, imgSize, CV_32FC1, mapRx, mapRy);
	std::cout << mapLx.size() << "\t" << mapLy.size() << "\t" << mapRx.size() << std::endl;

	//cv::Mat leftImage = cv::imread("./data/35.bmp", -1);
	//cv::Mat rightImage = cv::imread("./data/36.bmp", -1);
	//cv::resize(leftImage, leftImage, cv::Size(1280, 720));
	//cv::resize(rightImage, rightImage, cv::Size(1280, 720));
	//cv::flip(rightImage, rightImage, 0);
	//cv::flip(rightImage, rightImage, 1);

	cv::Mat leftImage = cv::imread("./data/left_image_1.png", -1);
	cv::Mat rightImage = cv::imread("./data/right_image_1.png", -1);
	cv::remap(leftImage, leftImage, mapLx, mapLy, cv::INTER_LINEAR);
	cv::remap(rightImage, rightImage, mapRx, mapRy, cv::INTER_LINEAR);
	cv::imwrite("rectifyleft.png", leftImage);
	cv::imwrite("rectifyright.png", rightImage);


	float f = Q.at<double>(2, 3);
	//std::cout << "..........f:" << f << std::endl;
	//std::cout << "left matrix:\n" << leftCameraMatrix << std::endl;
	//std::cout << "left distcoeffs:\n" << leftdistCoeffs << std::endl;
	//std::cout << "right matirx:\n" << rightCameraMatrix << std::endl;
	//std::cout << "right distcoeffs:\n" << rightdistCoeffs << std::endl;
	std::cout << "Q type:" << Q.type() << std::endl;
	//saveMap(mapLx, mapLy, mapRx, mapRy);
	//std::vector<cv::Point2f> keyPoints;
	//cv::Mat img_1 = cv::imread("IR_RAW8_1_768_1024_172412.png", 0);
	//cv::resize(img_1, img_1, cv::Size(384, 512));
	//depth2ori(img_1, leftCameraMatrix, leftdistCoeffs, Rl, Pl, 100, 200, 5, 5, keyPoints);
}