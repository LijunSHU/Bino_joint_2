//#include <iostream>
//#include<opencv2\opencv.hpp>
//#include<opencv2\xfeatures2d.hpp>
//#include<vector>
//
//using namespace std;
//using namespace cv;
//using namespace xfeatures2d;
//
//void linesapce(Mat& image, float begin, float finish, int w1, Mat& mask);
//void generate_mask(Mat& img, Mat& mask);
//
//int main()
//{
//	Mat room1 = imread("./data/35.bmp");
//	Mat room2 = imread("./data/36.bmp");
//	resize(room1, room1, Size((int)(room1.cols / 2), (int)(room1.rows / 2)), 0.5, 0.5);
//	resize(room2, room2, Size((int)(room2.cols / 2), (int)(room2.rows / 2)), 0.5, 0.5);
//	if (room1.empty() == true || room2.empty() == true) {
//		cout << "error" << endl;
//		return -1;
//	}
//	//imshow("room1", room1);
//	//imshow("room2", room2);
//	//创建ORB特征点提取对象，设置提取点数
//	auto orb = ORB::create(500);
//	//存放提取的特征点
//	vector<KeyPoint> kpts_orb_room1;
//	vector<KeyPoint> kpts_orb_room2;
//	//存放特征点描述子
//	Mat dec_orb_room1, dec_orb_room2;
//	//特征点提取和描述子计算
//	orb->detectAndCompute(room1, Mat(), kpts_orb_room1, dec_orb_room1);
//	orb->detectAndCompute(room2, Mat(), kpts_orb_room2, dec_orb_room2);
//
//
//	/*---------FLANN 匹配-------------*/
//	/*
//	if ((dec_orb_book.type() != CV_32F) && (dec_orb_book_in_scene.type() != CV_32F)) {
//		dec_orb_book.convertTo(dec_orb_book, CV_32F);
//		dec_orb_book_in_scene.convertTo(dec_orb_book_in_scene, CV_32F);
//	}
//	auto flann_matcher = FlannBasedMatcher::create();
//	vector<DMatch> matches_flann;
//	flann_matcher->match(dec_orb_book, dec_orb_book_in_scene, matches_flann);
//	Mat result_flann;
//	drawMatches(book, kpts_orb_book, book_in_scene, kpts_orb_book_in_scene, matches_flann, result_flann);
//	imshow("result_flann", result_flann);
//	*/
//	/*--------------------------------*/
//
//
//	/*---------暴力 匹配-------------*/
//
//	//创建暴力匹配子对象
//	auto bf_matcher = BFMatcher::create(NORM_HAMMING, false);
//	//存放描述子匹配关系
//	vector<DMatch> matches_bf;
//	//特征点描述子匹配
//	bf_matcher->match(dec_orb_room1, dec_orb_room2, matches_bf);
//
//
//	//特征点筛选
//	float good_rate = 0.025f;
//	int num_good_matchs = matches_bf.size() * good_rate;
//	std::sort(matches_bf.begin(), matches_bf.end());
//	matches_bf.erase(matches_bf.begin() + num_good_matchs, matches_bf.end());
//	//绘制筛选后匹配结果
//	Mat result_bf;
//	drawMatches(room1, kpts_orb_room1, room2, kpts_orb_room2, matches_bf, result_bf);
//	//获取两张图的特征点
//	vector<Point2f>room1_points;
//	vector<Point2f>room2_points;
//	for (size_t t = 0; t < matches_bf.size(); t++) {
//		room1_points.push_back(kpts_orb_room1[matches_bf[t].queryIdx].pt);
//		room2_points.push_back(kpts_orb_room2[matches_bf[t].trainIdx].pt);
//	}
//	//根据对应的特征点获取从demo->scene的变换矩阵
//	Mat h = findHomography(room2_points, room1_points, RANSAC);
//
//	//获取全景图的大小
//	int mix_high = max(room1.rows, room2.rows);
//	int mix_widht = room1.cols + room2.cols;
//	Mat mix_room_01 = Mat::zeros(Size(mix_widht, mix_high), CV_8UC3);
//	Rect roi;
//	roi.x = 0;
//	roi.y = 0;
//	roi.width = room1.cols;
//	roi.height = room1.rows;
//
//	//获取room2与room1的对齐图像
//	room1.copyTo(mix_room_01(roi));
//	//imshow("mix_room_01", mix_room_01);
//	Mat mix_room_02;
//	warpPerspective(room2, mix_room_02, h, Size(mix_widht, mix_high));
//	//imshow("mix_room2", mix_room_02);
//
//	//计算重合区域mask
//	Mat mask = Mat::zeros(Size(mix_widht, mix_high), CV_8UC1);
//	generate_mask(mix_room_02, mask);
//	//imshow("mask", mask);
//
//	//创建遮罩层并根据mask完成权重初始化
//	Mat mask1 = Mat::ones(Size(mix_widht, mix_high), CV_32FC1);
//	Mat mask2 = Mat::ones(Size(mix_widht, mix_high), CV_32FC1);
//
//
//	//room1 mask1
//	linesapce(mask1, 1, 0, room1.cols, mask);
//	//imshow("mask1", mask1);
//	//room2 mask2
//	linesapce(mask2, 0, 1, room2.cols, mask);
//	//imshow("mask2", mask2);
//
//	//左侧融合
//	Mat m1;
//	vector<Mat> mv;
//	mv.push_back(mask1);
//	mv.push_back(mask1);
//	mv.push_back(mask1);
//	merge(mv, m1);
//	mix_room_01.convertTo(mix_room_01, CV_32F);
//	multiply(mix_room_01, m1, mix_room_01);
//
//	//右侧融合
//	Mat m2;
//	mv.clear();
//	mv.push_back(mask2);
//	mv.push_back(mask2);
//	mv.push_back(mask2);
//	merge(mv, m2);
//	mix_room_02.convertTo(mix_room_02, CV_32F);
//	multiply(mix_room_02, m2, mix_room_02);
//
//	//合并
//	Mat mix_room;
//	add(mix_room_01, mix_room_02, mix_room);
//	mix_room.convertTo(mix_room, CV_8U);
//	imshow("mix_room", mix_room);
//
//	imshow("result_bf", result_bf);
//	waitKey(0);
//	return 0;
//}
//
//void generate_mask(Mat& img, Mat& mask) {
//	int w = img.cols;
//	int h = img.rows;
//	for (int row = 0; row < h; row++) {
//		for (int col = 0; col < w; col++) {
//			Vec3b p = img.at<Vec3b>(row, col);
//			int b = p[0];
//			int g = p[1];
//			int r = p[2];
//			if (b == g && g == r && r == 0) {
//				mask.at<uchar>(row, col) = 255;
//			}
//
//		}
//
//	}
//}
//void linesapce(Mat& image, float begin, float finish, int w1, Mat& mask) {
//	int offstx = 0;
//	float interval = 0;
//	float delta = 0;
//	for (int i = 0; i < image.rows; i++) {
//		offstx = 0;
//		interval = 0;
//		delta = 0;
//		for (int j = 0; j < image.cols; j++) {
//			int pv = mask.at<uchar>(i, j);
//			if (pv == 0 && offstx == 0) {
//				offstx = j;
//				delta = w1 - offstx;
//				interval = (finish - begin) / (delta - 1);
//				image.at<float>(i, j) = begin + (j - offstx) * interval;
//			}
//			else if (pv == 0 && offstx > 0 && (j - offstx) < delta) {
//				image.at<float>(i, j) = begin + (j - offstx) * interval;
//			}
//
//		}
//	}
//}