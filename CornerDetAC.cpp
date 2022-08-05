#include"CornerDetAC.h"
using namespace cv;
using namespace std;
CornerDetAC::CornerDetAC() {

}
CornerDetAC::~CornerDetAC() {

}
bool CornerDetAC::detectCorners(cv::Mat& Src, Corners& mcorners, dtype scoreThreshold, bool isrefine, cv::Size cornerSize, int check) {
	if (Src.type() != 0) {
		cv::cvtColor(Src, Src, 6);
	}
	cv::Mat gray, imageNorm;
	if (check == 0) {
		cv::GaussianBlur(Src, gray, cv::Size(3, 3), 1.5);
		cv::equalizeHist(gray, gray);
	}
	else {
		cv::GaussianBlur(Src, gray, cv::Size(3, 3), 1.5);
	}

	cv::normalize(gray, imageNorm, 0, 1, cv::NORM_MINMAX, mtype);
	cv::Mat img_double;
	imageNorm.convertTo(img_double, mtype);
	double minVal, maxVal;
	int minIdx[2] = {}, maxIdx[2] = {};
	cv::minMaxIdx(img_double, &minVal, &maxVal, minIdx, maxIdx);
	img_double = (img_double - minVal) / (maxVal - minVal);
	cv::Mat cxy = img_double.clone();
	cv::Mat Ixy = img_double.clone();
	cv::Mat c45 = img_double.clone();
	cv::Mat Ix = img_double.clone();
	cv::Mat Iy = img_double.clone();
	cv::Mat I_45_45 = img_double.clone();
	secondDerivCornerMetric(img_double, 2, &cxy, &c45, &Ix, &Iy, &Ixy, &I_45_45);
	cv::Mat imgCorners = cxy + c45;
	std::vector<cv::Point2f> cornerPoints;
	nonMaximumSuppression(imgCorners, cornerPoints, 5, 0.25, 5);
	std::cout << "nonMaximumSuppression:" << cornerPoints.size() << std::endl;
	//cv:Mat show = Src.clone();
	//for (int i = 0; i < cornerPoints.size(); i++) {
	//	cv::circle(show, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 2, cv::Scalar(255), 1);
	//}
	//cv::namedWindow("n", cv::WINDOW_NORMAL);
	//cv::imshow("n", show);
	//cv::waitKey(0);
	cv::Mat imageDu(gray.size(), mtype);
	cv::Mat imageDv(gray.size(), mtype);
	cv::Mat img_angle = cv::Mat::zeros(gray.size(), mtype);
	cv::Mat img_weight = cv::Mat::zeros(gray.size(), mtype);
	getImageAngleAndWeight(imageNorm, imageDu, imageDv, img_angle, img_weight);
	int checknum = 0;
	if (isrefine == true) {
		refineCorners(imageNorm, cornerPoints, imageDu, imageDv, img_angle, img_weight, 5);
		if (cornerPoints.size() > 0) {
			for (int i = 0; i < cornerPoints.size(); i++) {
				if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0) {
					cornerPoints[i].x = 0; cornerPoints[i].y = 0;
					checknum++;
				}
			}
		}
	}

	std::cout << "refine:" << cornerPoints.size() - checknum << std::endl;
	std::vector<float>score;
	scoreCorners(imageNorm, img_angle, img_weight, cornerPoints, score);
	int nlen = cornerPoints.size();
	if (nlen > 0)
	{
		for (int i = 0; i < nlen; i++)
		{
			if (score[i] > scoreThreshold)
			{
				mcorners.p.push_back(cornerPoints[i]);
				mcorners.v1.push_back(cv::Vec2f(cornersEdge1[i][0], cornersEdge1[i][1]));
				mcorners.v2.push_back(cv::Vec2f(cornersEdge2[i][0], cornersEdge2[i][1]));
				mcorners.score.push_back(score[i]);
			}
		}
	}
	std::vector<cv::Vec2f> corners_n1(mcorners.p.size());
	if (corners_n1.size() < 11 * 8) {
		return false;
	}
	for (int i = 0; i < corners_n1.size(); i++)
	{
		if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0)
		{
			mcorners.v1[i] = -mcorners.v1[i];
		}
		corners_n1[i] = mcorners.v1[i];
		float flipflag = corners_n1[i][0] * mcorners.v2[i][0] + corners_n1[i][1] * mcorners.v2[i][1];
		if (flipflag > 0)
			flipflag = -1;
		else
			flipflag = 1;
		mcorners.v2[i] = flipflag * mcorners.v2[i];
	}
	//std::cout << "detect end" << std::endl;
	std::cout << "score size:" << mcorners.p.size() << std::endl;

	return true;
}
void CornerDetAC::secondDerivCornerMetric(cv::Mat I, int sigma, cv::Mat* cxy, cv::Mat* c45, cv::Mat* Ix, cv::Mat* Iy, cv::Mat* Ixy, cv::Mat* I_45_45) {
	cv::Mat Ig;
	cv::GaussianBlur(I, Ig, cv::Size(sigma * 5 + 1, sigma * 5 + 1), sigma, sigma);
	cv::Mat du = (cv::Mat_<float>(1, 3) << 1, 0, -1);
	cv::Mat dv = du.t();
	cv::filter2D(Ig, *Ix, Ig.depth(), du, cv::Point(-1, -1));
	cv::filter2D(Ig, *Iy, Ig.depth(), dv, cv::Point(-1, -1));
	cv::Mat I_45 = I.clone();
	cv::Mat I_n45 = I.clone();
	dtype cosPi4 = cos(CV_PI / 4);
	dtype cosNegPi4 = cos(-CV_PI / 4);
	dtype sinPi4 = sin(CV_PI / 4);
	dtype sinNegPi4 = sin(-CV_PI / 4);
	std::cout << "sin cos:" << cosPi4 << "\t" << cosNegPi4 << "\t" << sinPi4 << "\t" << sinNegPi4 << std::endl;
	for (int i = 0; i < I.rows; i++) {
		for (int j = 0; j < I.cols; j++) {
			I_45.at<dtype>(i, j) = Ix->at<dtype>(i, j) * cosPi4 + Iy->at<dtype>(i, j) * sinPi4;
			I_n45.at<dtype>(i, j) = Ix->at<dtype>(i, j) * cosNegPi4 + Iy->at<dtype>(i, j) * sinNegPi4;
		}
	}
	cv::Mat tt = I_45 + I_n45;
	cv::filter2D(*Ix, *Ixy, Ix->depth(), dv, cv::Point(-1, -1));
	cv::Mat I_45_x, I_45_y;
	cv::filter2D(I_45, I_45_x, I_45.depth(), du, cv::Point(-1, -1));
	cv::filter2D(I_45, I_45_y, I_45.depth(), dv, cv::Point(-1, -1));
	for (int i = 0; i < I.rows; i++)
	{
		for (int j = 0; j < I.cols; j++)
		{
			I_45_45->at<dtype>(i, j) = I_45_x.at<dtype>(i, j) * cosNegPi4 + I_45_y.at<dtype>(i, j) * sinNegPi4;
			cxy->at<dtype>(i, j) = sigma * sigma * fabs(Ixy->at<dtype>(i, j)) - 1.5 * sigma * (fabs(I_45.at<dtype>(i, j)) + fabs(I_n45.at<dtype>(i, j)));
			if (cxy->at<dtype>(i, j) < 0) cxy->at<dtype>(i, j) = 0;
			c45->at<dtype>(i, j) = sigma * sigma * fabs(I_45_45->at<dtype>(i, j)) - 1.5 * sigma * (fabs(Ix->at<dtype>(i, j)) + fabs(Iy->at<dtype>(i, j)));
			if (c45->at<dtype>(i, j) < 0) c45->at<dtype>(i, j) = 0;
		}
	}
}
void CornerDetAC::nonMaximumSuppression(cv::Mat& inputCorners, std::vector<cv::Point2f>& outputCorners, int patchSize, dtype threshold, int margin) {
	for (int i = margin + patchSize; i <= inputCorners.cols - (margin + patchSize + 1); i = i + patchSize + 1)
	{
		for (int j = margin + patchSize; j <= inputCorners.rows - (margin + patchSize + 1); j = j + patchSize + 1)
		{
			dtype maxVal = inputCorners.ptr<dtype>(j)[i];
			int maxX = i; int maxY = j;
			for (int m = i; m <= i + patchSize; m++)
			{
				for (int n = j; n <= j + patchSize; n++)
				{
					dtype temp = inputCorners.ptr<dtype>(n)[m];
					if (temp > maxVal)
					{
						maxVal = temp; maxX = m; maxY = n;
					}
				}
			}
			if (maxVal < threshold)
				continue;
			int flag = 0;
			for (int m = maxX - patchSize; m <= cv::min(maxX + patchSize, inputCorners.cols - margin - 1); m++)//二次检查
			{
				for (int n = maxY - patchSize; n <= cv::min(maxY + patchSize, inputCorners.rows - margin - 1); n++)
				{
					if (inputCorners.ptr<dtype>(n)[m] > maxVal && (m<i || m>i + patchSize || n<j || n>j + patchSize))
					{
						flag = 1;
						break;
					}
				}
				if (flag)
					break;
			}
			if (flag)
				continue;
			outputCorners.push_back(cv::Point(maxX, maxY));
			std::vector<dtype> e1(2, 0.0);
			std::vector<dtype> e2(2, 0.0);
			cornersEdge1.push_back(e1);
			cornersEdge2.push_back(e2);
		}
	}
}

void CornerDetAC::getImageAngleAndWeight(cv::Mat img, cv::Mat& imgDu, cv::Mat& imgDv, cv::Mat& imgAngle, cv::Mat& imgWeight) {
	cv::Mat sobelKernel(3, 3, mtype);
	cv::Mat sobelKernelTrs(3, 3, mtype);
	//sobelKernel.col(0).setTo(cv::Scalar(-1.0));
	//sobelKernel.col(1).setTo(cv::Scalar(0.0));
	//sobelKernel.col(2).setTo(cv::Scalar(1.0));
	sobelKernel.at<dtype>(0, 0) = -1;
	sobelKernel.at<dtype>(0, 1) = 0;
	sobelKernel.at<dtype>(0, 2) = 1;
	sobelKernel.at<dtype>(1, 0) = -2;
	sobelKernel.at<dtype>(1, 1) = 0;
	sobelKernel.at<dtype>(1, 2) = 2;
	sobelKernel.at<dtype>(2, 0) = -1;
	sobelKernel.at<dtype>(2, 1) = 0;
	sobelKernel.at<dtype>(2, 2) = 1;
	sobelKernelTrs = sobelKernel.t();
	std::cout << "soble kernel:" << sobelKernel << std::endl;
	cv::filter2D(img, imgDu, img.depth(), sobelKernel);
	cv::filter2D(img, imgDv, img.depth(), sobelKernelTrs);
	if (imgDu.size() != imgDv.size())
		return ;
	cv::cartToPolar(imgDu, imgDv, imgWeight, imgAngle, false);
	cv::GaussianBlur(imgWeight, imgWeight, cv::Size(3, 3), 1.5);
	for (int i = 0; i < imgDu.rows; i++)
	{
		for (int j = 0; j < imgDu.cols; j++)
		{
			dtype* dataAngle = imgAngle.ptr<dtype>(i);
			if (dataAngle[j] < 0)
				dataAngle[j] = dataAngle[j] + CV_PI;
			else if (dataAngle[j] > CV_PI)
				dataAngle[j] = dataAngle[j] - CV_PI;
		}
	}
}

void CornerDetAC::refineCorners(cv::Mat image, vector<Point2f>& cornors, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius) {
	// image dimensions
	int width = imgDu.cols;
	int height = imgDu.rows;
	for (int i = 0; i < cornors.size(); i++)
	{
		//extract current corner location
		int cu = cornors[i].x;
		int cv = cornors[i].y;
		// estimate edge orientations
		int startX, startY, ROIwidth, ROIheight;
		startX = MAX(cu - radius, (dtype)0);
		startY = MAX(cv - radius, (dtype)0);
		ROIwidth = MIN(cu + radius + 1, (dtype)width - 1) - startX;
		ROIheight = MIN(cv + radius + 1, (dtype)height - 1) - startY;

		Mat roiAngle, roiWeight;
		roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
		roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
		edgeOrientations(roiAngle, roiWeight, i);
		if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0 || cornersEdge2[i][0] == 0 && cornersEdge2[i][1] == 0)
			continue;

		//% corner orientation refinement %
		cv::Mat A1 = cv::Mat::zeros(cv::Size(2, 2), mtype);
		cv::Mat A2 = cv::Mat::zeros(cv::Size(2, 2), mtype);

		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				// robust refinement of orientation 1
				dtype t0 = abs(o.x * cornersEdge1[i][0] + o.y * cornersEdge1[i][1]);
				if (t0 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u) * addtion;
					Mat addtionv = imgDv.at<dtype>(v, u) * addtion;
					for (int j = 0; j < A1.cols; j++)
					{
						A1.at<dtype>(0, j) = A1.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A1.at<dtype>(1, j) = A1.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
				// robust refinement of orientation 2
				dtype t1 = abs(o.x * cornersEdge2[i][0] + o.y * cornersEdge2[i][1]);
				if (t1 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u) * addtion;
					Mat addtionv = imgDv.at<dtype>(v, u) * addtion;
					for (int j = 0; j < A2.cols; j++)
					{
						A2.at<dtype>(0, j) = A2.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A2.at<dtype>(1, j) = A2.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
			}//end for
		// set new corner orientation
		cv::Mat v1, foo1;
		cv::Mat v2, foo2;
		cv::eigen(A1, v1, foo1);
		cv::eigen(A2, v2, foo2);//计算特征值和特征向量
		cornersEdge1[i][0] = -foo1.at<dtype>(1, 0);
		cornersEdge1[i][1] = -foo1.at<dtype>(1, 1);
		cornersEdge2[i][0] = -foo2.at<dtype>(1, 0);
		cornersEdge2[i][1] = -foo2.at<dtype>(1, 1);//构造梯度模板

		//%  corner location refinement  %

		cv::Mat G = cv::Mat::zeros(cv::Size(2, 2), mtype);
		cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), mtype);
		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				//robust subpixel corner estimation
				if (u != cu || v != cv)// % do not consider center pixel
				{
					//compute rel. position of pixel and distance to vectors
					cv::Point2f w(u - cu, v - cv);
					float wvv1 = w.x * cornersEdge1[i][0] + w.y * cornersEdge1[i][1];
					float wvv2 = w.x * cornersEdge2[i][0] + w.y * cornersEdge2[i][1];

					cv::Point2f wv1(wvv1 * cornersEdge1[i][0], wvv1 * cornersEdge1[i][1]);
					cv::Point2f wv2(wvv2 * cornersEdge2[i][0], wvv2 * cornersEdge2[i][1]);
					cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
					cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
					dtype d1 = norm2d(vd1), d2 = norm2d(vd2);
					//if pixel corresponds with either of the vectors / directions
					if ((d1 < 3) && abs(o.x * cornersEdge1[i][0] + o.y * cornersEdge1[i][1]) < 0.25 \
						|| (d2 < 3) && abs(o.x * cornersEdge2[i][0] + o.y * cornersEdge2[i][1]) < 0.25)
					{
						dtype du = imgDu.at<dtype>(v, u), dv = imgDv.at<dtype>(v, u);
						cv::Mat uvt = (Mat_<dtype>(2, 1) << u, v);
						cv::Mat H = (Mat_<dtype>(2, 2) << du * du, du * dv, dv * du, dv * dv);
						G = G + H;
						cv::Mat t = H * (uvt);
						b = b + t;
					}
				}
			}//endfor
		//set new corner location if G has full rank
		Mat s, u, v;
		SVD::compute(G, s, u, v);
		int rank = 0;
		for (int k = 0; k < s.rows; k++)
		{
			if (s.at<dtype>(k, 0) > 0.0001 || s.at<dtype>(k, 0) < -0.0001)// not equal zero
			{
				rank++;
			}
		}
		if (rank == 2)
		{
			//std::cout << "rank:" << rank << std::endl;
			cv::Mat mp = G.inv() * b;
			cv::Point2f  corner_pos_new(mp.at<dtype>(0, 0), mp.at<dtype>(1, 0));
			//  % set corner to invalid, if position update is very large
			if (norm2d(cv::Point2f(corner_pos_new.x - cu, corner_pos_new.y - cv)) >= 4)
			{
				cornersEdge1[i][0] = 0;
				cornersEdge1[i][1] = 0;
				cornersEdge2[i][0] = 0;
				cornersEdge2[i][1] = 0;
			}
			else
			{
				cornors[i].x = mp.at<dtype>(0, 0);
				cornors[i].y = mp.at<dtype>(1, 0);
			}
		}
		else//otherwise: set corner to invalid
		{
			cornersEdge1[i][0] = 0;
			cornersEdge1[i][1] = 0;
			cornersEdge2[i][0] = 0;
			cornersEdge2[i][1] = 0;
		}
	}
}

float CornerDetAC::norm2d(cv::Point2f o)
{
	return sqrt(o.x * o.x + o.y * o.y);
}
void CornerDetAC::edgeOrientations(cv::Mat imgAngle, cv::Mat imgWeight, int index) {
	//number of bins (histogram parameter)
	int binNum = 32;

	//convert images to vectors
	if (imgAngle.size() != imgWeight.size())
		return;
	std::vector<dtype> vec_angle, vec_weight;
	for (int i = 0; i < imgAngle.cols; i++)
	{
		for (int j = 0; j < imgAngle.rows; j++)
		{
			// convert angles from .normals to directions
			float angle = imgAngle.ptr<dtype>(j)[i] + CV_PI / 2;
			angle = angle > CV_PI ? (angle - CV_PI) : angle;
			vec_angle.push_back(angle);

			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}
	//create histogram
	dtype pin = (CV_PI / binNum);//把角度PI分为32份
	std::vector<dtype> angleHist(binNum, 0);
	for (int i = 0; i < vec_angle.size(); i++)
	{
		int bin = std::max(std::min((int)floor(vec_angle[i] / pin), binNum - 1), 0);
		angleHist[bin] = angleHist[bin] + vec_weight[i];
	}//相同角度下累加权重，建立直方图，领域非零方向只有水平和垂直的两条直线，则找到直方图对应的两个modes

	// find modes of smoothed histogram
	std::vector<dtype> hist_smoothed(angleHist);
	std::vector<std::pair<dtype, int> > modes;
	findModesMeanShift(angleHist, hist_smoothed, modes, 1);
	// if only one or no mode = > return invalid corner
	if (modes.size() <= 1)
		return;

	//compute orientation at modes and sort by angle
	float fo[2];
	fo[0] = modes[0].second * pin;
	fo[1] = modes[1].second * pin;
	dtype deltaAngle = 0;
	if (fo[0] > fo[1])
	{
		dtype t = fo[0];
		fo[0] = fo[1];
		fo[1] = t;
	}

	deltaAngle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (dtype)CV_PI);
	// if angle too small => return invalid corner
	if (deltaAngle <= 0.3)//这个0.3是经验值  直方图两个峰值之间的角度差值，这个差值要大于一定的阈值才认为该角点有效
		return;

	//set statistics: orientations
	cornersEdge1[index][0] = cos(fo[0]);
	cornersEdge1[index][1] = sin(fo[0]);
	cornersEdge2[index][0] = cos(fo[1]);
	cornersEdge2[index][1] = sin(fo[1]);//计算角点对应的水平和垂直的两条直线的角度
}
dtype CornerDetAC::normpdf(dtype dist, dtype mu, dtype sigma)//正态分布
{
	dtype s = exp(-0.5 * (dist - mu) * (dist - mu) / (sigma * sigma));
	s = s / (std::sqrt(2 * CV_PI) * sigma);
	return s;
}
int cmp(const std::pair<dtype, int>& a, const std::pair<dtype, int>& b)
{
	return a.first > b.first;
}

void CornerDetAC::findModesMeanShift(std::vector<dtype> hist, std::vector<dtype>& hist_smoothed, std::vector<std::pair<dtype, int>>& modes, dtype sigma) {
	//efficient mean - shift approximation by histogram smoothing
	//compute smoothed histogram
	bool allZeros = true;
	//std::cout << "hist size:" << hist.size() << std::endl;
	for (int i = 0; i < hist.size(); i++)
	{
		dtype sum = 0;
		for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
		{
			int idx = 0;
			idx = (i + j) % hist.size();//负数求余的问题
			//std::cout << "idx:" << (i + j) << "\t" << idx << std::endl;
			sum = sum + hist[idx] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = sum;
		if (abs(hist_smoothed[i] - hist_smoothed[0]) > 0.0001)
			allZeros = false;// check if at least one entry is non - zero
		//(otherwise mode finding may run infinitly)
	}
	if (allZeros)return;

	//mode finding
	for (int i = 0; i < hist.size(); i++)
	{
		int j = i;
		while (true)
		{
			float h0 = hist_smoothed[j];
			int j1 = (j + 1) % hist.size();
			int j2 = (j - 1) % hist.size();
			float h1 = hist_smoothed[j1];
			float h2 = hist_smoothed[j2];
			if (h1 >= h0 && h1 >= h2)
				j = j1;
			else if (h2 > h0 && h2 > h1)
				j = j2;
			else
				break;
		}//找到前后左右的值，判断j j-1 j+1三个值哪个最大
		bool ys = true;
		if (modes.size() == 0)
		{
			ys = true;
		}
		else
		{
			for (int k = 0; k < modes.size(); k++)
			{
				if (modes[k].second == j)
				{
					ys = false;
					break;
				}
			}//判断modes中是否有重复的值
		}
		if (ys == true)
		{
			modes.push_back(std::make_pair(hist_smoothed[j], j));
		}
	}
	std::sort(modes.begin(), modes.end(), cmp);
	//std::cout << "modes end" << std::endl;
}

void CornerDetAC::scoreCorners(cv::Mat img, cv::Mat imgAngle, cv::Mat imgWeight, std::vector<cv::Point2f>& corners, std::vector<float>& score) {
	radius.push_back(4);
	radius.push_back(8);
	radius.push_back(12);
	for (int i = 0; i < corners.size(); i++)
	{
		//corner location
		int u = corners[i].x + 0.5;
		int v = corners[i].y + 0.5;
		std::vector<float> scores;
		for (int j = 0; j < radius.size(); j++)
		{
			scores.push_back(0);
			int r = radius[j];
			if (u > r && u <= (img.cols - r - 1) && v > r && v <= (img.rows - r - 1))
			{
				int startX, startY, ROIwidth, ROIheight;
				startX = u - r;
				startY = v - r;
				ROIwidth = 2 * r + 1;
				ROIheight = 2 * r + 1;
				cv::Mat sub_img = img(cv::Rect(startX, startY, ROIwidth, ROIheight)).clone();
				cv::Mat sub_imgWeight = imgWeight(cv::Rect(startX, startY, ROIwidth, ROIheight)).clone();
				std::vector<cv::Point2f> cornersEdge;
				cornersEdge.push_back(cv::Point2f((float)cornersEdge1[i][0], (float)cornersEdge1[i][1]));
				cornersEdge.push_back(cv::Point2f((float)cornersEdge2[i][0], (float)cornersEdge2[i][1]));
				cornerCorrelationScore(sub_img, sub_imgWeight, cornersEdge, scores[j]);
			}
		}
		score.push_back(*max_element(begin(scores), end(scores)));
	}
}
void CornerDetAC::cornerCorrelationScore(cv::Mat img, cv::Mat imgWeight, std::vector<cv::Point2f> cornersEdge, float& score) {
	//center
	int c[] = { imgWeight.cols / 2, imgWeight.cols / 2 };
	cv::Mat img_filter = cv::Mat::ones(imgWeight.size(), imgWeight.type());
	img_filter = img_filter * -1;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			cv::Point2f p1 = cv::Point2f(i - c[0], j - c[1]);
			cv::Point2f p2 = cv::Point2f(p1.x * cornersEdge[0].x * cornersEdge[0].x + p1.y * cornersEdge[0].x * cornersEdge[0].y,
				p1.x * cornersEdge[0].x * cornersEdge[0].y + p1.y * cornersEdge[0].y * cornersEdge[0].y);
			cv::Point2f p3 = cv::Point2f(p1.x * cornersEdge[1].x * cornersEdge[1].x + p1.y * cornersEdge[1].x * cornersEdge[1].y,
				p1.x * cornersEdge[1].x * cornersEdge[1].y + p1.y * cornersEdge[1].y * cornersEdge[1].y);
			float norm1 = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
			float norm2 = sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y));
			if (norm1 <= 1.0 || norm2 <= 1.0)
			{
				img_filter.ptr<dtype>(j)[i] = 1;
			}
		}
	}

	//normalize
	cv::Mat mean, std, mean1, std1;
	cv::meanStdDev(imgWeight, mean, std);
	cv::meanStdDev(img_filter, mean1, std1);
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			imgWeight.ptr<dtype>(j)[i] = (dtype)(imgWeight.ptr<dtype>(j)[i] - mean.ptr<double>(0)[0]) / (dtype)std.ptr<double>(0)[0];
			img_filter.ptr<dtype>(j)[i] = (dtype)(img_filter.ptr<dtype>(j)[i] - mean1.ptr<double>(0)[0]) / (dtype)std1.ptr<double>(0)[0];
		}
	}

	//convert into vectors
	std::vector<float> vec_filter, vec_weight;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			vec_filter.push_back(img_filter.ptr<dtype>(j)[i]);
			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}

	//compute gradient score
	float sum = 0;
	for (int i = 0; i < vec_weight.size(); i++)
	{
		sum += vec_weight[i] * vec_filter[i];
	}
	sum = (dtype)sum / (dtype)(vec_weight.size() - 1);
	dtype score_gradient = sum >= 0 ? sum : 0;

	//create intensity filter kernel
	cv::Mat kernelA, kernelB, kernelC, kernelD;
	createkernel(atan2(cornersEdge[0].y, cornersEdge[0].x), atan2(cornersEdge[1].y, cornersEdge[1].x), c[0], kernelA, kernelB, kernelC, kernelD);//1.1 产生四种核

	//checkerboard responses
	float a1, a2, b1, b2;
	a1 = kernelA.dot(img);
	a2 = kernelB.dot(img);
	b1 = kernelC.dot(img);
	b2 = kernelD.dot(img);

	float mu = (a1 + a2 + b1 + b2) / 4;

	float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
	float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
	float score_1 = score_a >= score_b ? score_b : score_a;

	score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
	score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
	float score_2 = score_a >= score_b ? score_b : score_a;

	float score_intensity = score_1 >= score_2 ? score_1 : score_2;
	score_intensity = score_intensity > 0.0 ? score_intensity : 0.0;

	score = score_gradient * score_intensity;
}

void CornerDetAC::createkernel(float angle1, float angle2, int kernelSize, cv::Mat& kernelA, cv::Mat& kernelB, cv::Mat& kernelC, cv::Mat& kernelD)
{
	//kernelSize -> radius
	int width = (int)kernelSize * 2 + 1;
	int height = (int)kernelSize * 2 + 1;
	kernelA = cv::Mat::zeros(height, width, mtype);
	kernelB = cv::Mat::zeros(height, width, mtype);
	kernelC = cv::Mat::zeros(height, width, mtype);
	kernelD = cv::Mat::zeros(height, width, mtype);

	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			dtype vec[] = { u - kernelSize, v - kernelSize };//相当于将坐标原点移动到核中心
			dtype dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);//相当于计算到中心的距离
			dtype side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);//相当于将坐标原点移动后的核进行旋转，以此产生四种核
			dtype side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);//X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
			if (side1 <= -0.1 && side2 <= -0.1) {
				kernelA.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1 && side2 >= 0.1) {
				kernelB.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 <= -0.1 && side2 >= 0.1) {
				kernelC.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1 && side2 <= -0.1) {
				kernelD.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
		}
	}
	//归一化
	kernelA = kernelA / cv::sum(kernelA)[0];
	kernelB = kernelB / cv::sum(kernelB)[0];
	kernelC = kernelC / cv::sum(kernelC)[0];
	kernelD = kernelD / cv::sum(kernelD)[0];

}