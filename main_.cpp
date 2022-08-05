
//利用特征点进行双目拼接
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
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
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl
#define ISH
using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.4f;
string features_type = "orb";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.4f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

int main11(int argc, char* argv[])
{
    std::string path;
    path = "./data/8.01/9.bmp"; //./data/7.20/scene/2/2.bmp
    img_names.push_back(path);
    path = "./data/8.01/10.bmp";
    img_names.push_back(path);
    //path = "./data/3.jpg";
    //img_names.push_back(path);

    int num_images = static_cast<int>(img_names.size());
    std::cout << num_images << std::endl;
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    LOGLN("Finding features...");
    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

#ifdef ISH
    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();
        std::cout << "full_size:" << full_img.size() << std::endl;
        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                std::cout << "!is_work_scale_set" << std::endl;
                work_scale = min(1.0, sqrt(work_megapix * 1e8/ full_img.size().area()));
               // std::cout << "full_img area:" << work_megapix * 1e4 << "\t" << full_img.size().area() << "\t" << sqrt(work_megapix * 1e4 / full_img.size().area()) << std::endl;
                is_work_scale_set = true;
            }
            //std::cout << "work_scale:" << work_scale << std::endl;
            cv::resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
            //img size is 2560*1440
           // std::cout << "img size:" << img.size() << std::endl;
        }
        if (!is_seam_scale_set)
        {
            std::cout << "!is_seam_scale_set" << std::endl;

            seam_scale = min(1.0, sqrt(seam_megapix * 1e8/ full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        std::cout << "img size: " << img.size() << std::endl;

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
        std::cout << "seam_scale:" << seam_scale << std::endl;
        cv::resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        //img size is 2560*1440
        images[i] = img.clone();
        //std::cout << "images size:" << images[i].size() << std::endl;
    }


    //特征点匹配的方法
    full_img.release();
    img.release();
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine") {
        std::cout << ".............here 3........" << std::endl;
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);

    }
    else if (range_width == -1) {
        std::cout << ".............here 2........" << std::endl;
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);

    }
    else {
        std::cout << ".............here 1........" << std::endl;
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    }
    (*matcher)(features, pairwise_matches);
    for (int k = 0; k < features.size(); k++) {
        std::cout << "check match:" << features[k].keypoints.size() << "\t" << pairwise_matches[k].matches.size() << std::endl;
    }




    matcher->collectGarbage();
    // Check if we should save matches graph
    //if (save_graph)
    //{
    //    LOGLN("Saving matches graph...");
    //    ofstream f(save_graph_to.c_str());
    //    f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    //}
    // Leave only images we are sure are from the same panorama
    std::cout << "features size:" << features.size() << "\t" << pairwise_matches.size() << "\t" << conf_thresh << std::endl;
    for (int n = 0; n < features.size(); n++) {
        std::cout << "keypoints size:" << features[n].keypoints.size() << std::endl;
        cv::Mat result;
        std::cout << "show images size:" << images[n].size() << std::endl;
        cv::drawKeypoints(images[n], features[n].keypoints, result, cv::Scalar::all(-1));
        cv::namedWindow("fe", cv::WINDOW_NORMAL);
        cv::imshow("fe", result);
        cv::waitKey(0);

    }

    //确定重叠区域中匹配的点
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    std::cout << "indices:" << indices.size() << std::endl;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        //std::cout << "...indices:" << indices[i] << std::endl;
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        std::cout << "...........size:" << images[indices[i]].size() << std::endl;
    }
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    std::cout << "........." << num_images << std::endl;

    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    //根据重叠区域相同的特征点求单应矩阵
    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return -1;
    }
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);//转为32位
        cameras[i].R = R;
        LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);

        //std::cout << "...............camera matrix............." << std::endl;
       // std::cout << cameras[i].K() << "\n" << cameras[i].R << "\n" << cameras[i].t << std::endl;
    }
    
    //利用光束平差法优化内外参数
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return -1;
    }
    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    for (int tt = 0; tt < focals.size(); tt++) {
        std::cout << focals[tt] << std::endl;
    }
    std::cout << "end focals" << std::endl;

    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    std::cout << "warped_image_scale:" << warped_image_scale << std::endl;
    //选择较大的focals

    //由于进行光束平差法之后会产生弯曲的情况，则利用波形变化进行拉直
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }
    LOGLN("Warping images (auxiliary)... ");
    cv::FileStorage fs;
    fs.open("matrix.xml", cv::FileStorage::WRITE);
    fs << "R1" << cameras[0].R;
    fs << "K1" << cameras[0].K();
    fs << "R2" << cameras[1].R;
    fs << "K2" << cameras[1].K();
    fs.release();

    std::cout << cameras[0].R << "\n" << cameras[0].K() << "\n\n\n" << cameras[1].R << "\n" << cameras[1].K() << std::endl;
    std::cout << cameras[0].R.size() << "\t" << cameras[0].R.type() << std::endl;
    std::cout << cameras[1].R.size() << "\t" << cameras[1].R.type() << std::endl;
#else
    cv::FileStorage fs;
    fs.open("matrix.xml", cv::FileStorage::READ);
    //vector<CameraParams> cameras;
    std::vector<cv::Mat>KA, RA;
    cv::Mat K1 = cv::Mat::eye(3, 3, CV_64FC1);
    fs["K1"] >> K1;
    //K1.at<double>(0, 0) = 14096.51613534859;
    //K1.at<double>(0, 1) = 0;
    //K1.at<double>(0, 2) = 960;
    //K1.at<double>(1, 0) = 0;
    //K1.at<double>(1, 1) = 14096.51613534859;
    //K1.at<double>(1, 2) = 540;
    //cameras[0].K() = K1.clone();
   // K1.convertTo(cameras[0].K(), CV_32F); 
   // cameras[0].K()

    cv::Mat R1 = cv::Mat::zeros(3, 3, CV_32FC1);
    fs["R1"] >> R1;
    //[0.99888951, 0.022950353, 0.041146293;
    //3.2534204e-09, 0.87333339, -0.48712289;
    //-0.047114085, 0.48658195, 0.87236357]
 /*   R1.at<float>(0, 0) = 0.99888951;
    R1.at<float>(0, 1) = 0.022950353;
    R1.at<float>(0, 2) = 0.041146293;
    R1.at<float>(1, 0) = 3.2534204e-09;
    R1.at<float>(1, 1) = 0.87333339;
    R1.at<float>(1, 2) = -0.48712289;
    R1.at<float>(2, 0) = -0.047114085;
    R1.at<float>(2, 1) = 0.48658195;
    R1.at<float>(2, 2) = 0.87236357;*/
    //cameras[0].R = R1.clone();


   cv::Mat K2 = cv::Mat::eye(3, 3, CV_64FC1);
   fs["K2"] >> K2;
    //K2.at<double>(0, 0) = 19106.35672110942;
    //K2.at<double>(0, 1) = 0;
    //K2.at<double>(0, 2) = 1280;
    //K2.at<double>(1, 0) = 0;
    //K2.at<double>(1, 1) = 19106.35672110942;
    //K2.at<double>(1, 2) = 720;
    //cameras[1].K() = K2;
    cv::Mat R2 = cv::Mat::zeros(3, 3, CV_32FC1);
    fs["R2"] >> R2;
    //[0.99888295, -0.023234453, -0.041146293;
    //-7.4505806e-09, 0.87076324, -0.49170244;
    //0.047253117, 0.49115315, 0.86979049]
    //R2.at<float>(0, 0) = 0.99888295;
    //R2.at<float>(0, 1) = -0.023234453;
    //R2.at<float>(0, 2) = -0.041146293;
    //R2.at<float>(1, 0) = -7.4505806e-09;
    //R2.at<float>(1, 1) = 0.87076324;
    //R2.at<float>(1, 2) = -0.49170244;
    //R2.at<float>(2, 0) = 0.047253117;
    //R2.at<float>(2, 1) = 0.49115315;
    //R2.at<float>(2, 2) = 0.86979049;
    KA.push_back(K1);
    KA.push_back(K2);
    RA.push_back(R1);
    RA.push_back(R2);
   // cameras[1].R = R2;
    float warped_image_scale;
    if (K1.at<double>(0, 0) > K2.at<double>(0, 0))
        warped_image_scale = K1.at<double>(0, 0);
    else
        warped_image_scale = K2.at<double>(0, 0);

    
    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();
        images[i] = full_img.clone();
        //cv::imshow("1", images[i]);
        //cv::waitKey(0);
    }
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);
    // Preapre images masks
    //std::cout << "num_images:" << num_images << std::endl;
    for (int i = 0; i < num_images; ++i)
    {
        //std::cout << "check masks:" << images[i].size() << std::endl;
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }
    // Warp images and their masks
    //柱面投影原图和mask
    Ptr<WarperCreator> warper_creator;
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }
    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }
    //warped_image_scale为K的fx   seam_work_aspect== 1
    //warped_image_scale * seam_work_aspect = 4052.74;
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));//warped_image_scale * seam_work_aspect
    std::cout << "warper:" << warped_image_scale * seam_work_aspect << std::endl;
    //std::vector<cv::Point2f> temp1;
    //temp1.push_back(cv::Point2f(-2219, 5687));
    //temp1.push_back(cv::Point2f(-269, 5801));

    for (int i = 0; i < num_images; ++i)
    {
#ifdef ISH
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        std::cout << "seam_work_aspect:" << seam_work_aspect << std::endl;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;
        //std::cout << "K:" << K << std::endl;
        std::cout << images[i].size() << "......................." << std::endl;
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);//柱面投影图片
        std::cout << corners[i] << std::endl;
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);//柱面投影mask
#else
        Mat_<float> K;
        float swa = (float)seam_work_aspect;
        std::cout << "seam_work_aspect:" << seam_work_aspect << std::endl;

        KA[i].convertTo(K, CV_32F);
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;
        //std::cout << "k:" << K(0, 0) << "\t" << K(0, 2) << "\t" << K(1, 1) << "\t" << K(1, 2) << std::endl;
        //std::cout << "RR:" << RA[i] << std::endl;
        std::cout << "K:" << K << std::endl;

        corners[i] = warper->warp(images[i], K, RA[i], INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        //corners[i] = temp1[i];

        //sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, RA[i], INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

#endif

      //cv::imshow("ki1", images_warped[i]);
      //cv::waitKey(0);

    }
    std::cout << "warp end" << std::endl;
    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);
    //调整曝光
    std::cout << "compensator end" << std::endl;
    //拼接缝隙
    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }
    seam_finder->find(images_warped_f, corners, masks_warped);
    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
    LOGLN("Compositing...");
    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;

    //融合
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        //LOGLN("Compositing image #" << indices[img_idx] + 1);
        // Read image and resize it if necessary
        full_img = imread(samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;
            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);
            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                //// Update intrinsics
#ifdef  ISH
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;
#else
                KA[i].at<double>(0, 0) *= compose_work_aspect;
                KA[i].at<double>(0, 2) *= compose_work_aspect;
                KA[i].at<double>(1, 2) *= compose_work_aspect;

                //std::cout << KA[i].at<float>(0, 0) << "\t" << KA[i].at<float>(0, 2) << "\t" << KA[i].at<float>(1, 2) << std::endl;
#endif //  ISH
                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }
                Mat K;
#ifdef ISH
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
#else
                KA[i].convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, RA[i]);
#endif // ISH
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            cv::resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();
        std::cout << "......img_size:" << img_size << std::endl;
        Mat K;
        // Warp the current image
#ifdef ISH
        
        cameras[img_idx].K().convertTo(K, CV_32F);
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
#else
        KA[img_idx].convertTo(K, CV_32F);

        warper->warp(img, K, RA[img_idx], INTER_LINEAR, BORDER_REFLECT, img_warped);

#endif // DEBUG

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
#ifdef  ISH
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

#else
        warper->warp(mask, K, RA[img_idx], INTER_NEAREST, BORDER_CONSTANT, mask_warped);
#endif //  ISH
        
        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
        cv::dilate(masks_warped[img_idx], dilated_mask, Mat());
        cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;
        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            std::cout << "blend_width:" << blend_width << std::endl;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f / blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }
        // Blend the current image
        if (timelapse)
        {
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }
    if (!timelapse)
    {
        std::cout << "!timelapse" << std::endl;
        Mat result, result_mask;
        blender->blend(result, result_mask);
        imwrite(result_name, result);
    }
    return 0;
}








