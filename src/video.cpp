#include "UltraFace.hpp"
#include "Pfld.hpp"
#include "skin.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


int main(int argc, char **argv){

    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <mnn .mnn> [image files...]\n", argv[0]);
        return 1;
    }

    string ultraface_mnn_path = argv[1];
    string pfld_mnn_path = argv[2];
    UltraFace ultraface(ultraface_mnn_path, 320, 240, 4, 0.65); // config model input
    Pfld pfld(pfld_mnn_path, 1);

    VideoCapture capture(-1);
    if(!capture.isOpened()){
        cout << "[INFO]>>> 摄像头开启失败" << endl;
    }

    while(1){
        auto start = chrono::steady_clock::now();
        cv::Mat frame;
        capture >> frame;

        vector<FaceInfo> face_info;
        ultraface.detect(frame, face_info);

        for (auto face : face_info) {
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);

            // pfld
            cv::Mat resize_img, resize_img_copy;
            cv::Point s_point;
            cv::Size croped_wh;
            vector<LandmarkInfo> landmarks;

            resize_img = pfld.Get_Resize_Croped_Img(frame, resize_img_copy, pt1, pt2, s_point, croped_wh);
            if(!resize_img.data){
                continue;
            }
            pfld.Get_Landmark_Points(resize_img, resize_img_copy, croped_wh, s_point, landmarks);
            pfld.Pic_Landmark(frame, landmarks);

        }
        //frame = ChangeFacecolor(frame);
        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "[INFO]>>> fps:" << 1. / elapsed.count() << " s" << endl;

        cv::imshow("视频", frame);
        if(cv::waitKey(1) >= 0) {
            break;
        }
    }
    return 0;
}
