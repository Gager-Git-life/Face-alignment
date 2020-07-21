//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include "UltraFace.hpp"
#include "Pfld.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <mnn .mnn> [image files...]\n", argv[0]);
        return 1;
    }

    string ultraface_mnn_path = argv[1];
    string pfld_mnn_path = argv[2];
    UltraFace ultraface(ultraface_mnn_path, 320, 240, 4, 0.65); // config model input
    Pfld pfld(pfld_mnn_path, 8);


    string image_file = argv[3];
    //cout << "Processing " << image_file << endl;

    cv::Mat frame = cv::imread(image_file);
    auto start = chrono::steady_clock::now();
    vector<FaceInfo> face_info;
    ultraface.detect(frame, face_info);

    for (auto face : face_info) {
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        //cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);

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
        //string croped_landmark_name = "croped_landmark.jpg";
        //cv::imwrite(croped_landmark_name, resize_img_copy);
        pfld.Pic_Landmark(frame, landmarks);
        
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "[INFO]>>> all time: " << elapsed.count() << " s" << endl;
    cv::imshow("UltraFace", frame);
    cv::waitKey();
    //string result_name = "result" + to_string(i) + ".jpg";
    //cv::imwrite(result_name, frame);
    
    return 0;
}
