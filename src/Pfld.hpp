//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#ifndef Pfld_hpp
#define Pfld_hpp

#pragma once

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


using namespace std;

typedef struct LandmarkInfo {
    int index;
    cv::Point_<int> pt;

} LandmarkInfo;


class Pfld {
public:
    Pfld(const std::string &mnn_path, int num_thread_ = 4);

    ~Pfld();
    cv::Mat Get_Resize_Croped_Img(cv::Mat frame, cv::Mat &resize_img_copy, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh);
    void Get_Landmark_Points(cv::Mat resize_img, cv::Mat &resize_img_copy, cv::Size croped_wh, cv::Point s_point, vector<LandmarkInfo> &output);
    void Pic_Landmark(cv::Mat &frame, vector<LandmarkInfo> points);

private:

    std::shared_ptr<MNN::Interpreter> pfld_interpreter;
    MNN::Session *pfld_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    MNN::Tensor *nhwc_Tensor = nullptr;

    const float mean_vals[3] = {128, 128, 128};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
    const int INPUT_SIZE = 96;
    float factor;
    cv::Size InputSize = cv::Size(112, 112);
};

#endif /* UltraFace_hpp */
