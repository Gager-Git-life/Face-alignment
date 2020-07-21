#include "Pfld.hpp"


using namespace std;
using namespace cv;


Pfld::Pfld(const std::string &mnn_path, int num_thread_){

    //加载模型
    pfld_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    // 配置调度
    MNN::ScheduleConfig config;
    config.numThread = num_thread_;
    config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    // 配置后端
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    // 创建会话
    pfld_session = pfld_interpreter->createSession(config);
    input_tensor = pfld_interpreter->getSessionInput(pfld_session, nullptr);


    vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    //auto nhwc_data   = nhwc_Tensor->host<float>();
    //auto nhwc_size   = nhwc_Tensor->size();

    //cout << "[INFO]>>> model loaded" << endl;


}

Pfld::~Pfld(){

    pfld_interpreter->releaseModel();
    pfld_interpreter->releaseSession(pfld_session);
}

cv::Mat Pfld::Get_Resize_Croped_Img(cv::Mat frame, cv::Mat &resize_img_copy, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh){

    float cx, cy, halfw;
    cv::Mat resize_img, croped_img;
    cv::Point_<float> center_point;

    try{
        //cout << "[INFO]>>> pt1:" << pt1 << "\t pt2:" << pt2 << endl;
        center_point = (pt1 + pt2) / 2;
        //cout << "[INFO]>>> center_point:" << center_point << endl;
        cx = center_point.x;
        cy = center_point.y;
        //cout << "[INFO]>>> cx:" << cx << "\t cy:" << cy << endl;
        halfw = max((pt2.x - pt1.x)/2, (pt2.y - pt1.y)/2);
        //cout << "[INFO]>>> halfw:" << halfw << endl;
        float min_x = (cx-halfw) > 0 ? cx-halfw:0;
        float min_y = (cy-halfw) > 0 ? cy-halfw:0;
        croped_img = frame(cv::Rect(min_x, min_y, 2*halfw, 2*halfw));
        //string croped_name = "croped_img.jpg";
        //cv::imwrite(croped_name, croped_img);
        croped_wh = cv::Size(2*halfw, 2*halfw);
        //cout << "[INFO]>>> croped_wh:" << croped_wh << endl;

        s_point = cv::Point(min_x, min_y);
        //cout << "[INFO]>>> s_point:" << s_point << endl;
        if(halfw > 20){
            factor = 2*halfw / INPUT_SIZE;
            cv::resize(croped_img, resize_img, cv::Size(INPUT_SIZE, INPUT_SIZE));
            resize_img_copy = resize_img;
            resize_img.convertTo(resize_img, CV_32FC3);
            resize_img = (resize_img - 123.0) / 58.0;
        } 
    }
    catch(exception e){
        cout << "[INFO]>>> No face was detected!!!" << endl;
    }

    return resize_img;
}

void Pfld::Get_Landmark_Points(cv::Mat resize_img, cv::Mat &resize_img_copy, cv::Size croped_wh, cv::Point s_point, vector<LandmarkInfo> &outputs){

    auto start = chrono::steady_clock::now();

    // wrapping input tensor, convert nhwc to nchw
    //vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    //auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, resize_img.data, nhwc_size);

    auto input_tensor  = pfld_interpreter->getSessionInput(pfld_session, nullptr);
    input_tensor->copyFromHostTensor(nhwc_Tensor);


    //pfld_interpreter->resizeTensor(input_tensor, {1, 3, 112, 112});
    //pfld_interpreter->resizeSession(pfld_session);
    //std::shared_ptr<MNN::CV::ImageProcess> pretreat(
    //        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3,
    //                                      norm_vals, 3));
    //pretreat->convert(resize_img.data, 112, 112, resize_img.step[0], input_tensor);
    //cout << "[INFO]>>> img:" << nhwc_data << endl;

    // run network
    pfld_interpreter->runSession(pfld_session);
    // get output data
    string output_tensor_name = "conv5_fwd";
    MNN::Tensor *tensor_landmarks = pfld_interpreter->getSessionOutput(pfld_session, output_tensor_name.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);

    auto landmarks = tensor_landmarks_host.host<float>();
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    //cout << "[INFO]>>> pfld 关键点推理耗时:" << elapsed.count() << " s" << endl;


    //int batch   = tensor_landmarks->batch();
    //int channel = tensor_landmarks->channel();
    //int height  = tensor_landmarks->height();
    //int width   = tensor_landmarks->width();
    //int type    = tensor_landmarks->getDimensionType();
    //printf("%d, %d, %d, %d, %d\n", batch, channel, height, width, type);


    for(int i=0; i<98; i++){
        LandmarkInfo point_info;
        point_info.index = i;
        //cv::circle(resize_img_copy, cv::Point(landmarks[i*2+0], landmarks[i*2+1]), 1, Scalar(0,0,255), 1);
        point_info.pt.x = int(landmarks[i*2+0] * factor + s_point.x);
        point_info.pt.y = int(landmarks[i*2+1] * factor + s_point.y);
        outputs.push_back(point_info);
    }
}


Scalar Get_Color_(int num){

    if(num <= 32){
        return Scalar(0, 0, 255);}
    else if(num <= 50){
        return Scalar(0 ,250,154);}
    else if(num <= 54){
        return Scalar(0,0,0);}
    else if(num <= 59){
        return Scalar(255,255,0);}
    else if(num <= 75){
        return Scalar(255, 20, 147);}
    else if(num <= 87){
        return Scalar(0, 255, 0);}
    else if(num <= 95){
        return Scalar(255, 0, 0);}
    else if(num == 96 or num == 97){
        return Scalar(255, 20, 147);}

}

void Pfld::Pic_Landmark(cv::Mat &frame, vector<LandmarkInfo> points){

    int point_num = points.size();
    for(int i=0; i<point_num; i++){
        auto color_ = Get_Color_(i);
        cv::circle(frame, points[i].pt, 1, color_, 1);
    }
}
