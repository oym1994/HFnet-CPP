#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/c/c_api.h"
using namespace cv;
using namespace tensorflow;
using namespace  std;
#define Debug 1
#define show_time 1
#define show_image 1

vector<string> getFiles(string cate_dir);

vector<string> getpngFiles(string png_dir);

void image_show(Mat image, std::vector<KeyPoint> keypoints);

int filenamefilter(const struct dirent *cur);

#define output_feature {"keypoints"}

#define input_image {{"image:0", img_tensor},{"pred/simple_nms/radius", radius},{"pred/top_k_keypoints/k",keypoints}}


class Feature_point{

    unique_ptr<Session> sess;

    TF_Status* tf_status;

    GraphDef graph;

    vector<Tensor> outputs;

    Tensor keypoints_(int,TensorShape());

    Tensor radius_(int,TensorShape());

    bool Loadgraph(std::string model_Dir);

    bool Loadresampler(std::string resampler_Dir);

    bool create_graph();

    void mat2tensor(const Mat& image, Tensor* tensor);

    void detect_points(Mat& image, int num = 100, int r =4);
public:

    Feature_point(string model_Dir, string resampler_Dir, int num =100, int radius =4);

    void detect_superpoint(Mat& image, vector<KeyPoint>& keypoint,int num = 100, int r =4);

    ~Feature_point(){

      TF_DeleteStatus(tf_status);

      cout<<"Deconstruct Superpoint Object"<<endl;

           }

};






