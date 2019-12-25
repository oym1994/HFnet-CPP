#include "mat2tensor.h"
#include <time.h>
#include <chrono>


int main(int argc, char* argv[]){

    std::string model_path ="/home/oym/hfnet/model_path/experiment_data/saved_models/hfnet";

    string resampler_path ="/home/oym/tensorflow/bazel-bin/tensorflow/contrib/resampler/python/ops/_resampler_ops.so";

    string image_path("/home/oym/test/tensorflow-tutorial-master/cpp/floor.png");

    string Paths("/home/oym/test/tensorflow-tutorial-master/image"); //("/media/oym/source/oym/images");//

    vector<string> files=getpngFiles(Paths);  //得到路径下所有的文件名

    Mat image ;// =imread(image_path);IMREAD_GRAYSCALE

    Feature_point feature_point(model_path,resampler_path);

    vector<KeyPoint> keypoints;

    for (string path:files){

        image = imread(Paths+path);

        feature_point.detect_superpoint(image, keypoints,100, 0 );

        cout<<keypoints.size()<<endl;

#if show_image==1

        image_show(image, keypoints);

        waitKey();

#endif
        
        cv::goodFeaturesToTrack()

    }

    return 1;

}
