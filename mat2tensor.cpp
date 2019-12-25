#include "mat2tensor.h"
#include <assert.h>
#include <fstream>
#include <dirent.h>

vector<string> getFiles(string cate_dir)
{
    vector<string> files,files_jpg;//存放文件名
    DIR *dir; struct
    dirent *ptr;
    if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
        perror("Open dir error...");
        exit(1);
        }
    while ((ptr=readdir(dir)) != NULL)
    {
     if(ptr->d_type == 8)
         files.push_back(ptr->d_name);
    }
    closedir(dir);
    for (string s: files ){

        if (s.substr(s.length()-4,4) == ".png"){
                files_jpg.push_back('/'+s);
          }
    }
    return files_jpg;
}

int filenamefilter(const struct dirent *cur){

    std::string str(cur->d_name);

    if(str.find(".png") != std::string::npos){

        return 1;
    }

    return 0;
}

vector<string> getpngFiles(string png_dir)
{
   struct dirent **namelist;

   std::vector<std::string> ret;

   int n = scandir(png_dir.c_str(),&namelist,filenamefilter, alphasort);

   if(n <0 ){

       return ret;
   }

   for (int i=0; i<n; i++){

       std::string filepath(namelist[i]->d_name);

       ret.push_back("/"+filepath);
   }

   free(namelist);

   return ret;
}

void image_show(Mat image, std::vector<KeyPoint> keypoints){

    static int count =0;

//    Rect a;

//    a.x = 100;

//    a.y = (image.rows)/2-1 ;

//    a.height =(image.rows) /2;

//    a.width =image.cols-200;

//    cv::imwrite("../image/"+std::to_string(count++)+".png",image(a));

//    Mat resize_img;

//    cv::resize(image, resize_img, Size(0,0), 0.5, 0.5);

    for(KeyPoint kp : keypoints){

          cv::circle(image,kp.pt,2,Scalar(0,0,255),-1);

      }

    cv::namedWindow("Superpoint");

    cv::imshow("Superpoint",image);


//    cv::imwrite(std::to_string(r)+"-"+std::to_string(num)+".png", show_image);

}


bool Feature_point::Loadgraph(string modelDir){

    tensorflow::SessionOptions Session_options;

    tensorflow::RunOptions run_options;

    tensorflow::SavedModelBundle bundle;

    Status status = LoadSavedModel(Session_options,run_options,modelDir,{tensorflow::kSavedModelTagServe}, &bundle);

    if(!status.ok()){

        std::cerr<<"Error Reading graph defenition from"+modelDir+":"+status.ToString()<<std::endl;

        return -1;
    }
    std::cout<<"Sucessfully Reading graph defenition "<<std::endl;

    sess = std::move(bundle.session);

    Status status_create = sess->Create(graph);

    if(!status_create.ok()){

        std::cerr<<"ERROR create graph"<<endl;

        return -1;
    }


    return 1;

}

bool Feature_point::Loadresampler(std::string resampler_Dir){

    tf_status = TF_NewStatus();

    TF_LoadLibrary("/home/oym/tensorflow/bazel-bin/tensorflow/contrib/resampler/python/ops/_resampler_ops.so",tf_status);

    if (TF_GetCode(tf_status) !=TF_OK) {

        std::cerr << "TF_LoadLibrary  _resampler_ops.so ERROR, 加载resampler库失败，请指定正确的动态库路径和及正确的版本\n";

        return -1;

        }

    std::cout<<"Sucessfully Load resampler"<<std::endl;

    return 1;
}

bool Feature_point::create_graph(){

    Status status_create = sess->Create(graph);

    if(!status_create.ok()){

        std::cerr<<"ERROR create graph"<<endl;

        return -1;
    }

    return 1;

}

void Feature_point::mat2tensor(const Mat& image, Tensor *tensor){

//#if Debug==1

//    const auto t_start = std::chrono::system_clock::now();

//#endif

    float *p=tensor->flat<float>().data();

    cv::Mat imagepixel(image.rows,image.cols,CV_32F,p);

    image.convertTo(imagepixel,CV_32F);

//#if Debug==1

//    const auto t_end = std::chrono::system_clock::now();

//    const auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t_start-t_end).count();

//    std::cout<<"===Mat-->Tensor time comsumed===:   "<<t<<"ms"<<endl;

//#endif

}


Feature_point::Feature_point(string model_Dir, string resampler_Dir, int num , int r ){

    bool status_resampler = Loadresampler(resampler_Dir);

    bool status_loadgraph = Loadgraph(model_Dir);

    assert(status_resampler && status_loadgraph);

//    keypoints_.scalar<int>()() = num;

//    radius_.scalar<int>()() = radius;

}


void Feature_point::detect_points(Mat& image, int num, int r){

#if Debug==1

    const auto t_start = std::chrono::system_clock::now();

#endif

if(image.channels() != 1){

    cv::cvtColor(image,image,cv::COLOR_RGB2GRAY);

}

    Tensor keypoints(DT_INT32,TensorShape());

    Tensor radius(DT_INT32,TensorShape());

    keypoints.scalar<int>()() = num;

    radius.scalar<int>()() = r;

    Tensor img_tensor(DT_FLOAT,TensorShape({1,image.rows,image.cols,1}));

    mat2tensor(image, &img_tensor);

    outputs.clear();

    Status status = sess->Run(input_image,output_feature, {}, &outputs);//Run,得到运行结果，存到outputs中

    assert(status.ok());

#if Debug==1

    const auto t_end = std::chrono::system_clock::now();

    const auto t = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count();

    std::cout<<"===detect features time comsumed===:  "<<t<<" microseconds"<<endl;

#endif

}


void Feature_point::detect_superpoint(Mat& image, vector<KeyPoint>& keypoints,int num , int r){

#if show_time==1

    const auto t_start = std::chrono::system_clock::now();

#endif

    keypoints.clear();

    detect_points(image, num, r);

    Tensor tmap = outputs[0];

    int output_dim = tmap.shape().dim_size(1);

    auto data = tmap.tensor<int32,3>();

    KeyPoint kp;

    for(int i =0 ; i<output_dim; i++){

      kp.pt = Point2f(data(2*i),data(2*i+1));

      keypoints.push_back(kp);

      }

#if show_time==1

    const auto t_end = std::chrono::system_clock::now();

    const auto t = std::chrono::duration_cast<std::chrono::microseconds>(t_start-t_end).count();

    std::cout<<"===detect and collect features time comsumed:  "<<t<<" microseconds"<<endl;

#endif

}
