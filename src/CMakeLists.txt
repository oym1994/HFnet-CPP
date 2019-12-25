set(PROJECT_NAME tf_tutorial)
cmake_minimum_required (VERSION 2.8.8)
project (${PROJECT_NAME})
set(CMAKE_C_COMPILER "/usr/bin/gcc")
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_C_FLAGS "-std=c99")
#-DEIGEN_USE_MKL_ALL")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(OpenCV_DIR /usr/local/opencv/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
if(OpenCV_FOUND)
    message("Find OPENCV")
    message("${OpenCV_INCLUDE_DIRS}\n")
    message("${OpenCV_VERSION}")
    message("${OpenCV_LIBS}")

else()
    message("Not Found Opencv")
endif()



link_directories(/home/oym/test/test1/lib
                 /usr/local/cuda-9.0/lib64
                 /home/oym/test/tensorflow-c-mnist/lib
                 ${OpenCV_LIBS})


include_directories(
   /home/oym/tensorflow
   /home/oym/tensorflow/bazel-genfiles
   /home/oym/tensorflow/bazel-bin/tensorflow
   /home/oym/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
   /home/oym/test/eigen3.5
   /home/oym/tensorflow/tensorflow/contrib/makefile/downloads/absl
   /usr/local/protobuf/include
   src
   ${OpenCV_INCLUDE_DIRS})


add_executable(tf_test  src/mat2tensor.cpp src/Detect.cpp)
target_link_libraries(tf_test ${OPENCV_LIBS} tensorflow_cc tensorflow_framework ${OpenCV_LIBS} )

#/home/oym/tensorflow/tensorflow/c/c_api.cc
