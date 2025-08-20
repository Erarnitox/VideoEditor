cmake_minimum_required(VERSION 3.10)
project(VideoEditor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(SndFile REQUIRED)
find_package(TBB REQUIRED)

add_executable(video_editor main_streaming.cpp)

target_include_directories(video_editor PRIVATE ${OpenCV_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS})
target_link_libraries(video_editor PRIVATE TBB::tbb ${OpenCV_LIBS} ${SndFile_LIBRARIES})
