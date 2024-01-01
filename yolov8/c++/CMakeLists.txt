cmake_minimum_required(VERSION 3.4.1)

project(c++)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
set(RKNN_API_PATH /home/ubuntu/repositories/rknn/yolov8/c++/include)
include_directories(${RKNN_API_PATH}/include)
# opencv
# find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# link_directories(${CUDA_LIBRARIES})

add_executable(main main.cpp)

set(RKNN_RT_LIB /home/ubuntu/repositories/rknn/yolov8/c++/lib/librknnrt.so)
target_link_libraries(main ${RKNN_RT_LIB})
# target_link_libraries(main ${RKNN_RT_LIB} ${OpenCV_LIBS})


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME})
install(TARGETS main DESTINATION ./)
