cmake_minimum_required(VERSION 3.6)
project(openblas)
aux_source_directory(. DIRSRCS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(./ blas/include) # 头文件
link_libraries(blas/lib/libopenblas.lib)
add_subdirectory(blas) # 子目录
add_executable(openblas ${DIRSRCS})
target_link_libraries(openblas ) # 链接
