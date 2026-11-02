cmake_minimum_required(VERSION 3.22...4.2)
project(df11_device_decode)

set(CMAKE_CXX_STANDARD 20)

add_executable(df11_device_decode)
target_sources(df11_device_decode PRIVATE host/decode_host.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(df11_device_decode PUBLIC TT::Metalium)
target_include_directories(df11_device_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/host)
