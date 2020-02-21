link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pclomp/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/pclomp/lib)

add_executable(searchCameraLocation searchCameraLocation.cpp)
add_executable(searchScanLocation searchScanLocation.cpp)
add_executable(pointCloudProjection pointCloudProjection.cpp)
add_executable(createVideoSequence createVideoSequence.cpp)

add_executable(pointCloudFusion pointCloudFusion.cpp)
target_link_libraries (pointCloudFusion ${PCL_LIBRARIES})

add_executable(queryLocalPointClouds queryLocalPointClouds.cpp)
target_link_libraries (queryLocalPointClouds ${PCL_LIBRARIES} ndt_omp)

add_executable(pointCloudProjectionFromPly pointCloudProjectionFromPly.cpp)
target_link_libraries (pointCloudProjectionFromPly ${PCL_LIBRARIES})
