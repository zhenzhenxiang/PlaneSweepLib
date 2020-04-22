find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pclomp/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/pclomp/lib)

add_executable(searchCameraLocation searchCameraLocation.cpp)
add_executable(searchScanLocation searchScanLocation.cpp)

link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pointCloudProjection pointCloudProjection.cpp)

add_executable(createVideoSequence createVideoSequence.cpp)

add_executable(pointCloudFusion pointCloudFusion.cpp)

add_executable(queryLocalPointClouds queryLocalPointClouds.cpp)
target_link_libraries (queryLocalPointClouds ndt_omp)

add_executable(pointCloudProjectionFromPly pointCloudProjectionFromPly.cpp)

add_executable(syncCameraLidar syncCameraLidar.cpp)
target_link_libraries (syncCameraLidar ndt_omp)
