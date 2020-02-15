link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(searchCameraLocation searchCameraLocation.cpp)
add_executable(pointCloudProjection pointCloudProjection.cpp)
add_executable(createVideoSequence createVideoSequence.cpp)
