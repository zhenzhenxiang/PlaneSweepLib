link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pinholePlanesweepTest pinholePlanesweepTest.cpp)
add_executable(fisheyePlanesweepTest fisheyePlanesweepTest.cpp)
add_executable(fisheyePlanesweepTestSaic fisheyePlanesweepTestSaic.cpp)
target_link_libraries (fisheyePlanesweepTestSaic ${PCL_LIBRARIES})

add_executable(fisheyePlanesweepTestSaicVideo fisheyePlanesweepTestSaicVideo.cpp)
add_executable(fisheyePlanesweepTestSaicSequence fisheyePlanesweepTestSaicSequence.cpp)
