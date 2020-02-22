#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types_conversion.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;

void readPointCloudFromTxt(string filename, double minDist, double maxDist,
                           PointCloud::Ptr pointCloud);

int main(int argc, char** argv)
{
  if (argc != 6)
  {
    cout << "Usage: ./pointCloudFusion calib.txt point_clouds.txt "
            "scanLocation.txt "
            "timestamp pose_range" << endl;
    return 1;
  }

  string calibFilePath = argv[1];
  string pclListFilePath = argv[2];
  string posesFilePath = argv[3];

  uint64_t queryTimestamp = atoll(argv[4]);
  double queryPoseRange = atof(argv[5]);

  // open files
  ifstream calibFile(calibFilePath.c_str());
  ifstream pclListFile(pclListFilePath.c_str());
  ifstream posesFile(posesFilePath.c_str());

  if (!calibFile.is_open() || !pclListFile.is_open() || !posesFile.is_open())
  {
    cout << "Cannot open calib, pcl or poses file!" << endl;
    return 1;
  }

  // read extrinsic param from lidar to vehicle
  Matrix4d lidarToGps = Matrix4d::Identity();
  string desc;

  Vector3d T;
  calibFile >> desc >> T(0) >> T(1) >> T(2);

  double roll, pitch, yaw;
  calibFile >> roll >> pitch >> yaw;
  Matrix3d R;
  R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  lidarToGps.topLeftCorner(3, 3) = R;
  lidarToGps.topRightCorner(3, 1) = T;

  // read pcl file list
  int numPcl = 0;
  vector<string> pclList;
  string line;
  while (getline(pclListFile, line))
  {
    pclList.push_back(line);
    numPcl++;
  }

  cout << numPcl << " PCL files loaded!" << endl;

  // search query timestamp
  auto iterTimestamp = find_if(
      pclList.begin(), pclList.end(), [&queryTimestamp](const string& file)
      {
        uint64_t curTimestamp =
            atoll(file.substr(file.size() - 23, file.size() - 4).c_str());
        return queryTimestamp == curTimestamp;
      });

  if (iterTimestamp == pclList.end())
  {
    cout << "Cannot find the query timestamp in point_clouds.txt." << endl;
    return 1;
  }

  int queryIndex = iterTimestamp - pclList.begin();

  // read poses
  int numPose = 0;
  vector<Matrix4d> poses;
  Matrix4d queryPose = Matrix4d::Identity();
  bool findQueryPose = false;
  vector<uint64_t> timestamps;
  while (getline(posesFile, line))
  {
    if (line[0] == '#')
      continue;

    stringstream lineStr(line);

    uint64_t timestamp;

    double yaw, pitch, roll, t_x, t_y, t_z;
    lineStr >> timestamp >> t_x >> t_y >> t_z >> roll >> pitch >> yaw;

    Matrix3d R;
    R = AngleAxisd(yaw, Vector3d::UnitZ()) *
        AngleAxisd(pitch, Vector3d::UnitY()) *
        AngleAxisd(roll, Vector3d::UnitX());

    Vector3d T(t_x, t_y, t_z);

    Matrix4d curPose = Matrix4d::Identity();
    curPose.topLeftCorner(3, 3) = R;
    curPose.topRightCorner(3, 1) = T;

    if ((timestamp - queryTimestamp / 1e3 < 1) && numPose == queryIndex)
    {
      queryPose = curPose;
      findQueryPose = true;
    }

    poses.push_back(curPose);
    timestamps.push_back(timestamp);
    //    cout << "Pose #" << numPose << ":\n" << curPose << endl;

    numPose++;
  }

  if (!findQueryPose)
  {
    cout << "Failed to find query pose corresponding to query timestamp!"
         << endl;
    return 1;
  }

  // construct kd-tree for pose searching
  PointCloud::Ptr poseCloud(new PointCloud());
  for (auto pose : poses)
  {
    Vector3d pos = pose.topRightCorner(3, 1);
    Point pt;
    pt.x = pos(0);
    pt.y = pos(1);
    pt.z = pos(2);
    poseCloud->push_back(pt);
  }

  Vector3d queryPosition = queryPose.topRightCorner(3, 1);
  Point queryPoint;
  queryPoint.x = queryPosition(0);
  queryPoint.y = queryPosition(1);
  queryPoint.z = queryPosition(2);

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(poseCloud);

  // find the poses in the query range
  vector<int> pointIdxRadiusSearch;
  vector<float> dist;
  int posesInRange = kdtree.radiusSearch(queryPoint, queryPoseRange,
                                         pointIdxRadiusSearch, dist);
  cout << "Found poses in range: " << posesInRange << endl;

  // compute relative poses to query pose
  vector<Matrix4d> selectedPoses;
  for (int i = 0; i < posesInRange; i++)
  {
    int ind = pointIdxRadiusSearch[i];
    Matrix4d tmpPose =
        lidarToGps.inverse() * queryPose.inverse() * poses[ind] * lidarToGps;
    selectedPoses.push_back(tmpPose);
  }

  // read pcl data in the query range
  double minDist = 3.0;
  double maxDist = 100.0;
  vector<PointCloud::Ptr> pointClouds;
  for (int i = 0; i < posesInRange; i++)
  {
    int ind = pointIdxRadiusSearch[i];

    PointCloud::Ptr points(new PointCloud());
    readPointCloudFromTxt(pclList[ind], minDist, maxDist, points);
    pointClouds.push_back(points);
  }

  PointCloud::Ptr queryCloud(new PointCloud());
  readPointCloudFromTxt(pclList[queryIndex], minDist, maxDist, queryCloud);

  // create downsampling filter
  pcl::VoxelGrid<Point> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  // downsample target cloud
  PointCloud::Ptr downsampled(new PointCloud());
  voxelgrid.setInputCloud(queryCloud);
  voxelgrid.filter(*downsampled);
  queryCloud = downsampled;

  // create ndt aligner
  pclomp::NormalDistributionsTransform<Point, Point>::Ptr ndt_omp(
      new pclomp::NormalDistributionsTransform<Point, Point>());
  ndt_omp->setStepSize(0.1);
  ndt_omp->setResolution(1.0);

  // transform point clouds to the query frame
  vector<PointCloud::Ptr> pointCloudsAccumulated;
  vector<PointCloud::Ptr> pointCloudsAligned;
  vector<Matrix4d> alignedPoses;
  vector<Matrix4d> alignedAccPoses;

  PointCloud::Ptr queryAccCloud(new PointCloud());
  *queryAccCloud += *queryCloud;

  pcl::visualization::PCLVisualizer visAlign("visAlign");

  for (int i = 0; i < pointClouds.size(); i++)
  {
    // case 1: accumulate pointclouds
    PointCloud::Ptr pclAccumulated(new PointCloud());
    pcl::transformPointCloud(*pointClouds[i], *pclAccumulated,
                             selectedPoses[i].cast<float>());
    pointCloudsAccumulated.push_back(pclAccumulated);

    // case 2: align to query poincloud with NDT
    PointCloud::Ptr pclAligned(new PointCloud());

    // downsampling
    PointCloud::Ptr downsampled(new PointCloud());
    voxelgrid.setInputCloud(pointClouds[i]);
    voxelgrid.filter(*downsampled);
    pointClouds[i] = downsampled;

    // align
    Matrix4f initTransform = selectedPoses[i].cast<float>();
    ndt_omp->setInputSource(pointClouds[i]);
    ndt_omp->setInputTarget(queryCloud);
    ndt_omp->align(*pclAligned, initTransform);

    pointCloudsAligned.push_back(pclAligned);
    Matrix4f finalTransform = ndt_omp->getFinalTransformation();
    alignedPoses.push_back(finalTransform.cast<double>());

    // print status
    cout << "Aligned Cloud #" << i << endl;
    cout << "score: " << ndt_omp->getFitnessScore() << endl;
    cout << "initial transform: \n" << initTransform << endl;
    cout << "final transform: \n" << finalTransform << endl;

    // case 3: align to accumulated pointclouds with NDT
    PointCloud::Ptr pclAlignedAcc(new PointCloud());

    // downsampling
    PointCloud::Ptr downsampled2(new PointCloud());
    voxelgrid.setInputCloud(queryAccCloud);
    voxelgrid.filter(*downsampled2);
    queryAccCloud = downsampled2;

    // align
    ndt_omp->setInputTarget(queryAccCloud);
    ndt_omp->align(*pclAlignedAcc, initTransform);

    finalTransform = ndt_omp->getFinalTransformation();
    alignedAccPoses.push_back(finalTransform.cast<double>());
    *queryAccCloud += *pclAlignedAcc;

    // print status
    cout << "Aligned Accumulated Cloud #" << i << endl;
    cout << "score: " << ndt_omp->getFitnessScore() << endl;
    cout << "initial transform: \n" << initTransform << endl;
    cout << "final transform: \n" << finalTransform << endl;

    // display
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        query_handler(queryAccCloud, 255.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        aligned_handler(pclAlignedAcc, 0.0, 255.0, 0.0);
    visAlign.removeAllPointClouds();
    visAlign.addPointCloud(queryAccCloud, query_handler, "query");
    visAlign.addPointCloud(pclAlignedAcc, aligned_handler, "alignedAcc");
    visAlign.spinOnce();
  }

  // merge into one point cloud
  PointCloud::Ptr mergedAccumulatedCloud(new PointCloud());
  PointCloud::Ptr mergedAlignedCloud(new PointCloud());
  for (int i = 1; i < pointCloudsAligned.size(); i++)
  {
    *mergedAccumulatedCloud += *pointCloudsAccumulated[i];
    *mergedAlignedCloud += *pointCloudsAligned[i];
  }

  cout << "Total points of accumulated cloud: "
       << mergedAccumulatedCloud->points.size() << endl;
  cout << "Total points of aligned cloud: " << mergedAlignedCloud->points.size()
       << endl;

  // save to ply file
  pcl::PLYWriter writer;
  string mergedAccumulatedPlyFileName = "data/test/merged_accumulated_" +
                                        to_string(queryTimestamp) + '_' +
                                        to_string(queryPoseRange) + ".ply";
  writer.write(mergedAccumulatedPlyFileName, *mergedAccumulatedCloud, true);

  string mergedAlignedPlyFileName = "data/test/merged_aligned_" +
                                    to_string(queryTimestamp) + '_' +
                                    to_string(queryPoseRange) + ".ply";
  writer.write(mergedAlignedPlyFileName, *mergedAlignedCloud, true);

  string mergedAlignedAccPlyFileName = "data/test/merged_aligned_accumulated_" +
                                       to_string(queryTimestamp) + '_' +
                                       to_string(queryPoseRange) + ".ply";
  writer.write(mergedAlignedAccPlyFileName, *queryAccCloud, true);

  string queryPlyFileName =
      "data/test/query_" + to_string(queryTimestamp) + ".ply";
  writer.write(queryPlyFileName, *queryCloud, true);

  cout << "Finished to write to file:\n";
  cout << mergedAccumulatedPlyFileName << endl;
  cout << mergedAlignedPlyFileName << endl;
  cout << queryPlyFileName << endl;

  // visualization
  pcl::visualization::PCLVisualizer vis("vis");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      source_handler(mergedAccumulatedCloud, 0.0, 0.0, 255.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      aligned_handler(queryAccCloud, 0.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      target_handler(queryCloud, 255.0, 0.0, 0.0);
  vis.addPointCloud(mergedAccumulatedCloud, source_handler, "accumulated");
  vis.addPointCloud(queryAccCloud, aligned_handler, "alignedAcc");
  vis.addPointCloud(queryCloud, target_handler, "query");
  vis.spin();

  return 0;
}

void readPointCloudFromTxt(string filename, double minDist, double maxDist,
                           PointCloud::Ptr pointCloud)
{
  ifstream ifs(filename);
  string line;
  while (getline(ifs, line))
  {
    if (line[0] == 'n')
      continue;

    stringstream lineStr(line);
    string tmpStr;
    vector<double> point;
    for (int i = 0; i < 4; i++)
    {
      getline(lineStr, tmpStr, ',');
      point.push_back(atof(tmpStr.c_str()));
    }

    Vector3d pos;
    pos << point[0], point[1], point[2];

    // ignore points too close or too far
    if (pos.norm() < minDist || pos.norm() > maxDist)
      continue;

    Point pPCL;
    pPCL.x = point[0];
    pPCL.y = point[1];
    pPCL.z = point[2];
    pointCloud->push_back(pPCL);
  }
}
