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

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;

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

  // find the poses in the query range
  Vector3d queryPosition = queryPose.topRightCorner(3, 1);

  // - lower bound
  int lowerIndexBound = queryIndex;
  for (int i = queryIndex - 1; i >= 0; i--)
  {
    Vector3d tmpPosition = poses[i].topRightCorner(3, 1);
    if ((tmpPosition - queryPosition).norm() < queryPoseRange)
      lowerIndexBound = i;
    else
      break;
  }

  // - upper bound
  int upperIndexBound = queryIndex;
  for (int i = queryIndex + 1; i < numPose; i++)
  {
    Vector3d tmpPosition = poses[i].topRightCorner(3, 1);
    if ((tmpPosition - queryPosition).norm() < queryPoseRange)
      upperIndexBound = i;
    else
      break;
  }

  cout << "Query index: " << queryIndex << endl;
  cout << "Lower bound index: " << lowerIndexBound << endl;
  cout << "Upper bound index: " << upperIndexBound << endl;

  // compute relative poses to query pose
  vector<Matrix4d> selectedPoses;
  for (int i = lowerIndexBound; i <= upperIndexBound; i++)
  {
    Matrix4d tmpPose =
        lidarToGps.inverse() * queryPose.inverse() * poses[i] * lidarToGps;
    selectedPoses.push_back(tmpPose);
  }

  cout << "Total scans in query range: " << selectedPoses.size() << endl;

  // generate color for each pcl
  vector<int> colorR, colorG, colorB;
  double stepColor = 255.0 / selectedPoses.size();
  for (int i = 0; i < selectedPoses.size(); i++)
  {
    int value = i * stepColor;
    int r = value;
    int g = 255 - value;
    int b = 0;

    colorR.push_back(r);
    colorG.push_back(g);
    colorB.push_back(b);
  }

  // read pcl data in the query range
  double minDist = 3.0;
  double maxDist = 100.0;
  vector<PointCloud::Ptr> pointClouds;
  vector<double> intensities;
  for (int i = lowerIndexBound; i <= upperIndexBound; i++)
  {
    ifstream ifs(pclList[i].c_str());

    PointCloud::Ptr pointsPCL(new PointCloud());
    pointsPCL->height = 1;
    pointsPCL->is_dense = false;

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

      intensities.push_back(point[3]);

      Point pPCL;
      pPCL.x = point[0];
      pPCL.y = point[1];
      pPCL.z = point[2];
      pointsPCL->push_back(pPCL);
    }

    pointsPCL->width = pointsPCL->size();
    pointClouds.push_back(pointsPCL);
  }

  PointCloud::Ptr queryCloud = pointClouds[queryIndex - lowerIndexBound];

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

  pcl::visualization::PCLVisualizer visAlign("visAlign");

  for (int i = 0; i < pointClouds.size(); i++)
  {
    // case 1: accumulate pointclouds
    PointCloud::Ptr pclAccumulated(new PointCloud());
    pcl::transformPointCloud(*pointClouds[i], *pclAccumulated,
                             selectedPoses[i].cast<float>());
    pointCloudsAccumulated.push_back(pclAccumulated);

    // case 2: aline poinclouds with NDT
    PointCloud::Ptr pclAligned(new PointCloud());
    if (i != queryIndex - lowerIndexBound)
    {
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
      cout << "Cloud #" << i << endl;
      cout << "score: " << ndt_omp->getFitnessScore() << endl;
      cout << "initial transform: \n" << initTransform << endl;
      cout << "final transform: \n" << finalTransform << endl;
    }
    else
    {
      pclAligned = pointClouds[i];
      alignedPoses.push_back(selectedPoses[i]);
      pointCloudsAligned.push_back(pclAligned);
    }

    // display alignment
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        accumulated_handler(pclAccumulated, 0.0, 0.0, 255.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        aligned_handler(pclAligned, 0.0, 255.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        query_handler(queryCloud, 255.0, 0.0, 0.0);
    visAlign.removeAllPointClouds();
    visAlign.addPointCloud(pclAccumulated, accumulated_handler, "accumulated");
    visAlign.addPointCloud(pclAligned, aligned_handler, "aligned");
    visAlign.addPointCloud(queryCloud, query_handler, "query");
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
      aligned_handler(mergedAlignedCloud, 0.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      target_handler(queryCloud, 255.0, 0.0, 0.0);
  vis.addPointCloud(mergedAccumulatedCloud, source_handler, "accumulated");
  vis.addPointCloud(mergedAlignedCloud, aligned_handler, "aligned");
  vis.addPointCloud(queryCloud, target_handler, "query");
  vis.spin();

  return 0;
}
