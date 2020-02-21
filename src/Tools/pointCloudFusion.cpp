#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    cout << "Usage: ./pointCloudFusion calib.txt point_clouds.txt poses.txt"
         << endl;
    return 1;
  }

  string calibFilePath = argv[1];
  string pclListFilePath = argv[2];
  string posesFilePath = argv[3];

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

  // generate color for each pcl
  vector<int> colorR, colorG, colorB;
  int stepColor = 255 / numPcl;
  for (int i = 0; i < numPcl; i++)
  {
    // rainbow color map
    int value = i * stepColor;
    int r = value > 128 ? (value - 128) * 2 : 0;
    int g = value < 128 ? 2 * value : 255 - ((value - 128) * 2);
    int b = value < 128 ? 255 - (2 * value) : 0;

    colorR.push_back(r);
    colorG.push_back(g);
    colorB.push_back(b);
  }

  // read pcl data
  double minDist = 3.0;
  double maxDist = 100.0;
  vector<vector<Vector3d>> pointClouds;
  vector<PointCloud::Ptr> pointCloudsPCL;
  vector<double> intensities;
  for (int i = 0; i < pclList.size(); i++)
  {
    ifstream ifs(pclList[i].c_str());

    PointCloud::Ptr pointsPCL(new PointCloud);
    pointsPCL->height = 1;
    pointsPCL->is_dense = false;

    vector<Vector3d> pcl;
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

      pcl.push_back(pos);
      intensities.push_back(point[3]);

      Point pPCL;
      pPCL.x = point[0];
      pPCL.y = point[1];
      pPCL.z = point[2];
      pPCL.r = colorR[i];
      pPCL.g = colorG[i];
      pPCL.b = colorB[i];
      pointsPCL->push_back(pPCL);
    }

    pointClouds.push_back(pcl);

    pointsPCL->width = pcl.size();
    pointCloudsPCL.push_back(pointsPCL);
  }

  // read poses
  int numPose = 0;
  vector<Matrix4d> poses;
  Matrix4d initPose = Matrix4d::Identity();
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

    if (numPose == 0)
    {
      initPose = curPose;
      curPose = Matrix4d::Identity();
    }
    else
    {
      curPose =
          lidarToGps.inverse() * initPose.inverse() * curPose * lidarToGps;
    }

    poses.push_back(curPose);
    cout << "Pose #" << numPose << ":\n" << curPose << endl;

    numPose++;
  }

  // compute relative pose with ICP and initial guess
  pcl::IterativeClosestPoint<Point, Point> icp;
  pcl::PLYWriter writer;
  PointCloud mergedCloud;
  for (int i = 1; i < pointCloudsPCL.size(); i++)
  {
    PointCloud::Ptr src = pointCloudsPCL[i];
    PointCloud::Ptr tar = pointCloudsPCL[0];

    icp.setInputSource(src);
    icp.setInputTarget(tar);

    PointCloud final;
    icp.align(final, poses[i].cast<float>());
    mergedCloud += final;

    cout << "ICP results:" << endl;
    cout << "Has converged:\n" << icp.hasConverged() << endl;
    cout << " Score:\n" << icp.getFitnessScore() << endl;
    cout << "Transformation:\n" << icp.getFinalTransformation() << endl;

    writer.write("data/test/" + to_string(i) + ".ply", final, true);
  }

  // write ref point clouds as ply file
  writer.write("data/test/0.ply", *pointCloudsPCL[0], true);
  mergedCloud += *pointCloudsPCL[0];

  // write merged point cloud as ply file
  writer.write("data/test/merged.ply", mergedCloud, true);

  // transform point clouds to the first lidar frame
  for (int i = 0; i < numPcl; i++)
    for (auto& p : pointClouds[i])
    {
      Matrix3d R = poses[i].topLeftCorner(3, 3);
      Vector3d T = poses[i].topRightCorner(3, 1);
      p = R * p + T;
    }

  // save to VRML file
  string outputFile = "data/test/fusedPCL.wrl";
  ofstream vrmlFile(outputFile.c_str());
  if (!vrmlFile.is_open())
  {
    cout << "Cannot open the output vrml file!" << endl;
    return 1;
  }

  vrmlFile << "#VRML V2.0 utf8" << endl;
  vrmlFile << "Shape {" << endl;
  vrmlFile << "     appearance Appearance {" << endl;
  vrmlFile << "         material Material { " << endl;
  vrmlFile << "             diffuseColor     0.5 0.5 0.5" << endl;
  vrmlFile << "         }" << endl;
  vrmlFile << "     }" << endl;
  vrmlFile << "     geometry PointSet {" << endl;
  vrmlFile << "       coord Coordinate {" << endl;
  vrmlFile << "           point [" << endl;
  for (int i = 0; i < numPcl; i++)
    for (int j = 0; j < pointClouds[i].size(); j++)
    {
      vrmlFile << "               " << pointClouds[i][j](0) << " "
               << pointClouds[i][j](1) << " " << pointClouds[i][j](2) << ","
               << endl;
    }
  vrmlFile << "           ]" << endl;
  vrmlFile << "       }" << endl;
  vrmlFile << "       color Color {" << endl;
  vrmlFile << "         color [" << endl;

  for (int i = 0; i < numPcl; i++)
    for (int j = 0; j < pointClouds[i].size(); j++)
    {
      vrmlFile << "           " << colorR[i] / 255.0 << " " << colorG[i] / 255.0
               << " " << colorB[i] / 255.0 << "," << endl;
    }

  vrmlFile << "         ]" << endl;
  vrmlFile << "       }" << endl;
  vrmlFile << "   }" << endl;
  vrmlFile << "}" << endl;

  cout << "Finished to write to file: " << outputFile << endl;

  return 0;
}
