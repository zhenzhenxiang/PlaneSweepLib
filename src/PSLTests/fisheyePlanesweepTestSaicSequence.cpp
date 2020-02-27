// This file is part of PlaneSweepLib (PSL)

// Copyright 2016 Christian Haene (ETH Zuerich)

// PSL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// PSL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with PSL.  If not, see <http://www.gnu.org/licenses/>.
#include <string>
#include <boost/program_options.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <psl_base/exception.h>
#include <opencv2/opencv.hpp>
#include <psl_cudaBase/cudaFishEyeImageProcessor.h>
#include <psl_stereo/cudaFishEyePlaneSweep.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <psl_base/common.h>

void makeOutputFolder(std::string folderName)
{
  if (!boost::filesystem::exists(folderName))
  {
    if (!boost::filesystem::create_directory(folderName))
    {
      std::stringstream errorMsg;
      errorMsg << "Could not create output directory: " << folderName;
      PSL_THROW_EXCEPTION(errorMsg.str().c_str());
    }
  }
}

int main(int argc, char* argv[])
{
  std::string dataFolder;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message")(
      "dataFolder", boost::program_options::value<std::string>(&dataFolder)
                        ->default_value("data/fisheyeCamera/902-Seq"),
      "One of the data folders for pinhole planesweep provided with the plane "
      "sweep code.");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 1;
  }

  // read poses from file
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      systemR;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      systemT;
  std::vector<uint64_t> timestamps;

  std::ifstream systemPosesFile;
  systemPosesFile.open((dataFolder + "/poses_seq.txt").c_str());
  if (!systemPosesFile.is_open())
  {
    PSL_THROW_EXCEPTION("Error opening poses_seq.txt");
  }

  std::string line;
  while (std::getline(systemPosesFile, line))
  {
    if (line[0] == '#')
      continue;

    std::stringstream lineStr(line);

    uint64_t timestamp;

    double yaw, pitch, roll, t_x, t_y, t_z;
    lineStr >> timestamp >> t_x >> t_y >> t_z >> roll >> pitch >> yaw;

    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

    systemR.push_back(R);

    Eigen::Vector3d T(t_x, t_y, t_z);
    systemT.push_back(T);

    timestamps.push_back(timestamp);
  }

  // read in the calibration
  double k1, k2, p1, p2;
  std::string calibFileName = dataFolder + "/calib.txt";

  std::ifstream calibrationStr;
  calibrationStr.open(calibFileName.c_str());

  if (!calibrationStr.is_open())
  {
    PSL_THROW_EXCEPTION("Error opening calibration file calib.txt.")
  }

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  double xi;

  // intrinsic calibration and distortion parameters
  calibrationStr >> K(0, 0) >> K(0, 1) >> K(0, 2);
  calibrationStr >> K(1, 1) >> K(1, 2);
  calibrationStr >> xi;
  calibrationStr >> k1 >> k2 >> p1 >> p2;

  // extrinsic calibration (camera to system)
  double roll, pitch, yaw, t_x, t_y, t_z;
  calibrationStr >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;

  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d C(t_x, t_y, t_z);

  // open the video
  std::string videoFile = dataFolder + "/video_seq.avi";

  cv::VideoCapture videoCap(videoFile);
  if (!videoCap.isOpened())
  {
    PSL_THROW_EXCEPTION("Could not load video file")
  }

  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  makeOutputFolder("fisheyeTestResultsSaicSeq");
  makeOutputFolder("fisheyeTestResultsSaicSeq/grayscaleZNCC");
  makeOutputFolder(
      "fisheyeTestResultsSaicSeq/grayscaleZNCC/NoOcclusionHandling/");

  // Plane sweeping stereo
  {
    PSL::CudaFishEyePlaneSweep cFEPS;
    cFEPS.setScale(1.0);
    cFEPS.setMatchWindowSize(30, 30);
    cFEPS.setNumPlanes(5);
    cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_NONE);
    cFEPS.setPlaneGenerationMode(
        PSL::FISH_EYE_PLANE_SWEEP_PLANEMODE_UNIFORM_DEPTH_GROUND);
    cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_ZNCC);
    cFEPS.setSubPixelInterpolationMode(
        PSL::FISH_EYE_PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
    cFEPS.enableOutputBestDepth();
    cFEPS.enableOutputBestCosts();
    cFEPS.enableOuputUniquenessRatio(false);
    cFEPS.enableOutputCostVolume();
    cFEPS.enableSubPixel(false);

    int refId = 0;

    double groundDeltaRange = 0.3;
    double minZ = -groundDeltaRange / 2.0;
    double maxZ = groundDeltaRange / 2.0;
    cFEPS.setZRange(minZ, maxZ);

    double rollRange = 2.0 * M_PI / 180.0;
    double pitchRange = 2.0 * M_PI / 180.0;
    cFEPS.setRollAngleRange(rollRange);
    cFEPS.setPitchAngleRange(pitchRange);
    cFEPS.setNumRollAngles(5);
    cFEPS.setNumPitchAngles(5);

    // iterate frames in the video
    int numFrame = 0;
    while (true)
    {
      cv::Mat imageOrig;
      videoCap >> imageOrig;

      if (imageOrig.empty())
      {
        std::cout << "Video finished!" << std::endl;
        break;
      }

      // undistort and add the image
      cv::Mat imageGray;
      cv::cvtColor(imageOrig, imageGray, CV_BGR2GRAY);

      if (numFrame == 0)
        devImg.allocatePitchedAndUpload(imageGray);
      else
        devImg.reallocatePitchedAndUpload(imageGray);

      // Assemble camera matrix
      Eigen::Matrix4d cameraToSystem = Eigen::Matrix4d::Identity();
      cameraToSystem.topLeftCorner(3, 3) = R;
      cameraToSystem.topRightCorner(3, 1) = C;

      Eigen::Matrix4d systemToWorld = Eigen::Matrix4d::Identity();
      systemToWorld.topLeftCorner(3, 3) = systemR[numFrame];
      systemToWorld.topRightCorner(3, 1) = systemT[numFrame];

      Eigen::Matrix4d worldToCamera =
          cameraToSystem.inverse() * systemToWorld.inverse();

      PSL::FishEyeCameraMatrix<double> cam(K, worldToCamera.topLeftCorner(3, 3),
                                           worldToCamera.topRightCorner(3, 1),
                                           xi);

      cFEIP.setInputImg(devImg, cam);

      std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>>
          undistRes = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

      int id = cFEPS.addDeviceImage(undistRes.first, undistRes.second);

      numFrame++;
    }

    {
      refId = numFrame - 1;
      cFEPS.process(refId);
      PSL::FishEyeDepthMap<float, double> fEDM;
      fEDM = cFEPS.getBestDepth();
      cv::Mat refImage = cFEPS.downloadImage(refId);

      cv::imwrite("fisheyeTestResultsSaicSeq/grayscaleZNCC/"
                  "NoOcclusionHandling/refImg.png",
                  refImage);
      float minDepth = 3.0;
      float maxDepth = 60.0;
      fEDM.saveInvDepthAsColorImage("fisheyeTestResultsSaicSeq/"
                                    "grayscaleZNCC/NoOcclusionHandling/"
                                    "invDepthCol.png",
                                    minDepth, maxDepth);

      std::ofstream meshFile("fisheyeTestResultsSaicSeq/grayscaleZNCC/"
                                   "NoOcclusionHandling/mesh.wrl");
      fEDM.meshToVRML(meshFile, "refImg.png", 1.0, -1, maxDepth);

      cv::imshow("Reference Image", refImage);
      fEDM.displayInvDepthColored(minDepth, maxDepth, 1);

      cv::Mat colInvDepth;
      fEDM.getInvDepthColored(minDepth, maxDepth, colInvDepth);

      // show depth on the edges
      cv::Mat detectedEdges;
      cv::blur(refImage, detectedEdges, cv::Size(3, 3));
      cv::Canny(detectedEdges, detectedEdges, 30.0, 100.0, 3);

      cv::Mat edgeOnColInvDepth;
      cv::Mat invDetectedEdges =
          cv::Mat::ones(detectedEdges.size(), detectedEdges.type()) * 255 -
          detectedEdges;
      colInvDepth.copyTo(edgeOnColInvDepth, invDetectedEdges);

      cv::imshow("edges on the depth", edgeOnColInvDepth);
      cv::waitKey(0);
    }
  }
}
