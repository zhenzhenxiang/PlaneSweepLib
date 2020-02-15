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
                        ->default_value("DataFisheyeCamera/right"),
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

  // read in the calibration
  std::vector<PSL::FishEyeCameraMatrix<double> > cams;
  std::vector<std::vector<double> > dist_coeffs;
  std::string calibFileName = dataFolder + "/calib.txt";

  std::ifstream calibrationStr;
  calibrationStr.open(calibFileName.c_str());

  if (!calibrationStr.is_open())
  {
    PSL_THROW_EXCEPTION("Error opening calibration file calib.txt.")
  }

  int numCam;
  calibrationStr >> numCam;

  for (unsigned int i = 0; i < numCam; i++)
  {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    double xi;
    std::vector<double> dist_coeff(4);

    // intrinsic calibration and distortion parameters
    calibrationStr >> K(0, 0) >> K(0, 1) >> K(0, 2);
    calibrationStr >> K(1, 1) >> K(1, 2);
    calibrationStr >> xi;
    calibrationStr >> dist_coeff[0] >> dist_coeff[1] >> dist_coeff[2] >>
        dist_coeff[3];

    // extrinsic calibration
    double roll, pitch, yaw, t_x, t_y, t_z;
    calibrationStr >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;

    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

    Eigen::Vector3d C(t_x, t_y, t_z);

    cams.push_back(PSL::FishEyeCameraMatrix<double>(K, R.transpose(),
                                                    -R.transpose() * C, xi));
    dist_coeffs.push_back(dist_coeff);
  }

  // now load the video filenames
  std::string videoListFile = dataFolder + "/videos.txt";

  std::ifstream videosStream;
  videosStream.open(videoListFile.c_str());

  if (!videosStream.is_open())
  {
    PSL_THROW_EXCEPTION("Could not load videos list file")
  }

  std::vector<cv::VideoCapture> videoCaps;
  videoCaps.resize(numCam);

  {
    std::string videoFileName;
    for (unsigned int i = 0; i < numCam; i++)
    {
      videosStream >> videoFileName;
      videoFileName = dataFolder + "/" + videoFileName;

      videoCaps[i] = cv::VideoCapture(videoFileName);

      if (!videoCaps[i].isOpened())
        PSL_THROW_EXCEPTION("Could not open video file")
    }
  }

  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  makeOutputFolder("fisheyeTestResultsSaicVideo");
  makeOutputFolder("fisheyeTestResultsSaicVideo/grayscaleZNCC");
  makeOutputFolder(
      "fisheyeTestResultsSaicVideo/grayscaleZNCC/NoOcclusionHandling/");

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
    double minZ = - groundDeltaRange / 2.0;
    double maxZ = groundDeltaRange / 2.0;
    cFEPS.setZRange(minZ, maxZ);

    double rollRange = 2.0 * M_PI / 180.0;
    double pitchRange = 2.0 * M_PI / 180.0;
    cFEPS.setAngleRange(rollRange, pitchRange);
    cFEPS.setNumAngles(5);

    // iterate frames
    int numFrame = 0;
    int numSkip = 500;
    while (cv::waitKey(10) != 27)
    {
      if (numFrame < numSkip)
      {
        for (unsigned int i = 0; i < numCam; i++)
        {
          cv::Mat imageOrig;
          videoCaps[i] >> imageOrig;

          if (imageOrig.empty())
          {
            std::cout << "Video finished!" << std::endl;
            return 0;
          }
        }

        std::cout << "skipping frame #" << numFrame << std::endl;
        numFrame++;

        continue;
      }

      // undistort and update the images
      std::vector<cv::Mat> rawImages;
      for (unsigned int i = 0; i < numCam; i++)
      {
        cv::Mat imageOrig;
        videoCaps[i] >> imageOrig;

        if (imageOrig.empty())
        {
          std::cout << "Video finished!" << std::endl;
          return 0;
        }

        rawImages.push_back(imageOrig);

        cv::Mat imageGray;
        cv::cvtColor(imageOrig, imageGray, CV_BGR2GRAY);

        devImg.reallocatePitchedAndUpload(imageGray);

        cFEIP.setInputImg(devImg, cams[i]);

        double k1 = dist_coeffs[i][0];
        double k2 = dist_coeffs[i][1];
        double p1 = dist_coeffs[i][2];
        double p2 = dist_coeffs[i][3];
        std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double> >
            undistRes = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

        if (numFrame == numSkip)
          int id = cFEPS.addDeviceImage(undistRes.first, undistRes.second);
        else
          cFEPS.updateDeviceImage(i, undistRes.first);
      }

      // display raw images for all views
      cv::Mat mainWindowImage = rawImages[0];
      cv::Mat smallWindowImages;
      cv::hconcat(rawImages[1], rawImages[3], smallWindowImages);
      cv::hconcat(smallWindowImages, rawImages[2], smallWindowImages);
      int newWidth = mainWindowImage.cols;
      int newHeight = smallWindowImages.rows / 3;
      cv::resize(smallWindowImages, smallWindowImages, cv::Size(newWidth, newHeight));
      cv::Mat windowImage;
      cv::vconcat(mainWindowImage, smallWindowImages, windowImage);
      cv::resize(windowImage, windowImage, cv::Size(), 0.5, 0.5);
      cv::imshow("Raw images", windowImage);
      cv::waitKey(1);

      {
        cFEPS.process(refId);
        PSL::FishEyeDepthMap<float, double> fEDM;
        fEDM = cFEPS.getBestDepth();
        cv::Mat refImage = cFEPS.downloadImage(refId);

        cv::imwrite("fisheyeTestResultsSaicVideo/grayscaleZNCC/"
                    "NoOcclusionHandling/refImg.png",
                    refImage);
        float minDepth = 3.0;
        float maxDepth = 60.0;
        fEDM.saveInvDepthAsColorImage("fisheyeTestResultsSaicVideo/"
                                      "grayscaleZNCC/NoOcclusionHandling/"
                                      "invDepthCol.png",
                                      minDepth, maxDepth);

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
        cv::waitKey(1);
      }

      numFrame++;
    }
  }
}
