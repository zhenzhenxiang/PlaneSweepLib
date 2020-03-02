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

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>

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

void loadData(std::string dataFolder,
              std::vector<PSL::FishEyeCameraMatrix<double>>& cams,
              std::vector<std::vector<double>>& dist_coeffs,
              std::vector<std::string>& imageFileNames,
              std::vector<std::string>& freespaceFileNames,
              std::vector<std::string>& viewMaskFileNames,
              std::vector<std::string>& stereoMaskFileNames);

void erodeMask(cv::Mat& inMask, int kernelSize, cv::Mat& outMask);
void dilateMask(cv::Mat& inMask, int kernelSize, cv::Mat& outMask);

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
  std::vector<PSL::FishEyeCameraMatrix<double>> cams;
  std::vector<std::vector<double>> dist_coeffs;
  std::vector<std::string> imageFileNames, freespaceFileNames,
      viewMaskFileNames, stereoMaskFileNames;

  loadData(dataFolder, cams, dist_coeffs, imageFileNames, freespaceFileNames,
           viewMaskFileNames, stereoMaskFileNames);

  int numCam = cams.size();

  // get depth mask for CAM-F120
  int refId = 0;

  cv::Mat viewMaskF120 =
      cv::imread(viewMaskFileNames[refId], cv::IMREAD_GRAYSCALE);
  viewMaskF120 = viewMaskF120 > 0;

  erodeMask(viewMaskF120, 30, viewMaskF120);

  cv::Mat freespaceMaskF120 =
      cv::imread(freespaceFileNames[refId], cv::IMREAD_GRAYSCALE);
  freespaceMaskF120 = freespaceMaskF120 > 0;

  cv::Mat stereoMaskF120 =
      cv::imread(stereoMaskFileNames[refId], cv::IMREAD_GRAYSCALE);
  stereoMaskF120 = stereoMaskF120 > 0;

  cv::Mat depthMaskF120 = viewMaskF120.clone();
  depthMaskF120 = depthMaskF120 & freespaceMaskF120;
  depthMaskF120 = depthMaskF120 & stereoMaskF120;

  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  makeOutputFolder("fisheyeTestResultsSaic");

  // Plane sweeping stereo
  {
    PSL::CudaFishEyePlaneSweep cFEPS;
    cFEPS.setScale(1.0);
    cFEPS.setMatchWindowSize(30, 30);
    cFEPS.setNumPlanes(1);
    cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_NONE);
    cFEPS.setPlaneGenerationMode(
        PSL::FISH_EYE_PLANE_SWEEP_PLANEMODE_UNIFORM_DEPTH_GROUND);
    cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_ZNCC);
    cFEPS.setSubPixelInterpolationMode(
        PSL::FISH_EYE_PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
    cFEPS.enableOutputBestDepth();
    cFEPS.enableOutputBestCosts();
    cFEPS.enableOuputUniquenessRatio();
    cFEPS.enableOutputCostVolume();
    cFEPS.enableOutputBestPlanes();
    cFEPS.enableSubPixel(false);

    double groundDeltaRange = 0.0;
    double minZ = -groundDeltaRange / 2.0;
    double maxZ = groundDeltaRange / 2.0;
    cFEPS.setZRange(minZ, maxZ);

    double rollRange = 0.0 * M_PI / 180.0;
    double pitchRange = 10.0 * M_PI / 180.0;
    cFEPS.setRollAngleRange(rollRange);
    cFEPS.setPitchAngleRange(pitchRange);
    cFEPS.setNumRollAngles(1);
    cFEPS.setNumPitchAngles(60);

    // undistort and add the images
    for (unsigned int i = 0; i < numCam; i++)
    {
      std::string imageFileName = imageFileNames[i];
      cv::Mat imageOrig = cv::imread(imageFileName, 0);

      if (imageOrig.empty())
      {
        PSL_THROW_EXCEPTION("Error loading image.")
      }

      devImg.allocatePitchedAndUpload(imageOrig);

      cFEIP.setInputImg(devImg, cams[i]);

      double k1 = dist_coeffs[i][0];
      double k2 = dist_coeffs[i][1];
      double p1 = dist_coeffs[i][2];
      double p2 = dist_coeffs[i][3];
      std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>>
          undistRes = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

      // show undistorted image
      cv::Mat undistImage;
      undistRes.first.download(undistImage);

      int id = cFEPS.addDeviceImage(undistRes.first, undistRes.second);
    }

    // undistort mask
    devImg.allocatePitchedAndUpload(depthMaskF120);
    cFEIP.setInputImg(devImg, cams[refId]);

    double k1 = dist_coeffs[refId][0];
    double k2 = dist_coeffs[refId][1];
    double p1 = dist_coeffs[refId][2];
    double p2 = dist_coeffs[refId][3];
    std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>>
        undistMask = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

    undistMask.first.download(depthMaskF120);

    makeOutputFolder("fisheyeTestResultsSaic/grayscaleZNCC");

    {
      cFEPS.process(refId);
      PSL::FishEyeDepthMap<float, double> fEDM;
      fEDM = cFEPS.getBestDepth();
      cv::Mat refImage = cFEPS.downloadImage(refId);
      cv::imshow("Reference Image", refImage);

      // masked refImage
      cv::Mat refImageMasked;
      refImage.copyTo(refImageMasked, depthMaskF120);
      cv::imshow("Masked Reference Image", refImageMasked);

      PSL::Grid<float> bestCosts;
      bestCosts = cFEPS.getBestCosts();
      PSL::displayGridZSliceAsImage(bestCosts, 0, (float)0.0, (float)1.0, 1,
                                    "Best Costs");

      PSL::Grid<int> bestPlanes;
      bestPlanes = cFEPS.getBestPlanes();
      cv::Mat_<int> sliceMat(bestPlanes.getHeight(), bestPlanes.getWidth(),
                             &bestPlanes(0, 0, 0));

      float numPlanes = cFEPS.getNumPlanes() * cFEPS.getNumRollAngles() *
                        cFEPS.getNumPitchAngles();
      cv::Mat bestPlanesImage = cv::Mat::zeros(bestPlanes.getHeight(),
                                               bestPlanes.getWidth(), CV_8UC1);
      cv::Mat bestPlanesIndicesImage = cv::Mat::zeros(
          bestPlanes.getHeight(), bestPlanes.getWidth(), CV_8UC1);
      std::vector<int> countPlanes(numPlanes, 0);
      for (int r = 0; r < bestPlanes.getHeight(); r++)
        for (int c = 0; c < bestPlanes.getWidth(); c++)
        {
          if (depthMaskF120.at<uchar>(r, c) > 0)
          {
            int planeInd = sliceMat.at<int>(r, c);
            bestPlanesImage.at<uchar>(r, c) =
                static_cast<uchar>(planeInd / numPlanes * 255.0);
            bestPlanesIndicesImage.at<uchar>(r, c) = planeInd;

            countPlanes[planeInd]++;
          }
        }
      cv::imshow("Best Planes Index", bestPlanesImage);

      // remove planes with the counts less than the threshold
      int planeNumThreshold = 2000;
      std::vector<int> filteredPlaneIndices;
      for (int i = 0; i < numPlanes; i++)
      {
        if (countPlanes[i] > planeNumThreshold)
        {
          filteredPlaneIndices.push_back(i);
          // cout << "valid plane index #" << i << ", " << countPlanes[i]
          //     << " points" << endl;
        }
      }

      cv::Mat filteredPlaneMask;
      for (int ind : filteredPlaneIndices)
      {
        cv::Mat tmpMask = bestPlanesIndicesImage == ind;
        if (filteredPlaneMask.empty())
          filteredPlaneMask = tmpMask;
        else
          filteredPlaneMask |= tmpMask;
      }

      cv::Mat filteredBestPlanesImage;
      bestPlanesImage.copyTo(filteredBestPlanesImage, filteredPlaneMask);
      cv::imshow("Filtered Best Planes Index", filteredBestPlanesImage);

      // update depth mask with filtered plane mask
      depthMaskF120 &= filteredPlaneMask;

      // show uniqueness ratios
      PSL::Grid<float> uniquenessRatios;
      uniquenessRatios = cFEPS.getUniquenessRatios();
      PSL::displayGridZSliceAsImage(uniquenessRatios, 0, 1,
                                    "Uniqueness Ratios");
      cv::waitKey(1);

      bool showCostVolume = false;

      if (showCostVolume)
      {
        PSL::Grid<float> costVolume;
        costVolume = cFEPS.getCostVolume();
        for (unsigned int i = 0; i < costVolume.getDepth(); i++)
        {
          PSL::displayGridZSliceAsImage(costVolume, i, (float)0.0, (float)1.0,
                                        30, "Cost Volume");
        }
      }

      makeOutputFolder(
          "fisheyeTestResultsSaic/grayscaleZNCC/NoOcclusionHandling/");
      cv::imwrite(
          "fisheyeTestResultsSaic/grayscaleZNCC/NoOcclusionHandling/refImg.png",
          refImage);
      float minDepth = 3.0;
      float maxDepth = 60.0;
      fEDM.saveInvDepthAsColorImage("fisheyeTestResultsSaic/grayscaleZNCC/"
                                    "NoOcclusionHandling/invDepthCol.png",
                                    minDepth, maxDepth);

      fEDM.displayInvDepthColored(minDepth, maxDepth, 100);

      cv::Mat colInvDepth;
      fEDM.getInvDepthColored(minDepth, maxDepth, colInvDepth);

      // show depth on the edges
      cv::Mat detectedEdges;
      cv::blur(refImage, detectedEdges, cv::Size(3, 3));
      cv::Canny(detectedEdges, detectedEdges, 30.0, 100.0, 3);

      cv::Mat edgeColInvDepth, edgeColInvDepthMasked;
      colInvDepth.copyTo(edgeColInvDepth, detectedEdges);
      edgeColInvDepth.copyTo(edgeColInvDepthMasked, depthMaskF120);

      cv::Mat edgeOnColInvDepth, edgeOnColInvDepthMasked;
      cv::Mat invDetectedEdges =
          cv::Mat::ones(detectedEdges.size(), detectedEdges.type()) * 255 -
          detectedEdges;
      colInvDepth.copyTo(edgeOnColInvDepth, invDetectedEdges);
      edgeOnColInvDepth.copyTo(edgeOnColInvDepthMasked, depthMaskF120);

      cv::imwrite("fisheyeTestResultsSaic/grayscaleZNCC/"
                  "NoOcclusionHandling/edgeColInvDepth.png",
                  edgeColInvDepthMasked);
      cv::imwrite("fisheyeTestResultsSaic/grayscaleZNCC/"
                  "NoOcclusionHandling/edgeOnColInvDepth.png",
                  edgeOnColInvDepthMasked);

      cv::imshow("detected edges", detectedEdges);
      cv::imshow("invert depth of the edges", edgeColInvDepthMasked);
      cv::imshow("edges on the depth", edgeOnColInvDepthMasked);
      cv::waitKey();

      // get pointCloud as PCL
      PointCloud::Ptr cloud;
      cloud = fEDM.getPointCloudColoredPCL(refImage, maxDepth, depthMaskF120);

      // filter the local points
      PointCloud::Ptr filteredLocalCloud(new PointCloud());

      // -- passThrough filter
      pcl::PassThrough<Point> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(0.1, 6.0);
      pass.filter(*filteredLocalCloud);

      // estimate local plane
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr planeInliners(new pcl::PointIndices);
      // Create the segmentation object
      pcl::SACSegmentation<Point> seg;
      // Optional
      seg.setOptimizeCoefficients(true);
      // Mandatory
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.05);

      seg.setInputCloud(filteredLocalCloud);
      seg.segment(*planeInliners, *coefficients);

      if (planeInliners->indices.size() == 0)
      {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return (-1);
      }

      cout << "Plane coefficients: " << coefficients->values[0] << " "
           << coefficients->values[1] << " " << coefficients->values[2] << " "
           << coefficients->values[3] << endl;

      cout << "Plane inliers: " << planeInliners->indices.size() << endl;

      PointCloud::Ptr localPlaneCloud(new PointCloud());
      for (int i = 0; i < planeInliners->indices.size(); i++)
      {
        localPlaneCloud->push_back(
            filteredLocalCloud->points[planeInliners->indices[i]]);
      }

      // save pointCloud as Ply file
      std::string pointCloudFile = "fisheyeTestResultsSaic/grayscaleZNCC/"
                                   "NoOcclusionHandling/pointCloud.ply";
      std::string localPointCloudFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudLocal.ply";
      std::string localPlaneCloudFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudLocalPlane.ply";

      pcl::PLYWriter writer;
      writer.write(pointCloudFile, *cloud, true);
      writer.write(localPointCloudFile, *filteredLocalCloud, true);
      writer.write(localPlaneCloudFile, *localPlaneCloud, true);

      // point cloud visualization
      bool showPointCloud = true;

      if (showPointCloud)
      {
        pcl::visualization::PCLVisualizer vis("vis");
        vis.addPointCloud(localPlaneCloud, "localPlaneCloud");

        vis.spin();
      }
    }
  }
}

void loadData(std::string dataFolder,
              std::vector<PSL::FishEyeCameraMatrix<double>>& cams,
              std::vector<std::vector<double>>& dist_coeffs,
              std::vector<std::string>& imageFileNames,
              std::vector<std::string>& freespaceFileNames,
              std::vector<std::string>& viewMaskFileNames,
              std::vector<std::string>& stereoMaskFileNames)
{
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

  // now load the image filenames
  std::string imageListFile = dataFolder + "/images.txt";

  std::ifstream imagesStream;
  imagesStream.open(imageListFile.c_str());

  if (!imagesStream.is_open())
  {
    PSL_THROW_EXCEPTION("Could not load images list file")
  }

  {
    std::string imageFileName;
    while (imagesStream >> imageFileName)
    {
      imageFileNames.push_back(dataFolder + "/" + imageFileName);
    }
  }

  if (imageFileNames.size() != numCam)
  {
    PSL_THROW_EXCEPTION("The dataset does not contain correct number of images")
  }

  // load freespace mask
  std::string freespaceListFile = dataFolder + "/freespace.txt";

  std::ifstream freespaceStream;
  freespaceStream.open(freespaceListFile.c_str());

  if (!freespaceStream.is_open())
  {
    PSL_THROW_EXCEPTION("Could not load freespace list file")
  }

  {
    std::string freespaceFileName;
    while (freespaceStream >> freespaceFileName)
    {
      freespaceFileNames.push_back(dataFolder + "/" + freespaceFileName);
    }
  }

  if (freespaceFileNames.size() != numCam)
  {
    PSL_THROW_EXCEPTION(
        "The dataset does not contain correct number of freespace masks")
  }

  // load view masks
  std::string viewMaskListFile = dataFolder + "/view_masks.txt";

  std::ifstream viewMaskStream;
  viewMaskStream.open(viewMaskListFile.c_str());

  if (!viewMaskStream.is_open())
  {
    PSL_THROW_EXCEPTION("Could not load view mask list file")
  }

  {
    std::string viewMaskFileName;
    while (viewMaskStream >> viewMaskFileName)
    {
      viewMaskFileNames.push_back(dataFolder + "/" + viewMaskFileName);
    }
  }

  if (viewMaskFileNames.size() != numCam)
  {
    PSL_THROW_EXCEPTION(
        "The dataset does not contain correct number of view masks")
  }

  // load stereo masks
  std::string stereoMaskListFile = dataFolder + "/stereo_masks.txt";

  std::ifstream stereoMaskStream;
  stereoMaskStream.open(stereoMaskListFile.c_str());

  if (!stereoMaskStream.is_open())
  {
    PSL_THROW_EXCEPTION("Could not load stereo mask list file")
  }

  {
    std::string stereoMaskFileName;
    while (stereoMaskStream >> stereoMaskFileName)
    {
      stereoMaskFileNames.push_back(dataFolder + "/" + stereoMaskFileName);
    }
  }

  if (stereoMaskFileNames.size() != numCam)
  {
    PSL_THROW_EXCEPTION(
        "The dataset does not contain correct number of stereo masks")
  }
}

void erodeMask(cv::Mat& inMask, int kernelSize, cv::Mat& outMask)
{
  outMask = inMask.clone();

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));

  cv::morphologyEx(inMask, outMask, cv::MORPH_ERODE, element);
}

void dilateMask(cv::Mat& inMask, int kernelSize, cv::Mat& outMask)
{
  outMask = inMask.clone();

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));

  cv::morphologyEx(inMask, outMask, cv::MORPH_DILATE, element);
}
