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
#include <opencv2/flann/miniflann.hpp>
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
#include <pcl/kdtree/kdtree_flann.h>

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
              std::vector<PSL::FishEyeCameraMatrix<double>> &cams,
              std::vector<std::vector<double>> &dist_coeffs,
              Eigen::Vector4d &plane_in_lidar_frame,
              std::vector<std::string> &imageFileNames,
              std::vector<std::string> &freespaceFileNames,
              std::vector<std::string> &viewMaskFileNames,
              std::vector<std::string> &stereoMaskFileNames);

void erodeMask(cv::Mat &inMask, int kernelSize, cv::Mat &outMask);
void dilateMask(cv::Mat &inMask, int kernelSize, cv::Mat &outMask);

bool unprojectToPlane(const PSL::FishEyeCameraMatrix<double> &cam, cv::Point2f p2D, 
                      cv::Mat &image, Eigen::Vector3d n, double d, double maxDepth, 
                      Point& p3D);

int main(int argc, char *argv[])
{
  std::string dataFolder;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message")(
      "dataFolder", boost::program_options::value<std::string>(&dataFolder)->default_value("DataFisheyeCamera/right"),
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
  Eigen::Vector4d plane_in_lidar_frame;
  std::vector<std::string> imageFileNames, freespaceFileNames,
      viewMaskFileNames, stereoMaskFileNames;

  loadData(dataFolder, cams, dist_coeffs, plane_in_lidar_frame, imageFileNames, 
           freespaceFileNames, viewMaskFileNames, stereoMaskFileNames);

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
    double height = 1.6;
    double minZ = -groundDeltaRange / 2.0 + height;
    double maxZ = groundDeltaRange / 2.0 + height;
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
    depthMaskF120 = depthMaskF120 > 0;

    // -- clear the border of the mask in case of the undistortion effect
    int borderWidth = 5;
    cv::Mat borderMask = cv::Mat::zeros(depthMaskF120.rows, depthMaskF120.cols,
                                        depthMaskF120.type());
    cv::rectangle(borderMask, cv::Rect(borderWidth, borderWidth, depthMaskF120.cols - 2 * borderWidth, depthMaskF120.rows - 2 * borderWidth),
                  cv::Scalar(255), -1);

    {
      cv::Mat tmpMask;
      depthMaskF120.copyTo(tmpMask, borderMask);
      depthMaskF120 = tmpMask;
    }

    // extract the boundary of the freespace
    cv::Mat erodedFreespaceMask;
    erodeMask(freespaceMaskF120, 1, erodedFreespaceMask);
    cv::Mat boundaryFreespace = freespaceMaskF120 - erodedFreespaceMask;
    cv::Mat boundaryFreespaceMasked;
    boundaryFreespace.copyTo(boundaryFreespaceMasked, viewMaskF120);

    // -- undistort boundary
    devImg.allocatePitchedAndUpload(boundaryFreespaceMasked);
    cFEIP.setInputImg(devImg, cams[refId]);

    std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>>
        undistBoundary = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

    undistBoundary.first.download(boundaryFreespaceMasked);
    boundaryFreespaceMasked = boundaryFreespaceMasked > 0;

    // -- apply border mask
    {
      cv::Mat tmpMask;
      boundaryFreespaceMasked.copyTo(tmpMask, borderMask);
      boundaryFreespaceMasked = tmpMask;
    }

    std::vector<cv::Point2f> freespaceBoundaryPoints;
    for (int r = 0; r < boundaryFreespaceMasked.rows; r++)
      for (int c = 0; c < boundaryFreespaceMasked.cols; c++)
        if (boundaryFreespaceMasked.at<uchar>(r, c) > 0)
          freespaceBoundaryPoints.push_back(cv::Point2f(c, r));

    cout << "Boundary freespace points: " << freespaceBoundaryPoints.size()
         << endl;

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

      // find the consecutive planes that contain the ground plane
      // -- find the plane with maximum pixels
      int planeNumThreshold = 500;
      int maxPlaneNumInd = -1;
      int maxPlaneNum = 0;
      for (int i = 0; i < numPlanes; i++)
      {
        int cnt = countPlanes[i];
        if (cnt > planeNumThreshold && cnt > maxPlaneNum)
        {
          maxPlaneNumInd = i;
          maxPlaneNum = cnt;
        }
      }

      if (maxPlaneNumInd < 0)
      {
        cout << "Failed to find enough planes more than the threshold." << endl;
        return 1;
      }

      // -- find the consecutive planes on both sides
      std::vector<int> filteredPlaneIndices;
      filteredPlaneIndices.push_back(maxPlaneNumInd);

      for (int i = maxPlaneNumInd - 1; i >= 0; i--)
      {
        if (countPlanes[i] < planeNumThreshold)
          break;
        else
          filteredPlaneIndices.push_back(i);
      }

      for (int i = maxPlaneNumInd + 1; i < numPlanes; i++)
      {
        if (countPlanes[i] < planeNumThreshold)
          break;
        else
          filteredPlaneIndices.push_back(i);
      }

      // -- construct the mask
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
      float maxDepth = 100.0;
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

      // show freespace boundary on depth image
      for (auto p : freespaceBoundaryPoints)
        cv::circle(edgeOnColInvDepthMasked, p, 1, cv::Scalar(255, 255, 255),
                   -1);

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
      pass.setFilterFieldName("y");
      pass.setFilterLimits(0.1, 15.0);
      pass.filter(*filteredLocalCloud);

      pass.setInputCloud(filteredLocalCloud);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(-10.0, 10.0);
      pass.filter(*filteredLocalCloud);

      // estimate local plane
      pcl::ModelCoefficients::Ptr coefficientsLocal(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr planeInliners(new pcl::PointIndices);
      // Create the segmentation object
      pcl::SACSegmentation<Point> seg;
      // Optional
      seg.setOptimizeCoefficients(true);
      // Mandatory
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.3);

      seg.setInputCloud(filteredLocalCloud);
      seg.segment(*planeInliners, *coefficientsLocal);

      if (planeInliners->indices.size() == 0)
      {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return (-1);
      }
      cout << "Plane inliers: " << planeInliners->indices.size() << endl;

      cout << "Local plane: " << coefficientsLocal->values[0] << " "
           << coefficientsLocal->values[1] << " "
           << coefficientsLocal->values[2] << " "
           << coefficientsLocal->values[3] << endl;

      PointCloud::Ptr localPlaneCloud(new PointCloud());
      for (int i = 0; i < planeInliners->indices.size(); i++)
      {
        localPlaneCloud->push_back(
            filteredLocalCloud->points[planeInliners->indices[i]]);
      }

      // estimate plane for each freespace boundary point with its neighborhoods
      // -- get points with valid depth
      std::vector<cv::Point2f> nonZeroLocation;
      for (int r = 0; r < depthMaskF120.rows; r++)
        for (int c = 0; c < depthMaskF120.cols; c++)
          if (depthMaskF120.at<uchar>(r, c) > 0)
            nonZeroLocation.push_back(cv::Point2f(c, r));

      // -- construct kd-tree
      cv::flann::KDTreeIndexParams indexParams;
      cv::flann::Index kdtree(cv::Mat(nonZeroLocation).reshape(1), indexParams);
      cv::flann::SearchParams searchParams;

      // -- get organized cloud from the depth image
      PointCloud::Ptr cloudOrganized;
      cloudOrganized =
          fEDM.getPointCloudColoredOrganizedPCL(refImage, 300.0, depthMaskF120);

      // -- iterate each point
      int neighborhoodNum = 500;

      std::vector<pcl::ModelCoefficients::Ptr> freespacePlanesCoefficients;
      std::vector<cv::Point2f> validFreespaceBoundaryPoints;
      PointCloud::Ptr freespaceCloud(new PointCloud());
      PointCloud::Ptr freespaceCloudIPM(new PointCloud());
      PointCloud::Ptr freespaceCloudAdaptiveIPM(new PointCloud());

      // -- set parameters of the ground plane for IPM and adaptive IPM
      Eigen::Vector3d n_IPM;
      n_IPM[0] = plane_in_lidar_frame[0];
      n_IPM[1] = plane_in_lidar_frame[1];
      n_IPM[2] = plane_in_lidar_frame[2];

      double d_IPM = plane_in_lidar_frame[3];

      cout << "IPM ground plane: \n" 
            << "n: " << n_IPM.transpose() << "\n"
            << "d: " << d_IPM << endl;

      Eigen::Vector3d n_adaptive_IPM;
      n_adaptive_IPM[0] = coefficientsLocal->values[0];
      n_adaptive_IPM[1] = coefficientsLocal->values[1];
      n_adaptive_IPM[2] = coefficientsLocal->values[2];

      double d_adaptive_IPM = coefficientsLocal->values[3];

      cout << "Adaptive IPM ground plane: \n" 
            << "n: " << n_adaptive_IPM.transpose() << "\n"
            << "d: " << d_adaptive_IPM << endl;

      for (int i = 0; i < freespaceBoundaryPoints.size(); i++)
      {
        // find neighborhoods
        cv::Point2f queryPoint = freespaceBoundaryPoints[i];
        std::vector<float> query;
        query.push_back(queryPoint.x);
        query.push_back(queryPoint.y);
        std::vector<int> indices;
        std::vector<float> dists;
        kdtree.knnSearch(query, indices, dists, neighborhoodNum, searchParams);

        // skip the points too far away
        cv::Point2f furthestPoint = nonZeroLocation[indices[0]];
        float depth = fEDM(furthestPoint.x, furthestPoint.y);
        if (depth > maxDepth)
          continue;

        // get local cloud
        PointCloud::Ptr localCloud(new PointCloud());
        for (int j = 0; j < indices.size(); j++)
        {
          cv::Point2f point2D = nonZeroLocation[indices[j]];
          Point point3D =
              cloudOrganized
                  ->points[point2D.y * cloudOrganized->width + point2D.x];

          if (std::isnan(point3D.x))
            continue;

          localCloud->points.push_back(point3D);
        }

        // skip if not enough points in local cloud
        int minNumPoints = neighborhoodNum / 2;
        if (localCloud->points.size() < minNumPoints)
          continue;

        // estimate plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr planeInliners(new pcl::PointIndices);
        pcl::SACSegmentation<Point> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.5);

        seg.setInputCloud(localCloud);
        seg.segment(*planeInliners, *coefficients);

        if (planeInliners->indices.size() == 0)
        {
          cout << "Warning: Failed to estimate the local plane for freespace "
                  "boundary point #"
               << i << endl;
          continue;
        }

        // add to buffer
        freespacePlanesCoefficients.push_back(coefficients);
        validFreespaceBoundaryPoints.push_back(queryPoint);

        // unproject to the estimated plane (stereo)
        {
          Eigen::Vector3d n;
          n[0] = coefficients->values[0];
          n[1] = coefficients->values[1];
          n[2] = coefficients->values[2];

          double d = coefficients->values[3];

          // unproject and add to cloud
          Point pt;
          if (unprojectToPlane(fEDM.getCam(), queryPoint, refImage, 
                               n, d, maxDepth, pt))
            freespaceCloud->points.push_back(pt);
        }

        // unproject to the ground plane (IPM)
        {
          // unproject and add to cloud
          Point pt;
          if (unprojectToPlane(fEDM.getCam(), queryPoint, refImage, 
                               n_IPM, d_IPM, maxDepth, pt))
            freespaceCloudIPM->points.push_back(pt);
        }

        // unproject to the estiated local ground plane (adaptive IPM)
        {
          // unproject and add to cloud
          Point pt;
          if (unprojectToPlane(fEDM.getCam(), queryPoint, refImage, 
                               n_adaptive_IPM, d_adaptive_IPM, maxDepth, pt))
            freespaceCloudAdaptiveIPM->points.push_back(pt);
        }
      }

      cout << "Valid freespace points num: " << freespaceCloud->points.size()
           << endl;

      // save pointCloud as Ply file
      std::string pointCloudFile = "fisheyeTestResultsSaic/grayscaleZNCC/"
                                   "NoOcclusionHandling/pointCloud.ply";
      std::string localPointCloudFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudLocal.ply";
      std::string localPlaneCloudFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudLocalPlane.ply";
      std::string freespaceCloudFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudFreespace.ply";
      std::string freespaceCloudIPMFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudFreespaceIPM.ply";
      std::string freespaceCloudAdaptiveIPMFile =
          "fisheyeTestResultsSaic/grayscaleZNCC/"
          "NoOcclusionHandling/pointCloudFreespaceAdaptiveIPM.ply";

      pcl::PLYWriter writer;
      writer.write(pointCloudFile, *cloud, true);
      writer.write(localPointCloudFile, *filteredLocalCloud, true);
      writer.write(localPlaneCloudFile, *localPlaneCloud, true);
      writer.write(freespaceCloudFile, *freespaceCloud, true);
      writer.write(freespaceCloudIPMFile, *freespaceCloudIPM, true);
      writer.write(freespaceCloudAdaptiveIPMFile, *freespaceCloudAdaptiveIPM,
                   true);

      // point cloud visualization
      bool showPointCloud = false;

      if (showPointCloud)
      {
        pcl::visualization::PCLVisualizer vis("vis");
        vis.addPointCloud(localPlaneCloud, "localPlaneCloud");
        vis.addPointCloud(freespaceCloud, "freespaceCloud");
        vis.addPointCloud(freespaceCloudIPM, "freespaceCloudIPM");
        vis.addPointCloud(freespaceCloudAdaptiveIPM,
                          "freespaceCloudAdaptiveIPM");

        vis.spin();
      }
    }
  }
}

void loadData(std::string dataFolder,
              std::vector<PSL::FishEyeCameraMatrix<double>> &cams,
              std::vector<std::vector<double>> &dist_coeffs,
              Eigen::Vector4d &plane_in_lidar_frame,
              std::vector<std::string> &imageFileNames,
              std::vector<std::string> &freespaceFileNames,
              std::vector<std::string> &viewMaskFileNames,
              std::vector<std::string> &stereoMaskFileNames)
{
  std::string calibFileName = dataFolder + "/calib.txt";

  std::ifstream calibrationStr;
  calibrationStr.open(calibFileName.c_str());

  if (!calibrationStr.is_open())
  {
    PSL_THROW_EXCEPTION("Error opening calibration file calib.txt.")
  }

  std::string desc;

  int numCam;
  calibrationStr >> desc >> numCam;

  for (unsigned int i = 0; i < numCam; i++)
  {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    double xi;
    std::vector<double> dist_coeff(4);

    calibrationStr >> desc;

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

    Eigen::Vector3d t(t_x, t_y, t_z);

    cams.push_back(PSL::FishEyeCameraMatrix<double>(K, R.transpose(),
                                                    -R.transpose() * t, xi));
    dist_coeffs.push_back(dist_coeff);
  }

  // load tranformation from camera to vehicle (used for IPM with ground plane)
  int selected_cam_index;
  calibrationStr >> desc >> selected_cam_index;

  double roll, pitch, yaw, t_x, t_y, t_z;
  calibrationStr >> desc >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;
  Eigen::Matrix3d R_cam2vehicle;
  R_cam2vehicle = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t_cam2vehicle(t_x, t_y, t_z);

  Eigen::Matrix4d T_cam2vehicle = Eigen::Matrix4d::Identity();
  T_cam2vehicle.topLeftCorner(3, 3) = R_cam2vehicle;
  T_cam2vehicle.topRightCorner(3, 1) = t_cam2vehicle;

  // -- get transformation from vehicle to lidar
  Eigen::Matrix4d T_lidar2cam = Eigen::Matrix4d::Identity();
  T_lidar2cam.topLeftCorner(3, 3) = cams[selected_cam_index].getR();
  T_lidar2cam.topRightCorner(3, 1) = cams[selected_cam_index].getT();

  Eigen::Matrix4d T_vehicle2lidar = T_lidar2cam.inverse() * T_cam2vehicle.inverse();

  // -- transform the ground plane vehicle frame to lidar frame
  // ref: https://math.stackexchange.com/questions/2502857/transform-plane-to-another-coordinate-system
  Eigen::Vector4d plane_in_vehicle_frame;
  plane_in_vehicle_frame << 0, 0, 1, 0;
  plane_in_lidar_frame = T_vehicle2lidar.inverse().transpose() * plane_in_vehicle_frame;

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

void erodeMask(cv::Mat &inMask, int kernelSize, cv::Mat &outMask)
{
  outMask = inMask.clone();

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));

  cv::morphologyEx(inMask, outMask, cv::MORPH_ERODE, element);
}

void dilateMask(cv::Mat &inMask, int kernelSize, cv::Mat &outMask)
{
  outMask = inMask.clone();

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));

  cv::morphologyEx(inMask, outMask, cv::MORPH_DILATE, element);
}

bool unprojectToPlane(const PSL::FishEyeCameraMatrix<double> &cam, cv::Point2f p2D, 
                      cv::Mat &image, Eigen::Vector3d n, double d, double maxDepth, 
                      Point& p3D)
{
  // compute point's 3D position
  Eigen::Vector3d pointRay;
  pointRay =
      cam.unprojectPointToRay(p2D.x, p2D.y);

  Eigen::Matrix3d R;
  R = cam.getR();
  Eigen::Vector3d t;
  t = cam.getT();

  double scaleFactor = (-d + (n.transpose() * R.transpose() * t)[0]) /
                        (n.transpose() * R.transpose() * pointRay)[0];
  pointRay *= scaleFactor;

  // check if the unprojection is valid
  if (scaleFactor < 0 || pointRay[2] > maxDepth)
    return false;

  Eigen::Vector4d point;
  point = cam.localPointToGlobal(pointRay[0], pointRay[1],
                                            pointRay[2]);

  // output the 3D point with color
  p3D.x = point[0];
  p3D.y = point[1];
  p3D.z = point[2];

  if (image.channels() == 3)
  {
    cv::Vec3b pixel = image.at<cv::Vec3b>(p2D);
    p3D.b = pixel[0];
    p3D.g = pixel[1];
    p3D.r = pixel[2];
  }
  else if (image.channels() == 1)
  {
    uchar pixel = image.at<uchar>(p2D);
    p3D.b = pixel;
    p3D.g = pixel;
    p3D.r = pixel;
  }

  return true;
}