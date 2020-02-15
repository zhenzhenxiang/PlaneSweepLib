#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <psl_base/exception.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

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

int main(int argc, char** argv)
{

  int startInd = 1000;
  int length = 5;
  cv::Size frameSize = cv::Size(1920, 1208);
  bool outputFrames = false;

  if (argc != 3)
  {
    cout << "Usage: ./createVideoSequence video.h264 cameraFrameLocation.txt"
         << endl;
    return 1;
  }

  string videoFilePath = argv[1];
  string posesFilePath = argv[2];

  // open files
  cv::VideoCapture cap(videoFilePath);
  ifstream posesFile(posesFilePath.c_str());

  if (!cap.isOpened() || !posesFile.is_open())
  {
    cout << "Cannot open video or poses file!" << endl;
    return 1;
  }

  // skip comment lines
  string line;
  std::getline(posesFile, line);

  // prepare output files
  cv::VideoWriter outputVideo("data/test/video_seq.avi",
                              cv::VideoWriter::fourcc('F', 'F', 'V', '1'), 5,
                              frameSize, true);
  string videoSeqFramesFolder = "data/test/video_seq/";
  if (outputFrames)
    makeOutputFolder(videoSeqFramesFolder);

  ofstream posesSeqFile("data/test/poses_seq.txt");
  posesSeqFile << "# timestamp x y z roll pitch yaw" << endl;

  ofstream imageListFile("data/test/images.txt");

  // main loop
  int frameCnt = 0;
  cv::Mat image;
  Matrix4d initPose = Matrix4d::Identity();
  while (true)
  {
    // skip some frames
    if (frameCnt < startInd)
    {
      cap >> image;
      std::getline(posesFile, line);

      frameCnt++;
      continue;
    }

    if (frameCnt >= startInd + length)
    {
      cout << "Generating video and poses sequence finished!" << endl;
      break;
    }
    else
    {
      std::getline(posesFile, line);

      std::stringstream lineStr(line);
      uint64_t timestamp;
      double roll, pitch, yaw, t_x, t_y, t_z;
      lineStr >> timestamp >> t_x >> t_y >> t_z >> roll >> pitch >> yaw;

      Matrix3d R;
      R = AngleAxisd(yaw, Vector3d::UnitZ()) *
          AngleAxisd(pitch, Vector3d::UnitY()) *
          AngleAxisd(roll, Vector3d::UnitX());

      Vector3d T(t_x, t_y, t_z);

      Matrix4d curPose = Matrix4d::Identity();
      curPose.topLeftCorner(3, 3) = R;
      curPose.topRightCorner(3, 1) = T;

      if (frameCnt == startInd)
      {
        initPose = curPose;

        R = Matrix3d::Identity();
        T = Vector3d::Zero();
      }
      else
      {
        Matrix4d relPose = initPose.inverse() * curPose;
        R = relPose.topLeftCorner(3, 3);
        T = relPose.topRightCorner(3, 1);
      }

      Vector3d angles = R.eulerAngles(2, 1, 0);

      posesSeqFile << timestamp << ' ';
      posesSeqFile << std::fixed << std::setprecision(6) << T(0) << ' ' << T(1)
                   << ' ' << T(2) << ' ' << angles(2) << ' ' << angles(1) << ' '
                   << angles(0) << endl;

      cap >> image;
      outputVideo << image;

      if (outputFrames)
      {
        string imageName =
            videoSeqFramesFolder + std::to_string(timestamp) + ".jpg";
        cv::imwrite(imageName, image);
      }

      imageListFile << std::to_string(timestamp) + ".jpg" << endl;

      frameCnt++;
    }
  }

  posesFile.close();
  posesSeqFile.close();

  return 0;
}
