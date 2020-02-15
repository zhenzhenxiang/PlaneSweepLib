#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <stdint.h>

#define GetCurrentDir getcwd

using namespace std;

std::string GetCurrentWorkingDir(void)
{
  char buff[FILENAME_MAX];
  GetCurrentDir(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

struct Pose
{
  uint64_t timestamp;
  float x;
  float y;
  float z;
  float roll;
  float pitch;
  float yaw;
};

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    cout << "Usage: ./searchCameraLocation locationDataFile.txt "
            "cameraTimestamp.txt" << endl;
    return 1;
  }

  cout << "Current working directory: " << GetCurrentWorkingDir() << endl;

  string locationDataFilePath = argv[1];
  string cameraTimestampPath = argv[2];

  ifstream locationDataFile(locationDataFilePath.c_str());
  ifstream cameraTimestamp(cameraTimestampPath.c_str());

  string cameraFrameLocationPath = "data/test/cameraFrameLocation.txt";
  ofstream cameraFrameLocation(cameraFrameLocationPath.c_str());

  if (!locationDataFile.is_open() || !cameraTimestamp.is_open() ||
      !cameraFrameLocation.is_open())
  {
    cout << "Cannot open data file(s)!" << endl;
    return 1;
  }

  // output description
  cout << "Output path: " << cameraFrameLocationPath << endl;
  cameraFrameLocation << "# timestamp x y z roll pitch yaw" << endl;

  // load data
  vector<Pose> locations;
  vector<uint64_t> timestampCams;

  string line;
  while (getline(locationDataFile, line))
  {
    if (line[0] == '#')
      continue;

    Pose loc;
    std::stringstream lineStr(line);
    lineStr >> loc.timestamp >> loc.x >> loc.y >> loc.z >> loc.roll >>
        loc.pitch >> loc.yaw;

    locations.push_back(loc);
  }

  cout << "Num of loaded locations: " << locations.size() << endl;

  while (getline(cameraTimestamp, line))
  {
    int ind;
    uint64_t timestamp;

    std::stringstream lineStr(line);
    lineStr >> ind >> timestamp;

    timestampCams.push_back(timestamp);
  }

  cout << "Num of loaded camera timestamps: " << timestampCams.size() << endl;

  int curLocationInd = 0;
  for (int indTime = 0; indTime < timestampCams.size(); indTime++)
  {
    uint64_t curTimestamp = timestampCams[indTime];

    if (curTimestamp < locations[0].timestamp)
    {
      cout << "Warning: ignore camera timestamp less than the first location "
              "timestamp!" << endl;
      continue;
    }

    for (int indLoc = curLocationInd; indLoc < locations.size(); indLoc++)
    {
      if (locations[indLoc].timestamp > curTimestamp)
      {
        int preTimeInterval = curTimestamp - locations[indLoc - 1].timestamp;
        int subsequentTimeInterval = locations[indLoc].timestamp - curTimestamp;
        Pose tmpLocation = (preTimeInterval < subsequentTimeInterval)
                               ? locations[indLoc - 1]
                               : locations[indLoc];

        cameraFrameLocation << curTimestamp << ' ';
        cameraFrameLocation
            << std::fixed << std::setprecision(6) << tmpLocation.x << ' '
            << tmpLocation.y << ' ' << tmpLocation.z << ' ' << tmpLocation.roll
            << ' ' << tmpLocation.pitch << ' ' << tmpLocation.yaw << endl;

        // update current location index
        curLocationInd = indLoc;

        break;
      }
    }
  }

  cameraFrameLocation.close();
  cout << "Finished!" << endl;

  return 0;
}
