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
    cout << "Usage: ./searchScanLocation locationDataFile.txt "
            "point_clouds.txt" << endl;
    return 1;
  }

  cout << "Current working directory: " << GetCurrentWorkingDir() << endl;

  string locationDataFilePath = argv[1];
  string scanTimestampPath = argv[2];

  ifstream locationDataFile(locationDataFilePath.c_str());
  ifstream scanTimestamp(scanTimestampPath.c_str());

  string scanLocationPath = "data/test/scanLocation.txt";
  ofstream scanLocation(scanLocationPath.c_str());

  if (!locationDataFile.is_open() || !scanTimestamp.is_open() ||
      !scanLocation.is_open())
  {
    cout << "Cannot open data file(s)!" << endl;
    return 1;
  }

  // output description
  cout << "Output path: " << scanLocationPath << endl;
  scanLocation << "# timestamp x y z roll pitch yaw" << endl;

  // load data
  vector<Pose> locations;
  vector<uint64_t> timestampScans;

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

  while (getline(scanTimestamp, line))
  {
    uint64_t timestamp = atoll(line.substr(0, line.size() - 4).c_str());
    timestamp /= 1e3;

    timestampScans.push_back(timestamp);
  }

  cout << "Num of loaded scan timestamps: " << timestampScans.size() << endl;

  int curLocationInd = 0;
  for (int indTime = 0; indTime < timestampScans.size(); indTime++)
  {
    uint64_t curTimestamp = timestampScans[indTime];

    if (curTimestamp < locations[0].timestamp)
    {
      cout << "Warning: ignore scan timestamp less than the first location "
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

        scanLocation << curTimestamp << ' ';
        scanLocation << std::fixed << std::setprecision(6) << tmpLocation.x
                     << ' ' << tmpLocation.y << ' ' << tmpLocation.z << ' '
                     << tmpLocation.roll << ' ' << tmpLocation.pitch << ' '
                     << tmpLocation.yaw << endl;

        // update current location index
        curLocationInd = indLoc;

        break;
      }
    }
  }

  scanLocation.close();
  cout << "Finished!" << endl;

  return 0;
}
