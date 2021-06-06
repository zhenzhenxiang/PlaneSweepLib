# Vision based and vision-LiDAR based freespace fusion

## Description

This project provides two functions:

1. Multi-cam freespace fusion based on dense depth estimation with [plane sweeping method](https://github.com/bastienjacquet/PlaneSweepLib);
2. Vision-LiDAR based freespace fusion based on cam-LiDAR calibration and back-projection.

## Examples

### Multi-cam freespace fusion

```
./fisheyePlanesweepTestSaic --dataFolder path/to/data/fisheyeCamera/902-Four-Cam
```

### Vision-LiDAR freespace fusion

```
./pointCloudProjectionFromPly path/to/data/test/projectTestPly.txt
```
