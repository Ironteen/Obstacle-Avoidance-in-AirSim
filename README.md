# A simple Reinforcement Learning Demo for Obstacle Avoidance  of using Microsoft AirSim

This repository contains Python scripts showing how you can use [Microsoft AirSim](https://github.com/Microsoft/AirSim) to  collect image data from a moving vehicle, then use that data to train the vehicle to avoid obstacles  in TensorFlow. The RL  algorithm we used is D3QN(Double Deep Q Network with Dueling architecture)。

![screenshot](https://github.com/Ironteen/Obstacle-Avoidance-in-AirSim/blob/master/img/screenshot.png)

## Prerequisites

- [Recommended hardware](https://wiki.unrealengine.com/Recommended_Hardware) for running UnrealEngine4, required for AirSim. Although it is possible build AirSim on OS X and Linux, we found it easiest to use the pre-compiled Windows binaries.
- This map we show aboved is a simple demo,which was built on Block.
- [Python3](https://www.python.org/ftp/python/3.6.3/python-3.6.3-amd64.exe) for 64-bit Windows
- [TensorFlow](https://www.tensorflow.org/install/install_windows). To run TensorFlow on your GPU as we and most people do, you'll need to follow the [directions](https://www.tensorflow.org/install/install_windows) for installing CUDA and CuDNN. We recommend setting aside at least an hour to make sure you do this right.

## Document

- ```
  D3QN_training_standard.py
  ```

  This script is a standard Monocular-Obstacle-Avoidance training program. With only a monocular, the moving vehicle can learning to avoid obstacles.

- ```
  D3QN_training_Lidar.py
  ```

  With a monocular and lidar, the moving vehicle can learning to avoid obstacles more efficiently.

- ```
  D3QN_testing_Lidar_grid.py
  ```

   When the vehicle is well trained,  you can run this test program. When running,  the car records the explored space in a grid map simultaneously, which will be saved as .pkl file in the same path.


## Instructions

1. Clone this repository.

2. Open or build a map, set the SimMode:"Car"  in the setting.json, and then run it.

3. Choose a train model and modified the destination coordinate, then run 

   ```
   python 3QN_training_standard.py or python D3QN_training_Lidar.py
   ```

   It will take a long time. if you choose training your car with lidar, it will be more efficient.

4. when you find the moving vehicle trained well, then run

   ```
   python D3QN_testing_Lidar_grid.py
   ```

   The car is testing without lidar for the lidar is just a auxiliary tools in training task

5. It's a simple demo for Obstacle Avoidance with D3QN, you can change the structure with DDPG or A3C quite easily.

## show our result

![Obstacle-Avoidance](https://github.com/Ironteen/Obstacle-Avoidance-in-AirSim/blob/master/img/Obstacle-voidance.gif)

The average steps in a episode

| methods             | Steps |
| ------------------- | ----- |
| D3QN training       | 850   |
| D3QN+Lidar training | 3200  |
| Test without Lidar  | 2185  |

Note: When the moving vehicle reached the destination or has collided, a episode is over.

## Acknowledgement

This code repository is highly inspired from work of  Linhai Xie, Sen Wang, Niki trigoni, Andrew Markham 

[[link\]]: https://github.com/xie9187/Monocular-Obstacle-Avoidance
