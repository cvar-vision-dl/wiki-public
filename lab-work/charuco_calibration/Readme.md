# Charuco Calibration Script
This folder contains a Python3 script for calibrating a camera from images stored in a ```datadir```. Images must be in ```.png``` format.

## Requirements.
This script needs opencv and matplotlib python libs to work.
Those can be installed in the system using 
```
$ sudo apt install python3-opencv python3-matplotlib
```

If you prefer to use a virtual env use 

```
$ pip3 install opencv-python 
$ pip3 install matplot lib 
```

## Usage

The file considers the following configuration of the Aruco Board:

```
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(11, 11, .1, .08, aruco_dict)
```
Check [Opencv Charuco Board Documentation](https://docs.opencv.org/3.4/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html) for more information.
> If you want to modify it change lines 128 and 129 of the python script.

Here all the possible options of the script are provided.


```
usage: charucoCalib.py [-h] --datadir DATADIR [--show-aruco-board] [--show-undistorted] [--minimun-detected-markers MINIMUN_DETECTED_MARKERS] [--all_distortion_coefficients]

Calibrate camera using charuco board.

options:
  -h, --help            show this help message and exit
  --datadir DATADIR     Directory containing images.
  --show-aruco-board    Show the aruco board.
  --show-undistorted    Show the undistorted images.
  --minimun-detected-markers MINIMUN_DETECTED_MARKERS
                        Minimum number of markers to detect in the image.
  --all_distortion_coefficients
                        Show all distortion coefficients.
```