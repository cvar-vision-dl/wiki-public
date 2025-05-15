# Using two different RealSense Cameras at the same time
In this document there will be an explanation on how to use T265 and D436i as an example with Aerostack2.

## Installation

1. Install and configure [Aerostack2](https://aerostack2.github.io/_00_getting_started/source_install.html).

2. Re-install librealsense

```
# First uninstall ros-humble-realsense package
sudo apt-get remove ros-humble-librealsense2 -y
 
# install an older version
 
 
# sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
# sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main" -u
# sudo apt-get install librealsense2=2.53.1-0~realsense0.703  -y  --allow-downgrades
# sudo apt-get install librealsense2-gl=2.53.1-0~realsense0.703  -y --allow-downgrades 
# sudo apt-get install librealsense2-utils=2.53.1-0~realsense0.703  -y --allow-downgrades
# sudo apt-get install librealsense2-dev=2.53.1-0~realsense0.703  -y --allow-downgrades
 
cd $HOME
git clone https://github.com/IntelRealSense/librealsense.git -b v2.50.0
cd librealsense
mkdir build
cd build
cmake .. -DFORCE_RSUSB_BACKEND=1
make -j8
sudo make install
 
cd ~
sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
```
## Running both cameras
1. Run T265

`ros2 launch as2_realsense_interface as2_realsense_interface_launch.py device:=t265`

2. Run D435i

`ros2 launch as2_realsense_interface as2_realsense_interface_launch.py device:=d435i`

3.  Run remapping of topics

`ros2 run topic_tools relay /drone0/sensor_measurements/realsense/odom /drone0/sensor_measurements/odom`

4. Run state estimator

`ros2 launch as2_state_estimator state_estimator_launch.py namespace:=drone0 plugin_name:=raw_odometry odom_topic:=/drone0/sensor_measurements/realsense/odom`
