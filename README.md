# Design Methodology for Deep Out-of-Distribution Detectors in Real-Time Cyber-Physical Systems
This branch contains the implementation for the RTCSA2022 paper: *Design Methodology for Deep Out-of-Distribution Detectors in Real-Time Cyber-Physical Systems*.  To reproduce the results as they appear in the paper, you will also require the dataset, which can be found [here](https://<TODO>).

## Dependencies

### Model Training
All models were trained using PyTorch 1.11.0 with CUDA 11.6.  Image transforms were performed with Torchvision (0.12.0) or OpenCV (4.5.5.64).  Unless otherwise stated, the Cpython interpretor for Python 3.9 was used to run all scripts.  The ```training/requirements.txt``` contains a fozen list of all packages installed on our training environment.  We used [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage our dependencies during training.

### Edge Execution
Candidate OOD detectors were evaluated on 2 platforms: the DB21m (Jetson Nano 2GB) and the DB18 (RPi 4 + Coral TPU). Instructions to recreate the environment used on both robots are listed below.  In addition to the physical robots, you will need a host computer with duckietown shell installed as per the instructions [here](https://docs.duckietown.org/daffy/duckietown-robotics-development/draft/dt_shell.html).

#### DB21M
1. Flash the SD card with the default DuckieTown OS:
```
dts init_sd_card --hostname safeduckie --type duckiebot --configure DB21M --wifi <SSID>:<password>
```
2. Boot the system, then edit your ```/etc/apt/sources.list.d/nvidia-l4t-apt-sources.list``` file to look like this:
```
deb https://repo.download.nvidia.com/jetson/common r32.5 main
deb https://repo.download.nvidia.com/jetson/t210 r32.5 main
deb https://repo.download.nvidia.com/jetson/rt-kernel r32.5 main
```
3. Run the follwoing commmands:
```
sudo apt update
sudo apt install nvidia-l4t-rt-kernel nvidia-l4t-rt-kernel-headers
sudo shutdown -r now
```
4. When the system reboots, check that the RT kernel patch is applied:
```
duckie@safeduckie:~$ uname -a
Linux safeduckie 4.9.201-rt134-tegra #1 SMP PREEMPT RT Tue Mar 2 19:54:23 PST 2021 aarch64 aarch64 aarch64 GNU/Linux
```
5. Now we will install ROS2.  Add the ROS2 repo to apt:
```
sudo apt update && sudo apt install curl gnupg2 lsb-release
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
```
6. Install ROS2.  (Note: The DuckieTown OS for the DB21M is built on JetPack, which is built on Ubuntu 18, so the only ROS2 release we can currently use in Eloqent Elusor.)
```
sudo apt install ros-eloquent-ros-base
```
7. Install rosdep and colcon, which will be needed for building ROS2 packages:
```
source /opt/ros/eloquent/setup.bash
sudo apt install python-rosdep python3-colcon-common-extensions
sudo rosdep init
sudo rosdep fix-permissions
rosdep update
```
8. Because the Jetson is running Eloquent, you will need this work around to build fast RTPS:
```
export CMAKE_PREFIX_PATH=$AMENT_PREFIX_PATH
```
9. There is a special build of PyTorch for Jetson.  We installed torch 1.9.0 from the [official Nvidia binaries](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
10. We also found that Jetson Nano 2GB does not have enough memory to run most of our models on the GPU.  We increased its zram to 6GB by modify the followin line in ```/etc/systemd/nvzramconfig.sh```:
```
mem=$((("${totalmem}" / 2 / "${NRDEVICES}") * 1024))
```
11. We also ensured we used all 4 CPUs when evaluating the models:
```
sudo nvpmodel -m 0
```

#### DB18
1. Flash the SD card with the default DuckieTown OS
```
dts init_sd_card --hostname safeduckie --type duckiebot --configure DB18 --wifi <SSID>:<password>
```
2. Boot the robot and install dependencies required for building the RT_PRREEMPT kernel patch:
```
sudo apt install git bc bison flex libssl-dev make libncurses5-dev
```
3. Prepare to build the patch:
```
git clone --depth=1 https://github.com/raspberrypi/linux.git
wget https://mirrors.edge.kernel.org/pub/linux/kernel/projects/rt/5.10/older/patch-5.10.52-rt47.patch.xz
wget https://raw.githubusercontent.com/kdoren/linux/rpi-5.10.35-rt/0001-usb-dwc_otg-fix-system-lockup.patch
cd linux
git checkout -b rpi-5.10.52-rt 7772256a0635f0dd5b03f4736402c6c0d6371f32
xzcat ../patch-5.10.52-rt47.patch.xz | patch -p1
cat ../0001-usb-dwc_otg-fix-system-lockup.patch | patch -p1
```
4. Apply the default kernel configuration for RPi 4 (32-bit):
```
export KERNEL=kernel7l
cd ~/rpi-kernel/linux/
make bcm2711_defconfig
```
5. Using menuconfig, set options PREEMT_RT, 1000Hz timer, and performance CPU governor:
```
make menuconfig
```
6. Build the kernel and install:
```
make -j4 zImage modules dtbs
sudo make modules_install
sudo cp arch/arm/boot/dts/*.dtb /boot/
sudo cp arch/arm/boot/dts/overlays/*.dtb* /boot/overlays
sudo cp arch/arm/boot/dts/overlays/README /boot/overlays
sudo cp arch/arm/boot/zImage /boot/$KERNEL.img
```
7. On the RPi platform Pytorch can be installed via pip

## β-VAE OOD Detector Design

### Data Generation (Phase I)
In our experiments we considered a β-VAE trained on two in-distribution data partitions: rain and brightness.  In the ```data``` folder, the ```prep_vids.py``` script can be used to digitally augment images with rain or brightness.  The brightness transform is a simple value reduction in HSV color space.  Rain is simulated by randomly scattering 3 different drop sizes across the image; the more intense the rain, the greater the probability of encountering a drop at any given point.

To generate the training, calibration, and test data sets, you can use the ```prep_dataset.sh``` in the ```data``` folder.  This script requires curl as it will attempt to download the raw images from Dr. NTU.

### Hyperparameter Tuning (Phase II)

### Genetic Algorithm (Phase III)

### Quantization (Phase III)

### Deployment (Phase IV)

## Optical Flow OOD Detector Design

### Data Generation (Phase I)

### Hyperparmater Tuning (Phase II)

### Genetic Algorithm (Phase III)

### Quantization (Phase III)

### Deployment (Phase IV)
