# SafeML
This repository contains the implementation used in the DESTION2021 paper: *Embedded Out-of-Distribution Detection on an Autonomous Robot Platform*.  The data gathered during the experiments presented in the paper is available [here](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/FVVHNK).

## Quick Start
This repo contains a Duckietown package used to perform experiments on the Safe ML project.  Before you can run the project, you will have to install the required dependencies.

### Dependency Installation
Please note: currently only DB18 hardware is supported.

1. [Intall Docker](https://www.docker.com).

2. Install Duckietown on your host.  If you are using Ubuntu, you can follow the instructions [here](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/laptop_setup.html).  On MacOS you can follow the steps below to launch the Python virtual environment included in this repo.  The Pipfile contains all the dependencies you need for this project:

```
$ git clone <this repo>
$ cd CPS-NTU-Public/dt_package
$ pip install pipenv
$ pipenv install
$ pipenv shell
```

3. Setup and install Hypriot OS on the DB18 with the Duckietown framework as described [here](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/setup_duckiebot.html).

### Running the Duckiebot

1. Power up the Duckiebot.

2. Build the SafeML package on the Duckiebot:

```
$ dts devel build -f -H <Name of Duckiebot>
```

3. Tape out some lane lines on the ground.

4. Run the package:

```
$ dts devel run -H <Name of Duckiebot>
```

Currently there are two aspects to the package.  One is lane following and the other is OOD detection.  If an object is placed in front of the Duckiebot while it is following a lane, it should be detected as OOD and come to a stop.

## Tweaking the Program
There are three configuration files you can tweak to adjust performance and run experiments:

* **ood_params.yml** - this file sets the model for OOD detection.  If you train your own model, save the weights in a *.pt file and include a *.pt_config file with topz and threshold.

* **lane_following_params.yml** - this file contains parameters used to tune the lane following node.  If you are having trouble following lanes in a certain environment, try tweaking these parameters and check for improvement.

* **motor_control_params.yml** - this file contains the PID control parameters for the motor control node.  Tune these if you find the steering too impulsive or too sluggish.

## Collecting Data
This package uses ROS's built-in logger display runtime information.  Currently the following information is logged:

* Node initialization times

* Errors that occur during lane detection and the time they occurred

* When the OOD detector finishes processing an image, and the timestamp when that image was captured

* When the emergency stop is activated and the wheel velocities have been set to 0cm/s

## Plotting Results
The code used to process our raw experimental data and generate the plots featured in the paper is located in the plots/ folder.  *raw_console_logs.json* is the console output for each of the 40 iterations of our experiment.  *parse_logs.py* is used to extract the timing data from the raw text logs and store it in CSV format to make analysis and visualization easier.  *measured_data.csv* contains data we measured by hand, externally from from the running Duckiebot.  *plots.py* takes the CSV data and generates the figures featured in the paper.

## Helpful Hints
We found Rosbag to be a useful tool for debugging.  Rosbag allows you to dump all the messages for a group of Rostopics to a bagfile for later analysis.  We have included *rosbag_parser.py* in the Duckietown package to help us gather training and measurement data from Rosbags.

You can record a Rosbag of images from the Duckiebot camera with the following command:

```
$ rosbag record -O <bagfile name> /<Duckiebot name>/camera_node/image/compressed
```
