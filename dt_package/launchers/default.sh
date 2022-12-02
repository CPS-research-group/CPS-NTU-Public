#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
#roscore &
#sleep 5
rosparam set /${VEHICLE_NAME}/camera_node/framerate 30.0
rosparam set /${VEHICLE_NAME}/camera_node/res_w 160
rosparam set /${VEHICLE_NAME}/camera_node/res_h 120
dt-exec rosrun yolov7 detector.py --size 160
dt-exec rosrun ood_detector detector.py --weights vae80_5flows_enc.pt --threshold 0.75

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
