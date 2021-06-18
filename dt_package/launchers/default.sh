#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# Launch ood_node and lane_following first followed by sleep so they are
# fully initialized by the time we start the motors.
dt-exec rosrun safe_ml ood_node.py
sleep 20
dt-exec rosrun safe_ml lane_following_node.py
sleep 10
dt-exec rosrun safe_ml motor_control_node.py

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
