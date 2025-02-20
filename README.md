# **CRLLK: Constrained Reinforcement Learning for Lane Keeping in Autonomous Driving**  
This repository provides the **implementation** of **CRLLK**, a **Constrained Reinforcement Learning (RL) approach** for **lane-keeping in autonomous driving**, as described in our **AAMAS 2025** paper:  
📄 *CRLLK: Constrained Reinforcement Learning for Lane Keeping in Autonomous Driving*  

---
## **📌 Overview**  
CRLLK formulates **lane-keeping** as a **constrained RL problem**, where weight coefficients for different objectives (e.g., **travel distance, lane deviation, collision avoidance**) are **automatically learned** without manual tuning. The repository includes:  
- **Training pipeline** for CRLLK in a simulated environment  
- **Docker-based environment setup**  
- **Real-world deployment using ROS2 on a Duckiebot**  

---
## **🚀 Getting Started**  

### **1️⃣ Environment Setup**
We recommend using **Docker** to manage dependencies.  
#### **🔹 Prerequisites**
- **Ubuntu 18.04 / 20.04** (Recommended)  
- **NVIDIA GPU with CUDA support**  
- **Docker & NVIDIA Docker Toolkit**  
#### **🔹 Install NVIDIA Docker**
Follow the official **[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```
#### **🔹 Clone Repository & Build Docker Image**
```bash
# Clone the repository
mkdir -p ~/Project
cd ~/Project
git clone -b Publish_CRLLK --single-branch https://github.com/CPS-research-group/CPS-NTU-Public AAMAS2025 --depth 1
cd Publish_CRLLK

# Build the Docker environment
sudo docker build -t crllk_docker .
```
##### **🔹 Run Docker Container**
```bash
# Start the container with GPU access
sudo docker run --runtime=nvidia --gpus all -it --shm-size=100gb -v ~/Project:/Project crllk_docker bash
```

### **2️⃣ Training CRLLK in Simulation**
We use a **customized Duckietown Gym** environment for training.
#### **🔹 Start Training**
Select the discrete or continuous folder. Run the following command to train CRLLK in simulation:
```bash
xvfb-run -s "-screen 0 1400x900x24" python Driver.py
```
#### **🔹 Modify Training Parameters**
Adjust hyperparameters in **parameters.py**, including:
- **Learning rate**
- **Lagrange constraint thresholds for constraints**
- **Environment parameters**

### **3️⃣ Deploying CRLLK in the Real World**
The real-world deployment uses **ROS2** and runs on a **Jetson Nano-powered Duckiebot**.
**🔹 Setup ROS2 for CRLLK**
1. Install dependencies following [Jetson_ROS](https://github.com/GAO-XINWEI/Jetson_ROS).
2. Replace the **Duckietown ROS2 source folder** and 'source bash' the ROS bag:
```bash
cd ~/dt_ws
rm -rf src
git clone https://github.com/GAO-XINWEI/Duckietown_ROS2_RL
```
3. Modify the **ROS bag at**:
```bash
/src/cps/dt-core-cps/rl_control
```
 - Replace it with your **trained network** for real-world testing.
 4. Run the Lane Following RL within two different terminal. The first command will launch the `rl_control_node`, which is loading the RL network; the second command will launch the Duckiebot standard underlaying service, like motor, camera and so on:
```bash
ros2 launch rl_control rl_control_node.launch.xml
```
```bash
ros2 launch dt_demos rl_lane_following_a.launch.xml
```

## **🛠 Troubleshooting**
• **Docker GPU Issues:** Ensure nvidia-docker2 is installed and check GPU access using:
```
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0-base nvidia-smi
```
• **Simulation Crashes:** Check that **Xvfb** is installed and run:
```bash
sudo apt install xvfb
```

## **📄 Citation**
If you use **CRLLK** in your research, please cite:
```bash
@inproceedings{CRLLK_AAMAS2025,
  author    = {Xinwei Gao, Arambam James Singh, Gangadhar Royyuru, Michael Yuhas, Arvind Easwaran},
  title     = {CRLLK: Constrained Reinforcement Learning for Lane Keeping in Autonomous Driving},
  booktitle = {Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)},
  year      = {2025}
}
```