# Use PyTorch as the base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# install dependencies
RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    wget \
    curl \
    vim \
    nano\
    git \
    xvfb\
    && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Update pip
RUN pip install --upgrade pip

# Install your dependencies
RUN pip install wandb==0.13.5 ray==2.1.0

# If you have additional dependencies add them in the line above, in the format:
# package-name==version
# replacing 'package-name' and 'version' with your package names and versions.

# Clone the git repo
RUN git clone https://github.com/duckietown/gym-duckietown.git
WORKDIR /app/gym-duckietown
# Fix and install version 6.1.31
RUN git checkout 97fd95a
RUN pip cache purge
RUN pip install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -e .
# Change back to the app directory
WORKDIR /app

# Change the simulator, see detial in github
RUN git clone https://github.com/GAO-XINWEI/gym-duckietown-simulator.git
# Change on simulator
RUN cp -f /app/gym-duckietown-simulator/simulator.py /app/gym-duckietown/src/gym_duckietown/
# Add maps
RUN cp /app/gym-duckietown-simulator/zigzag_without_obj.yaml /opt/conda/lib/python3.9/site-packages/duckietown_world/data/gd1/maps
RUN cp /app/gym-duckietown-simulator/loop_dyn_duckiebots_new.yaml /opt/conda/lib/python3.9/site-packages/duckietown_world/data/gd1/maps
RUN rm -r gym-duckietown-simulator
# Fix version of pgylet and numpy version
RUN pip install pyglet==1.5.11 numpy==1.23.4

# Improve the simulator performance, according to gym-duckietown
RUN export PYGLET_DEBUG_GL=True

# Expose the port your app runs on
EXPOSE 5000

# Run your command
# CMD ["python", "your_script.py"]

# Remove this before publish
WORKDIR /Project
