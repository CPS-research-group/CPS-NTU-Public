# parameters
ARG REPO_NAME="duckietown-safe-ml-demo"
ARG DESCRIPTION="Safe ML on the Duckie Town Platform"
ARG MAINTAINER="michaelj004@e.ntu.edu.sg"
# pick an icon from: https://fontawesome.com/v4.7.0/icons/
ARG ICON="cube"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH=arm32v7
ARG DISTRO=daffy
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG BASE_IMAGE=dt-ros-commons
ARG LAUNCHER=default

# define base image
FROM duckietown/${BASE_IMAGE}:${BASE_TAG} as BASE

# Changed to have the full suite of DT-packages.
#ARG AIDO_REGISTRY=docker.io
#FROM ${AIDO_REGISTRY}/duckietown/dt-car-interface:${BASE_TAG} AS dt-car-interface
#FROM ${AIDO_REGISTRY}/duckietown/dt-core:${BASE_TAG} AS base

# recall all arguments
ARG ARCH
ARG DISTRO
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG ICON
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER

# check build arguments
RUN dt-build-env-check "${REPO_NAME}" "${MAINTAINER}" "${DESCRIPTION}"

# define/create repository path
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
ARG LAUNCH_PATH="${LAUNCH_DIR}/${REPO_NAME}"
RUN mkdir -p "${REPO_PATH}"
RUN mkdir -p "${LAUNCH_PATH}"
#WORKDIR "${REPO_PATH}"
WORKDIR /code
#COPY --from=dt-car-interface ${CATKIN_WS_DIR}/src/dt-car-interface ${CATKIN_WS_DIR}/src/dt-car-interface

# keep some arguments as environment variables
ENV DT_MODULE_TYPE "${REPO_NAME}"
ENV DT_MODULE_DESCRIPTION "${DESCRIPTION}"
ENV DT_MODULE_ICON "${ICON}"
ENV DT_MAINTAINER "${MAINTAINER}"
ENV DT_REPO_PATH "${REPO_PATH}"
ENV DT_LAUNCH_PATH "${LAUNCH_PATH}"
ENV DT_LAUNCHER "${LAUNCHER}"

# Daniel - Add some dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python3-tk \
         python3-wheel \
         python3-pip  \
         python3-pillow  \
         software-properties-common \
         #Pytorch dependencies
         libopenblas-dev \
         libblas-dev \
         libcairo2-dev \
         pkg-config \
         m4 \
         cmake \
         cython \
         python3-dev \
         python3-yaml \
         curl \
         python3-setuptools && \
     rm -rf /var/lib/apt/lists/*

RUN echo PYTHONPATH=$PYTHONPATH
RUN pip3 install -U "pip>=20.2" pipdeptree setuptools wheel
RUN pip3 install pycairo
COPY ./libraries "${REPO_PATH}/libraries"
RUN pip3 install "${REPO_PATH}/libraries/numpy-1.20.0-cp38-cp38-linux_armv7l.whl"
RUN pip3 install "${REPO_PATH}/libraries/torch-1.7.1-cp38-cp38-linux_armv7l.whl"
RUN pip3 install "${REPO_PATH}/libraries/torchvision-0.8.0a0+45f960c-cp38-cp38-linux_armv7l.whl"

RUN pipdeptree
RUN pip list

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
    sudo apt-get update && \
    sudo apt-get install libedgetpu1-std


# install apt dependencies
COPY ./dependencies-apt.txt "${REPO_PATH}/"
RUN dt-apt-install ${REPO_PATH}/dependencies-apt.txt

# install python3 dependencies
COPY ./dependencies-py3.txt "${REPO_PATH}/"
RUN pip3 install --use-feature=2020-resolver -r ${REPO_PATH}/dependencies-py3.txt

# copy the source code
COPY ./packages "${REPO_PATH}/packages"

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# install launcher scripts
COPY ./launchers/. "${LAUNCH_PATH}/"
COPY ./launchers/default.sh "${LAUNCH_PATH}/"
RUN dt-install-launchers "${LAUNCH_PATH}"

# define default command
CMD ["bash", "-c", "dt-launcher-${DT_LAUNCHER}"]

# store module metadata
LABEL org.duckietown.label.module.type="${REPO_NAME}" \
    org.duckietown.label.module.description="${DESCRIPTION}" \
    org.duckietown.label.module.icon="${ICON}" \
    org.duckietown.label.architecture="${ARCH}" \
    org.duckietown.label.code.location="${REPO_PATH}" \
    org.duckietown.label.code.version.distro="${DISTRO}" \
    org.duckietown.label.base.image="${BASE_IMAGE}" \
    org.duckietown.label.base.tag="${BASE_TAG}" \
    org.duckietown.label.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================
