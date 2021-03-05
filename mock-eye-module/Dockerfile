# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Note that this Docker container bases itself off of https://hub.docker.com/r/openvino/ubuntu18_dev
# which is itself based off of Ubuntu 18.04. The Intel Distribution of OpenVINO (and therefore this
# Docker container) requires that you agree to an
# EULA (here: https://software.intel.com/content/www/us/en/develop/articles/end-user-license-agreement.html#inpage-nav-2)
# which you implicitly agree to when you use their software (and again, therefore this container).
FROM openvino/ubuntu18_dev:2021.1
WORKDIR /
USER root

# Overwrite the default shell command form to include -xo pipefail
# See here for an explanation: https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

# Install stuff we need
RUN apt-get update && apt-get install -y \
  make \
  gdb \
  pkg-config \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  && rm -rf /var/lib/apt/lists/*

# Install application - we install it to /mock_eye_app/build/mock_eye_app
ARG appname=mock_eye_app
RUN mkdir ${appname}
COPY main.cpp ${appname}/
COPY CMakeLists.txt ${appname}/
COPY kernels ${appname}/kernels
COPY modules ${appname}/modules
RUN source /opt/intel/openvino/bin/setupvars.sh && \
    cd ${appname} && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make

# Use the OpenVINO user from the base image
USER openvino
WORKDIR /opt/intel/openvino

CMD ["/bin/bash"]