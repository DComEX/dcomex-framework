FROM docker.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
ARG GIT_SSL_NO_VERIFY=1
ARG TZ=Europe/Zurich
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update
RUN apt-get -qq --fix-missing upgrade
RUN apt-get -qq install --no-install-recommends apt-transport-https
RUN apt-get -qq install --no-install-recommends cmake
RUN apt-get -qq install --no-install-recommends g++
RUN apt-get -qq install --no-install-recommends git
RUN apt-get -qq install --no-install-recommends libeigen3-dev
RUN apt-get -qq install --no-install-recommends libgsl-dev
RUN apt-get -qq install --no-install-recommends libmpich-dev
RUN apt-get -qq install --no-install-recommends pkg-config
RUN apt-get -qq install --no-install-recommends python3-dev
RUN apt-get -qq install --no-install-recommends python3-matplotlib
RUN apt-get -qq install --no-install-recommends python3-mpi4py
RUN apt-get -qq install --no-install-recommends python3-numpy
RUN apt-get -qq install --no-install-recommends python3-pybind11
RUN apt-get -qq install --no-install-recommends python3-scipy
RUN apt-get -qq install --no-install-recommends wget
WORKDIR /
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
RUN dpkg --install /tmp/packages-microsoft-prod.deb
RUN apt-get -qq update
RUN apt-get -qq install --no-install-recommends dotnet-sdk-6.0
RUN git clone --quiet --single-branch --depth 1 https://github.com/slitvinov/dcomex-prototype src
WORKDIR /src/examples/korali/
RUN git clone --quiet --single-branch https://github.com/cselab/korali
RUN cd korali && git checkout c70d8e32258b7e2b9ed977576997dfe946816419
RUN make install -j4
