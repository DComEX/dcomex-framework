ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src
WORKDIR /src
RUN MAKEFLAGS=-j4 make lkorali lbin lib lmsolve
RUN chmod +x msolve/runTest.sh
