ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src
WORKDIR /src
RUN make lbin
RUN make ldotnet
RUN make lib
RUN make lmsolve
RUN pip install pip install git+https://github.com/slitvinov/GliomaSolver
RUN MAKEFLAGS=-j4 make lkorali
