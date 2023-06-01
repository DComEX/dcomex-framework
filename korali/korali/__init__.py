import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.append(extdir)


def Engine():
  from libkorali import Engine
  return Engine()


def Experiment():
  from libkorali import Experiment
  return Experiment()


def getWorkerMPIComm():
  from libkorali import getWorkerMPI4PyComm
  return getWorkerMPI4PyComm()
