#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Plot DEA results (read from .json files)
def plot(genList, **kwargs):
  firstKey = next(iter(genList))

  numdim = len(genList[firstKey]['Variables'])
  numgens = len(genList)
  
  fig, ax = plt.subplots(num='Korali Results', figsize=(8, 8))

  last = None
  lastGen = -1
  for i in genList:
    if genList[i]['Current Generation'] > lastGen:
      last = genList[i]

  plt.suptitle('SSM Diagnostics', fontweight='bold', fontsize=12)

  time = last["Results"]["Time"]
  meanTrajectories = last["Results"]["Mean Trajectory"]

  for i in range(numdim):
    varName = last['Variables'][i]['Name']
    ax.plot(time, meanTrajectories[i], label=varName)

  ax.set_xlabel(r"$t$")
  ax.set_ylabel(r"$x$")
  ax.set_xlim(0,time[-1])
  ax.set_ylim(0,)
  ax.legend()
