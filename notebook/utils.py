import xml.etree.ElementTree as ET
import os
import math


def fix_time(time):
    t = 0
    n = len(time)
    ans = [0]
    for i in range(n - 1):
        dt = time[i + 1] - time[i]
        if i > 0 and not math.isclose(dt, prev):
            t += prev
        else:
            t += dt
        ans.append(t)
        prev = dt
    return ans


def read(path):
    root = ET.parse(path)
    params = [
        float(root.find('./Parameters/' + key).text)
        for key in ("k1", "mu", "svTumor")
    ]
    root = ET.parse(os.path.join(os.path.dirname(path), "MSolveOutput-x.xml"))
    time, volume = zip(
        *[[float(t.get("time")), float(t.text)]
          for t in root.findall("./TumorVolumes/TumorVolume")])
    return params, fix_time(time), volume
