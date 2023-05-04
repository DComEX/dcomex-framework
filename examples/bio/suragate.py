import sys
import statistics
import numpy as np

A = [ ]
y = [ ]
for line in sys.stdin:
    k1, mu, V = map(float, line.split())
    if mu < 0.4:
        A.append([k1, mu, k1 * mu, 1])
        y.append(V)

x, residuals, *rest= np.linalg.lstsq(A, y, rcond=None)
y0 = np.dot(A, x)
for (k1, mu, *rest), y, y0 in zip(A, y, y0):
    print(k1, mu, y0)
a, b, ab, c = x
sys.stderr.write("a = %.16e\n" %  a)
sys.stderr.write("b = %.16e\n" %  b)
sys.stderr.write("ab = %.16e\n" %  ab)
sys.stderr.write("c = %.16e\n" %  c)
