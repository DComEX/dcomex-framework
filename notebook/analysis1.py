import glob
import matplotlib.pyplot as plt
import numpy as np
import utils
import sklearn.pipeline
import sklearn.linear_model
import sklearn.compose


def g_list(i):
    slist = sorted(set(param[i] for param, *rest in D))
    colors = {
        x: y
        for x, y in zip(slist, plt.cm.jet(np.linspace(0, 1, len(slist))))
    }
    return slist, colors


def save(path):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()


D = [utils.read(path) for path in glob.glob("1/*/MSolveInput.xml")]
k1_list, k1_color = g_list(0)
mu_list, mu_color = g_list(1)
sv_list, sv_color = g_list(2)

tmax = max(t for param, time, volume in D for t in time)
vmin = min(v for param, time, volume in D for v in volume)

vtransform = sklearn.preprocessing.FunctionTransformer(
    lambda x: np.log(np.divide(x, vmin))**0.75)
for (k1, mu, sv, *rest), time, volume in D:
    plt.plot(time, volume, color=mu_color[mu], alpha=0.1)
save("analysis1.0.png")
X = []
y = []
for (k1, mu, sv, *rest), time, volume in D:
    X.extend(time)
    y.extend(volume)
    plt.plot(time, vtransform.transform(volume), color=mu_color[mu], alpha=0.1)
save("analysis1.1.png")

reshape = sklearn.preprocessing.FunctionTransformer(
    lambda x: np.reshape(x, (-1, 1)))
pipe = sklearn.pipeline.make_pipeline(
    reshape,
    sklearn.preprocessing.FunctionTransformer(lambda x: np.divide(x, tmax)),
    sklearn.linear_model.LinearRegression())
model = sklearn.compose.TransformedTargetRegressor(
    regressor=pipe,
    transformer=sklearn.pipeline.make_pipeline(reshape, vtransform),
    check_inverse=False)
model.fit(X, y)

ltransform = sklearn.pipeline.make_pipeline(
    vtransform,
    sklearn.preprocessing.FunctionTransformer(
        lambda x: x - model.predict(time)))
ltransform
for (k1, mu, sv, *rest), time, volume in D:
    plt.plot(time, ltransform.transform(volume), color=mu_color[mu])
save("analysis1.2.png")
exit(0)

axis = None
for i, k10 in enumerate(k1_list):
    for (k1, mu, sv, *rest), time, volume in D:
        if mu == mu_list[0] and k1 == k1_list[-1] and sv == sv_list[0]:
            volume0 = ltransform.transform(volume)
            plt.plot(time, volume0, "k-")
    for (k1, mu, sv, *rest), time, volume in D:
        if mu == mu_list[0] and k1 == k10:
            volume = vtransform.transform(volume)
            volume -= model.predict(time)
            l2 = np.linalg.norm(np.subtract(volume0, volume))
            print(k1, sv, l2)
            plt.plot(time, volume, color=sv_color[sv])
            if axis is None:
                x0, x1, y0, y1 = plt.axis()
                l = y1 - y0
                y0 -= 0.05 * l
                y1 += 0.05 * l
                axis = x0, x1, y0, y1
            plt.axis((x0, x1, y0, y1))
    save("analysis1.3.%02d.png" % i)
n = len(D)
l2min = float("inf")
for i in range(n - 1):
    param0, time, volume0 = D[i]
    k10, mu0, sv0 = param0
    volume0 = vtransform.transform(volume0)
    volume0 -= model.predict(time)
    for j in range(i + 1, n):
        param1, time, volume1 = D[j]
        k11, mu1, sv1 = param1
        if mu0 == mu_list[0] and mu1 == mu_list[
                0] and sv1 != sv0 and k11 == k1_list[0]:
            volume1 = vtransform.transform(volume1)
            volume1 -= model.predict(time)
            l2 = np.linalg.norm(np.subtract(volume0, volume1))
            if l2 < l2min:
                l2min = l2
                v0, v1 = volume0, volume1
                p0, p1 = param0, param1
plt.plot(time, v0, "-g")
plt.plot(time, v1, "-r")
save("analysis1.4.png")
