import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

path = './day_intervals/results'
files = os.listdir(path)

res = []
for file in files:
    file_handle = open(path+"/"+file, "r", encoding="UTF-8")
    app = []
    for line in file_handle:
        days = line[1:-2].replace(' ', '').split(',')
        for day in days:
            try:
                app.append(int(day))
            except Exception as e:
                print(e)
    res.append(app)


# res_y = np.random.rand(len(res))
# plt.scatter(res,res_y)
# plt.show()
print(res)



plt.figure()
index = 1
apps = ['Instagram', 'Pandora', 'Reddit', 'SHEIN', 'Spotify', 'ZOOM']
for r in res:
    r.sort()
    gap = 30
    right = -1250+gap

    count = 0
    y = []
    for num in r:
        if num <= right:
            count += 1
        else:
            y.append(count)
            right += gap
            while num > right:
                y.append(0)
                right = right + gap
            count = 1

    y.append(count)

    x = []
    for i in range(len(y)):
        x.append(-1250+gap*i)

    from scipy.interpolate import make_interp_spline, BSpline

    xnew = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=1)  # type: BSpline
    power_smooth = spl(xnew)

    plt.subplot(2,3, index)
    plt.title(apps[index-1])
    plt.plot(xnew, power_smooth)
    index += 1

plt.show()
