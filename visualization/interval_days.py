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
        days = line[1:-2].replace(' ','').split(',')
        for day in days:
            try:
                res.append(int(day))
            except Exception as e:
                print(e)



# res_y = np.random.rand(len(res))
# plt.scatter(res,res_y)
# plt.show()
res.sort()
print(res)

gap = 20
right = -1250+gap

count = 0
y = []
for num in res:
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

xnew = np.linspace(min(x),max(x),1000)
spl = make_interp_spline(x,y, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)
plt.show()



