import sys
sys.path.append("./")

from day_intervals.intervals_cal import read_rn_date, get_interval
import os



apps = ["Spotify", "Reddit", "Pandora", "Instagram", "SHEIN", "ZOOM"]

before,after = [], []

for app in apps:
    path = './get_intersection/Intersection_labelled/'+app
    files = os.listdir(path)
    RNs = read_rn_date(app)
    index = 0
    for file in files:
        file_handle = open(path+"/"+file, "r", encoding="UTF-8")

        for line in file_handle:
            try:
                date = line.split("&")[1][:10]
                rating = line.split("&")[0]
                interval = get_interval(RNs[index], date)
                if interval <= 0:
                    before.append(rating)
                if interval > 0:
                    after.append(rating)
            except Exception as e:
                print(file + " is an empty file")
        index += 1
        file_handle.close()

before = [int(num) for num in before]
after = [int(num) for num in after]


from collections import Counter

before_counter = dict(Counter(before))
after_counter = dict(Counter(after))
all_counter = dict(Counter(after + before))

import matplotlib.pyplot as plt
import numpy as np

all_counter =sorted(all_counter.items(),key=lambda x:x[0],reverse=False)
x_data = []
y_data = []
for key, value in all_counter:
    x_data.append(key)
    y_data.append(value)

print(x_data)
print(y_data)

for i in range(len(x_data)):
	plt.bar(x_data[i], y_data[i], width=0.8)

plt.xlabel('Ratings')
plt.ylabel('Number of user reviews')
plt.show()
