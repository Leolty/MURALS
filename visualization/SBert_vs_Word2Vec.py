# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:08:13 2022

@author: Leo
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import *

data = {
        'Word2Vec':[0.017, 0.104, 0.2667, 0.175, 0.5, 0.775],
        'SBert':[0.033, 0.108, 0.329, 0.183,0.5125, 0.7125 ],
        'Intersection':[0.05, 0.3478, 0.333, 0.617, 0.768, 0.963],
        'Union':[0.02, 0.08,0.2227, 0.12867, 0.43, 0.716]
        }

y1 = data['Word2Vec']
y2 = data['SBert']
y3 = data['Intersection']
y4 = data['Union']

x = [
     'SHEIN',
     'ZOOM',
     'Reddit',
     'Pandora',
     'Spotify',
     'Instagram'
     ]

plt.ylim(0, 1)


plt.plot(x, y1, marker='*', label='Word2Vec')
plt.plot(x, y2, marker='^', label='SBert')
plt.plot(x, y3, marker='s', label='Intersection')
plt.plot(x, y4, marker='o', label='Union')
plt.legend()  # legend
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("App") 
# plt.title("111")
plt.show()
