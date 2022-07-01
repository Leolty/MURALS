# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:43:56 2022

@author: Leo
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

SBert = [577.7194137573242, 1131.2622256278992, 1655.2691028118134, 2151.665398836136, 2612.166377067566, 3049.119006872177]
Word2Vec = [6.136999130249023, 12.027751445770264, 17.841376543045044, 23.539204835891724, 29.054048776626587, 34.36428356170654]

x = [50000, 100000, 150000,200000,250000,300000]

plt.ylim(0, 3600)

plt.plot(x, SBert, marker='*', label='SBert')
plt.plot(x, Word2Vec, marker='^', label='Word2Vec')

plt.legend()  # legend
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Number of reviews") 
plt.ylabel("Time cosuming (s)")

for x1,y1 in zip(x,SBert):
    plt.text(x1,y1,'%.0f' % y1, ha='center', va='bottom', fontsize=10)

for x2,y2 in zip(x,Word2Vec):
    plt.text(x2,y2,'%.0f' % y2, ha='center', va='bottom',fontsize=10)

# plt.title("111")
plt.show()
