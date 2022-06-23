# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:40:18 2022

@author: Leo
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import nltk.data
from collections import Counter
import time


def intersection(a, b):
    c1 = Counter(a)
    temp = []
    for i in b:
        if i in c1 and c1[i]>0:
            temp.append(i)
            c1[i] -= 1
    return temp

def union(a,b):
    temp = intersection(a,b)
    # list is mutable, a copy is needed
    copy_a = a.copy()
    for i in temp:
        copy_a.remove(i)
    
    return copy_a+b


def main():
    df = pd.read_excel('Spotify_top80s_hk.xlsx', 'Sheet1')
    
    r1 = df['review1'].tolist()
    r2 = df['review2'].tolist()
    
    r1_1 = df['review1'].loc[df['fin1']==1].tolist()
    r2_1 = df['review2'].loc[df['fin2']==1].tolist()
    
    
    
    print(len(r1),len(r2),len(intersection(r1,r2)))
    print(len(r1_1),len(r2_1),len(intersection(r1_1,r2_1)))
    print(len(union(r1,r2)))
    print(len(union(r1_1,r2_1)))
    

    
if __name__ == '__main__':
    main()
