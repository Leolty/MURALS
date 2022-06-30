# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:03:31 2022

@author: Leo
"""

import datetime
import os

'''
This function is to calculate the interval days of two dates
Noted that the inpu format should be 'yyyy-mm(or m)-dd(or d)'
And the input should be a string
'''
def get_interval(rn:str, ur:str) -> int:
    # transform the date string to datetime
    rn_year, rn_month, rn_day = rn.split('-', 2)
    ur_year, ur_month, ur_day = ur.split('-', 2)
    
    rn_datetime = datetime.date(int(rn_year), int(rn_month), int(rn_day))
    ur_datetime = datetime.date(int(ur_year), int(ur_month), int(ur_day))
    
    return (ur_datetime-rn_datetime).days

'''
This function is to read the date of 12 release notes
The input format should be the app name and it should be a string
'''
def read_rn_date(app:str):
    file = open ("./rn_selection/"+app+".txt", "r", encoding="UTF-8")
    res = []
    for i in range(12):
        line = file.readline()
        date = line.split()[1]
        res.append(date)
    file.close()

    return res

def main():
    apps = ["Spotify", "Reddit", "Pandora", "Instagram", "SHEIN", "ZOOM"]
    for app in apps:
        path = './get_intersection/Intersection_labelled/'+app
        files = os.listdir(path)
        RNs = read_rn_date(app)
        index = 0
        for file in files:
            file_handle = open(path+"/"+file, "r", encoding="UTF-8")
            temp = []
            for line in file_handle:
                try:
                    date = line.split("&")[1][:10]
                    temp.append(get_interval(RNs[index], date))
                except Exception as e:
                    print(file + " is an empty file")
            index += 1
            file_handle.close()
            file_write = open("day_intervals/results/"+app+".txt", "a")
            file_write.write(str(temp) + '\n')
            file_write.close()




if __name__ == '__main__':
    main()