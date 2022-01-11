# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 05:46:36 2020

@author: 이재헌
"""
import axelrod as axl
import matplotlib
import matplotlib.pyplot as plt
import pickle

with open('training_info_{}.pkl'.format(4399), 'rb') as fin:
    dic = pickle.load(fin)

report_fifty = dic['report_fifty']
report_whole = dic['report_whole']
report_C = dic['report_C']
report_D = dic['report_D']

print(len(report_C))
print(len(report_whole))

xs = []
for i in range(4350):
    xs.append(i+1)

plt.plot(xs, report_whole, 'r-', label='report_whole')  # x, y, 선 색깔 및 타입 (r: 빨강, -: 실선)
plt.ylim(-10,60)
plt.xlabel("trial")
plt.ylabel("Recent 50 games, count the number of C")
plt.title("please")

#plt.plot(xs, report_C, 'r', label='C')
#plt.plot(xs, report_D, 'b', label='D')
#plt.ylim(50,300)
#plt.xlim(900,1400)
plt.show()