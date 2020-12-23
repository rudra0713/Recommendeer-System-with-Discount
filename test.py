# import numpy as np
# import matplotlib.pyplot as plt
#
# N = 3
# ind = np.arange(N)  # the x locations for the groups
# width = 0.27       # the width of the bars
#
# fig = plt.figure()
# # plt = fig.add_subplot(111)
#
# yvals = [4, 9, 2]
# rects1 = plt.bar(ind, yvals, width, color='r')
# zvals = [1,2,3]
# rects2 = plt.bar(ind+width, zvals, width, color='g')
# kvals = [11,12,13]
# rects3 = plt.bar(ind+width*2, kvals, width, color='b')
#
# # plt.set_ylabel('Scores')
# # # plt.set_xticks(ind+width)
# # plt.set_xticklabels( ('2011-Jan-4', '2011-Jan-5', '2011-Jan-6') )
# plt.legend( (rects1[0], rects2[0], rects3[0]), ('y', 'z', 'k') )
#
# def autolabel(rects):
#     for rect in rects:
#         h = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
#                 ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
#
# plt.show()

import re

