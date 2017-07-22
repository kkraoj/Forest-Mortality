# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:50:10 2017

@author: Krishna Rao
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MyDir = 'C:/Krishna/Acads/Stanford Academics/Q1/Matlab/Codes'   #Type the path to your data


fid = open(MyDir+'/'+'./anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/'+'./anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone];
       
       


nrows=int(np.shape(late)[0])
ncols=int(np.shape(late)[1])
lat_diff=np.zeros(nrows-1)
lats=np.zeros(nrows-1)

for i in list(range(nrows-1)):
    lat_diff[i]=late[i][1]-late[i+1][1]
    lats[i]=late[i][1]
    
    
lon_diff=np.zeros(ncols-1)
lons=np.zeros(ncols-1)

for i in list(range(ncols-1)):
    lon_diff[i]=lone[1][i+1]-lone[1][i]
    lons[i]=lone[1][i]

plt.figure()
f, (ax1, ax2) = plt.subplots(1,2, sharey=True)   

ax1.plot(lats,lat_diff)
ax1.set_xlabel('Latitude (deg)')
ax1.set_ylabel('Resolution (deg)')
#plt.xlabel('Latitude (deg)')
#plt.ylabel('Resolution (deg)')
plt.suptitle('variation of VOD resolution with latitude and longitude')
ax1.axvline(32, linestyle='dashed')
ax1.axvline(42,linestyle='dashed')
ax1.axhline(0.25,linestyle='dashed')
ax1.text(15,1,'CA state range', rotation='vertical')


ax2.plot(lons,lon_diff)
ax2.set_xlabel('Longitude (deg)')
ax2.axvline(-125, linestyle='dashed')
ax2.axvline(-113.5,linestyle='dashed')
ax2.text(-155,1,'CA state range', rotation='vertical')



f.savefig('res_summary.jpeg')

plt.figure()
import gdal 

#
#img=plt.imread('myraster.TIF')
#imgplot = plt.imshow(img)





