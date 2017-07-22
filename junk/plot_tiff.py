# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 


import numpy as np
import pandas as pd
import matplotlib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.io
import os
import arcpy
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr



Dir1='D:/Krishna/Tree_Mort Project/data/Mort Data/CA'
Dir2='D:/Krishna/Tree_Mort Project/figures'
#Dir3='D:/Krishna/Tree_Mort Project/data/Mort Data/CA Mortality Data/CA_proc.gdb'


arcpy.Clip_management(Dir2+'/'+"myraster.tif", "#", Dir2+'/'+"clip_new.tif",Dir1+'/'+"CA.shp", "0", "ClippingGeometry")

# Plotting 2070 projected August (8) precip from worldclim

gdata = gdal.Open(Dir2+'/'+'clip.tif')
geo = gdata.GetGeoTransform()
data = gdata.ReadAsArray()

xres = geo[1]
yres = geo[5]

latcorners=[32,42.5]
loncorners=[-125,-113.5]    

# A good LCC projection for USA plots
#m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',
        llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],
        llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],)

# This just plots the shapefile -- it has already been clipped
m.readshapefile(Dir1+'/'+'CA','CA',drawbounds=True, color='0.3')
m.readshapefile(Dir1+'/'+'grid','grid',drawbounds=True, color='black')

xmin = geo[0] + xres * 0.5
xmax = geo[0] + (xres * gdata.RasterXSize) - xres * 0.5
ymin = geo[3] + (yres * gdata.RasterYSize) + yres * 0.5
ymax = geo[3] - yres * 0.5

x,y = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
x,y = m(x,y)

cmap = 'Oranges'
#cmap.set_under ('1.0')
#cmap.set_bad('0.8')

im = m.pcolormesh(x,y, data.T, cmap=cmap, vmin=0, vmax=3)

cb = plt.colorbar( orientation='vertical', fraction=0.10, shrink=0.7)
plt.title('VOD clipped')
plt.show()


