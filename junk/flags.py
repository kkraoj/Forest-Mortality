# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 03:12:25 2017

@author: Krishna Rao
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 


import numpy as np
import os
import pandas as pd
import matplotlib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.io

os.chdir('C:/Krishna/Acads/Stanford Academics/Q3/project/codes')
#import pandas as pd

MyDir = 'C:/Krishna/Acads/Stanford Academics/Q3/project/data'   #Type the path to your data
year = '2008';          #Type the year
date = '003';           #Type the date (pad with zeros to three digits: e.g. '006')
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'flag';           #Type the parameter
factor = 1e-0;          #Type the multiplier associated with the factor
bnds = [0.0,3.0];       #Type the lower and upper bounds of the parameter
maps = 'yes';           #sEnter 'yes' to seem maps and 'no' to see images

#Load the data to view
#plt.latlon = False
fid = open(MyDir+'/flags'+'/AMSRU_Mland_'+year+date+pass_type+'.'+param,'rb');
x=np.fromfile(fid,dtype='byte')
fid.close()

#for i in list(range(len(x))):
#    if x[i]<=0.0:
#        x[i]=np.nan
         
#x[x<=0.0] = np.NaN
#x = [i*factor for i in x]; 
data = x
#data = -np.log(x);
#scipy.io.savemat(MyDir+'/'+'datapy.mat', mdict={'x': x})                        
             
              
#Load latitude and longitude data (only used for mapping, not for producing
#images
fid = open(MyDir+'/'+'anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/'+'anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone];

from mkgrid_global import mkgrid_global              
datagrid = mkgrid_global(data);
##scipy.io.savemat(MyDir+'/'+'datagridpy.mat', mdict={'datagrid': datagrid})                        
#                        
#              
datagridm = ma.masked_invalid(datagrid)
latcorners=[32,42.5]
loncorners=[-125,-113.5]    

#
#latcorners=[-90,90]
#loncorners=[-180,180]                   
                        



map = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',
        llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],
        llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],)
plt.title('Flag data for Day %s, Year %s, Pass %s' % (date, year, pass_type))
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
#map.drawcountries(linewidth=0.25)
#map.fillcontinents(color='coral',lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='white')
# draw lat/lon grid lines every 30 degrees.
#    map.drawmeridians(np.arange(0,360,30))
#    map.drawparallels(np.arange(-90,90,30))
#    map.latlon = True
#    map.plot(0, 45, marker='D',color='m')
map.pcolormesh(lone,late,datagridm,cmap='RdBu_r')
#map.pcolormesh(grid_y,grid_x,grid_z,cmap='Oranges',vmin=0, vmax=3)

ax=matplotlib.pyplot.gca()
map.colorbar(cmap='RdBu_r', ax=ax)
map.readshapefile(MyDir+'/'+'cb_2016_us_state_20m'+'/'+'cb_2016_us_state_20m', 'states')

plt.savefig('flags_sample.png')