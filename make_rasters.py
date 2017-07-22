# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from __future__ import division

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
from arcpy.sa import *


arcpy.env.overwriteOutput=True

MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'
year_range=range(2005,2016)
date_range=range(1,367,1)
scale_factor=1e4
pass_type = 'D';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter
factor = 1e-0;          #Type the multiplier associated with the factor
bnds = [0.0,3.0];       #Type the lower and upper bounds of the parameter      
fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone]
latcorners=[32,42.5]
loncorners=[-125,-113.5]
nrows=int(np.shape(late)[0])
ncols=int(np.shape(late)[1])
y=np.asarray(lone)
y=y.flatten()
x=np.asarray(late)
x=x.flatten()
grid_x, grid_y = np.mgrid[90:-90.25:-0.25, -180:180.25:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_y),np.min(grid_x),np.max(grid_y),np.max(grid_x)]
                
for k in year_range:
    year = '%s' %k          #Type the year 
    print('Processing data for year '+year+' ...')
    for j in date_range:        
        date='%03d'%j       
        fname=MyDir+'/'+param+'/'+year+'/AMSRU_Mland_'+year+date+pass_type+'.'+param
        if  os.path.isfile(fname):
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()            
            for i in list(range(len(data))):
                if data[i]<=0.0:
                    data[i]=np.nan                     
            data = [i*factor for i in data]; 
            data = -np.log(data)            
            from mkgrid_global import mkgrid_global              
            datagrid = mkgrid_global(data)                                           
            datagridm = ma.masked_invalid(datagrid)
            from scipy.interpolate import griddata
            z=np.asarray(datagridm)
            z=z.flatten()                                
            grid_z = griddata((y,x), z, (grid_y, grid_x), method='linear')
            grid_z = ma.masked_invalid(grid_z)
            nrows,ncols = np.shape(grid_z)
            nrows=np.int(nrows)
            ncols=np.int(ncols)            
            xres = (xmax-xmin)/(ncols-1)
            yres = (ymax-ymin)/(nrows-1)
            geotransform=(xmin,xres,0,ymax,0, -yres)   
            arcpy.env.workspace=Dir_fig
            os.chdir(Dir_fig)
            output_raster = gdal.GetDriverByName('GTiff').Create('VOD_%s_%s_%s.tif' %(year,date,pass_type),ncols, nrows, 1 ,gdal.GDT_Float32,)  # Open the file
            output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
            srs = osr.SpatialReference()                 # Establish its coordinate encoding
            srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                         # Anyone know how to specify the 
                                                         # IAU2000:49900 Mars encoding?
            output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                               # to the file
            output_raster.GetRasterBand(1).WriteArray(grid_z)   # Writes my array to the raster
            output_raster.FlushCache()
            output_raster = None
            arcpy.Clip_management('VOD_%s_%s_%s.tif' %(year,date,pass_type), "#",'VOD_%s_%s_%s_clip.tif' %(year,date,pass_type),Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
            ##mapping algebra * 10000
            inRaster = 'VOD_%s_%s_%s_clip.tif'%(year,date,pass_type)
            arcpy.CheckOutExtension("Spatial")
            outRaster = Raster(inRaster)*scale_factor
            outRaster.save(Dir_fig+'/'+'VOD_%s_%s_%s_clip_map.tif'%(year,date,pass_type))
            ##copy raster
            inRaster=outRaster
            pixel_type='16_BIT_UNSIGNED'
            arcpy.CopyRaster_management(inRaster, 'VOD_%s_%s_%s_clip_map_copy.tif'%(year,date,pass_type), pixel_type='16_BIT_UNSIGNED',nodata_value='0')
            ##make raster table
            inRaster='VOD_%s_%s_%s_clip_map_copy.tif'%(year,date,pass_type)
            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")

            
            
            
            
            
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
