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
arcpy.env.overwriteOutput=True

MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'

arcpy.env.workspace=Dir_mort+'/Mortality_intersect.gdb'

for i in range(5,17):
    if i<10:       
        year="0%s" %i
    else:
        year="%s" %i
    ###########################
    inFeature = 'ADS%s_intersect' %year
    fieldName = "threshold"
    arcpy.AddField_management(inFeature, fieldName, "short")
#    expression="0"
    expression="myfun(!frac_area_mort!,!SEVERITY1!)"
    codeblock = """def myfun(input,sev):
        if sev>=-2:
            if input >= 0.05:
                n=5
            elif input >= 0.04:
                n=4
            elif input >=0.03:
                n=3
            elif input >=0.02:
                n=2
            elif input >=0.01:
                n=1
            else:
                n=0
        else:
            n=-1
                
        return n
            """
    arcpy.CalculateField_management(inFeature, fieldName, expression,"PYTHON_9.3",codeblock)
    ######################################## joining
    dir=Dir_mort+'/CA_proc.gdb'
    targetFeatures=dir+'/'+'CA_grid'
    joinFeatures=inFeature
    out_feature_class=dir+'/'+'CA_grid_%s' %year
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(targetFeatures)
    fieldmappings.addTable(joinFeatures)
    ##################
    keepers = ["threshold"] # etc.

# Remove all output fields you don't want.
    for field in fieldmappings.fields:
        if field.name not in keepers:
            fieldmappings.removeFieldMap(fieldmappings.findFieldMapIndex(field.name))
    join_type="KEEP_COMMON"    
    arcpy.SpatialJoin_analysis(targetFeatures, joinFeatures, out_feature_class, "#", join_type, fieldmappings)