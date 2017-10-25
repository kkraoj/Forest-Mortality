# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:25:52 2017

@author: kkrao
"""

df=store['vod_005_grid']
df=RWC(df,0.95)
df.index.name='RWC_005_grid'
store[df.index.name]=df
timeseries_maps(grid_size=5, var2='RWC_005_grid',end_year=2016,\
                var2_range=[2e-1,1], var1_range=[1e-5, 1], var2_label='Relatve\nwater content',cmap2='inferno',\
                title='Timeseries of observed mortality and RWC')

df=store['vod_pm']
df=RWC(df,0.95)
df.index.name='RWC'
store[df.index.name]=df
timeseries_maps(grid_size=25, var2='RWC',end_year=2015,\
                var2_range=[1e-5,1], var1_range=[1e-5, 0.4], var2_label='Relatve\nwater content',cmap2='inferno',\
                title='Timeseries of observed mortality and RWC')
timeseries_maps()
