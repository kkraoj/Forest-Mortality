# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
sns.set_style("white")

### map plots
latcorners=[33,42.5]
loncorners=[-124.5,-117]    
#latcorners=[37,38]
#loncorners=[-120,-119] 


## VOD PM ascending pass
store = pd.HDFStore(Dir_CA+'/vodDf.h5')#ascending is 1:30 PM
VOD_PM=store['vodDf']
VOD_PM.index.name='gridID'
VOD_PM=VOD_PM.T
VOD_PM.drop('gridID',inplace=True)
VOD_PM.index=[x[:-1] for x in VOD_PM.index] 
VOD_PM.index=pd.to_datetime(VOD_PM.index,format='%Y%j')
VOD_PM=VOD_PM[VOD_PM.index.dayofyear!=366]
VOD_PM_anomaly=median_anomaly(VOD_PM)
store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort.index.name='gridID'
mort=mort.T
mort.drop('gridID',inplace=True)
mort.index=[x[-4:] for x in mort.index] 
mort.index=pd.to_datetime(mort.index)
mort_05_15=mort[mort.index.year!=2016]
store.close()
year_range=mort_05_15.index.year
cols=mort_05_15.shape[0]
zoom=2
rows=2
vod_min=np.nanmin(VOD_PM_anomaly.iloc[:, :].values)
vod_max=np.nanmax(VOD_PM_anomaly.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
grids=Dir_mort+'/CA_proc.gdb/grid'
grid_size=0.25
marker_factor=2
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height))
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
plt.subplots_adjust(wspace=0.04,hspace=0.001)
for year in year_range:   
    mort_data_plot=mort_05_15[mort_05_15.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=1.5*marker_size,c=mort_data_plot,cmap='Reds'\
                        ,marker='s',vmin=0,vmax=0.4)
    
    VOD_data_plot=VOD_PM_anomaly[VOD_PM_anomaly.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_vod=m.scatter(lats, lons,s=1.5*marker_size,c=VOD_data_plot,cmap='RdPu_r'\
                       ,marker='s',vmin=-2.6,vmax=-0)
cb1=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.005,aspect=20,pad=0.02)
cb2=fig.colorbar(plot_vod,ax=axs[1,:].ravel().tolist(), fraction=0.005,aspect=20,pad=0.02)
cb2.ax.invert_yaxis()
fig.text(0.1,0.67,'Fractional area \n Mortality',horizontalalignment='center',verticalalignment='center',rotation=90)
fig.text(0.1,0.33,'Min. VOD (pm) \n anomaly',horizontalalignment='center',verticalalignment='center',rotation=90)
plt.show()

## all in one kde plots
sns.set_style("darkgrid")
mort=mort_05_15[mort_05_15.index.year>=2009]
data=VOD_PM_anomaly[VOD_PM_anomaly.index.year>=2009]
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
plt.subplots_adjust(wspace=0.35,hspace=0.2)
mort.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[0])
axs[0].set_ylabel('Kernel Density \n of FAM')
axs[0].legend(mort.index.year)
data.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[1])
axs[1].set_ylabel('Kernel Density \n of $VOD_{pm}$')
axs[1].legend(data.index.year)
axs[1].invert_xaxis()



