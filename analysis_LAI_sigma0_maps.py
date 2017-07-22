# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
sns.set(font_scale=1.2)
sns.set_style("white")
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2009,2017)    
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
mort.index=pd.to_datetime(mort.index,format='%Y')
mort=mort[(mort.index.year>=np.min(year_range))&(mort.index.year<=np.max(year_range))]
store = pd.HDFStore(Dir_CA+'/LAI.h5')
data=store['RWC_min_anomaly_smallgrid']
store.close()

grids=Dir_mort+'/CA_proc.gdb/smallgrid'
grid_size=0.05
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
### map plots
latcorners=[33,42.5]
loncorners=[-124.5,-117]    
#latcorners=[37,38]
#loncorners=[-120,-119] 

cols=mort.shape[0]
zoom=2
rows=2
fig_width=zoom*cols
fig_height=1.5*zoom*rows
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height))
plt.subplots_adjust(wspace=0.04,hspace=0.001)
marker_factor=1
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
for year in year_range:   
    mort_plot=mort[mort.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap='Reds',\
                        marker='s',vmin=0,vmax=1)
    
    data_plot=data[data.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap='RdPu_r',marker='s',\
                       vmin=-2.0,vmax=1)
cb1=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.0074,\
                 aspect=20,pad=0.02)
cb2=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.0074,\
                 aspect=20,pad=0.02)
cb2.ax.invert_yaxis()
fig.text(0.1,0.67,'Fractional area \n Mortality',horizontalalignment='center',\
         verticalalignment='center',rotation=90)
fig.text(0.1,0.33,r'Min. $\frac{VOD_{ASCAT}}{LAI}$'+'\n anomaly',\
         horizontalalignment='center',verticalalignment='center',rotation=90)
plt.show()


#KDE
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
                        sharey='row')
plt.subplots_adjust(wspace=0.2,hspace=0.2)
for year in year_range:   
    mort_plot=mort[mort.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    plot_mort=mort_plot.T.plot(kind='kde',ax=ax,legend=False,color='r')  
    ax.set_ylabel('Kernel Density \n of FAM')
    ax.set_xlim([-0.1,1.1])
    data_plot=data[data.index.year==year]
#    data_plot=data_plot[data_plot>=-100]
    ax=axs[1,year-year_range[0]]
    plot_data=data_plot.T.plot(kind='kde',ax=ax,legend=False,color='m')
    ax.set_ylabel('Kernel Density \n of  '+r'$\frac{VOD_{ASCAT}}{LAI}$')
    ax.set_xlim([-3,2])
    ax.invert_xaxis()
plt.show()

## all in one kde plots
sns.set_style("darkgrid")
mort=mort[mort.index.year>=2009]
data=data[data.index.year>=2009]
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
plt.subplots_adjust(wspace=0.45,hspace=0.2)
mort.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[0])
axs[0].set_ylabel('Kernel Density \n of FAM')
axs[0].legend(mort.index.year)
data.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[1])
axs[1].set_ylabel('Kernel Density \n of  '+r'$\frac{VOD_{ASCAT}}{LAI}$')
axs[1].legend(data.index.year)
axs[1].invert_xaxis()
