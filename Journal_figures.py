# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:41:22 2017

@author: kkrao
"""
###response is number of dead trees
import os
import arcpy
import matplotlib as mpl
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from dirs import Dir_CA, Dir_mort, get_marker_size, RWC,clean_xy, piecewise_linear
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle  

os.chdir(Dir_CA)
sns.set(font_scale=1.2)
mpl.rcParams['font.size'] = 15
store=pd.HDFStore('data.h5')

#inputs
data=(store['vod_pm']) 
grid_size=25
start_year=2009
start_month=7
months_window=3
data_label="Predicted fractional\narea of mortality"
cmap='inferno_r'
alpha=0.7
mort_label="Observed fractional\narea of mortality"

#----------------------------------------------------------------------
mort=store['mortality_%03d_grid'%(grid_size)]
mort=mort[mort>0]
data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
data_anomaly=RWC(data)
end_year=2015
data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
                          (data_anomaly.index.year<=end_year)]
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
pred_mort=store['predicted_FAM']
year_range=mort.index.year
cols=mort.shape[0]
zoom=1.1
rows=2
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
tree_min=np.nanmin(mort.iloc[:, :].values)
tree_max=np.nanmax(mort.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
if grid_size==25:
    grids=Dir_mort+'/CA_proc.gdb/grid'
    marker_factor=7
    scatter_size=20
elif grid_size==5:
    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
    marker_factor=2
    scatter_size=4
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
sns.set_style("white")
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
                        sharey='row')
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
plt.subplots_adjust(wspace=0.04,hspace=0.15,top=0.83)
for year in year_range:   
    mort_plot=mort[mort.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year),size=10)
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap=cmap,\
                        marker='s',\
                        vmin=0.9*tree_min,vmax=0.9*tree_max,\
                        norm=mpl.colors.LogNorm()\
                                               )
    #---------------------------------------------------------------
    data_plot=pred_mort[pred_mort.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap\
                       ,marker='s',vmin=0.9*tree_min,vmax=0.9*tree_max,\
                       norm=mpl.colors.LogNorm()\
                                              )
    #-------------------------------------------------------------------
cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb0.ax.tick_params(labelsize=10) 
cb1=fig.colorbar(plot_mort,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb1.ax.tick_params(labelsize=10) 
axs[0,-1].annotate('Fractional area\nof mortality', xy=(1.1, 1.05), xycoords='axes fraction',\
            ha='left',size=10)
axs[0,0].set_ylabel(mort_label,size=10)
axs[1,0].set_ylabel(data_label,size=10)
fig.suptitle('Timeseries of observed and predicted mortality')
plt.show()
#------------------------------------------------------------------------
def import_mort_leaf_habit(species,grid_size=25,start_year=2009,end_year=2015):
    import os
    import pandas as pd
    from dirs import Dir_CA
    os.chdir(Dir_CA)
    store=pd.HDFStore('data.h5')
    mort=store['mortality_%s_%03d_grid'%(species,grid_size)]
    mort=mort[mort>0]
    mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
    return mort
sns.set(font_scale=1.1)
data_label="Relative water content"
mort_label='Fractional area of mortality'
cmap='viridis'
### scatter plot linear scale 
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(3,6),sharex='col')
plt.subplots_adjust(hspace=0.13)
ax=axs[0]
species='evergreen'
mort=import_mort_leaf_habit(species=species)
x=data_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y)
plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
ax.set_ylabel(mort_label)
guess=(0.01,0.05,1e-4,1e-2)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
plot_data=ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                color='r',alpha=0.6)
ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
ax.set_title('%s trees'%species.title(),loc='left')
#ax.annotate('%s trees'%species, xy=(0.5, 0.43), xycoords='axes fraction',\
#            ha='left',size=10)

ax=axs[1]
species='deciduous'
mort=import_mort_leaf_habit(species=species)
x=data_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y)
plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                color='r',alpha=0.6)

ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
#fig.suptitle('Scatter plot relating mortality with indicators')
#ax.annotate('%s trees'%species, xy=(0.5, 0.4), xycoords='axes fraction',\
#            ha='left',size=10)
ax.set_title('%s trees'%species.title(),loc='left')
cbaxes = fig.add_axes([0.7, 0.75, 0.03, 0.05])
cb=fig.colorbar(plot2_data,ax=axs[1],\
                ticks=[min(z), max(z)],cax=cbaxes)
cb.ax.set_yticklabels(['Low', 'High'],size=6)
cbaxes.text(0,1.2,'Scatter plot\ndensity',size=8)
