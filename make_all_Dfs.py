# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:08:35 2017

@author: kkrao
"""

from dirs import *
os.chdir(Dir_CA)

store_major=pd.HDFStore('data.h5')
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
mort.index=pd.to_datetime(mort.index,format='%Y')
store_major['mortality_005_grid']=mort

store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort.index.name='gridID'
mort=mort.T
mort.drop('gridID',inplace=True)
mort.index=[x[-4:] for x in mort.index] 
mort.index=pd.to_datetime(mort.index,format='%Y')
mort.fillna(0,inplace=True)
store_major['mortality_025_grid']=mort

store = pd.HDFStore(Dir_CA+'/vodDf.h5')#ascending is 1:30 PM
VOD_PM=store['vodDf']
VOD_PM.index.name='gridID'
VOD_PM=VOD_PM.T
VOD_PM.drop('gridID',inplace=True)
VOD_PM.index=[x[:-1] for x in VOD_PM.index] 
VOD_PM.index=pd.to_datetime(VOD_PM.index,format='%Y%j')
VOD_PM=VOD_PM[VOD_PM.index.dayofyear!=366]           
store_major['vod_pm']=VOD_PM
           
store = pd.HDFStore(Dir_CA+'/vod_D_Df.h5')
VOD_AM=store['vod_D_Df']
VOD_AM=VOD_AM[VOD_AM.index.dayofyear!=366]           
store_major['vod_am']=VOD_AM

store = pd.HDFStore(Dir_CA+'/LAI.h5')        
store_major['LAI_005_grid'] = store['LAI_smallgrid']
store_major['LAI_025_grid'] = store['LAI_grid']
          
store = pd.HDFStore(Dir_CA+'/sigma0.h5')
sigma0=store['sigma0']
sigma0.index=pd.to_datetime(sigma0.index,format='%Y%j')           
store_major['vod_005_grid']=sigma0

store = pd.HDFStore(Dir_CA+'/Young_Df.h5') 
cwd=store['cwd_acc']
cwd.index.name='gridID'
cwd=cwd.T
cwd.drop('gridID',inplace=True)
cwd.index=pd.to_datetime(cwd.index,format='%Y')
store_major['cwd']=cwd
    
store_major.close()
store.close()

###----------------------------------------------------------------------------
def seasonal_transform(data,season):
    start_month=7
    months_window=3
    if season=='win':
        start_month=1
    data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
    return data

store['RWC']=data_anomaly 
cwd=store['cwd']
cwd.index.name='cwd'
store['cwd']=cwd
     
mort=store['mortality_025_grid']
mort.index.name='FAM'
store['mortality_025_grid']=mort

LAI=store['LAI_025_grid']
for season in ['sum','win']:
    df=seasonal_transform(LAI,season)
    df=df.groupby(df.index.year).mean()
    df.index=pd.to_datetime(df.index,format='%Y')
    df.index.name='LAI_%s'%season
    store['LAI_025_grid_%s'%season]=df

###----------------------------------------------------------------------------
df=store['BPH_025_grid']
df.index.name='live_tree_density'
store['BPH_025_grid']=df

###----------------------------------------------------------------------------
import pickle
os.chdir(Dir_CA)
nos=370
with open('grid_to_species.txt', 'rb') as fp:
    for_type = pickle.load(fp)
Df=pd.DataFrame(np.full((11,nos),np.NaN), \
                           index=[pd.to_datetime(year_range,format='%Y')],\
                                 columns=range(nos))
for species in ['c','d']:
    index=[i[0] for i in for_type if i[1]==species]
    Df.loc[:,Df.columns.intersection(index)]=species
Df.replace(['c','d'],['evergreen','deciduous'],inplace=True)
Df.index.name='dominant_leaf_habit'
store[Df.index.name]=Df
store.close()
###----------------------------------------------------------------------------
store=pd.HDFStore(Dir_CA+'/data.h5')
for field in ['ppt']:
    for season in ['sum','win']:
        df=store[field]
        df=seasonal_transform(df,season)
        df=df.groupby(df.index.year).sum()
        df.index=pd.to_datetime(df.index,format='%Y')
        df.index.name='%s_%s'%(field,season)
        store[df.index.name]=df
###----------------------------------------------------------------------------
store=pd.HDFStore(Dir_CA+'/data.h5')
for field in ['PEVAP','EVP']:
    for season in ['sum','win']:
        df=store[field]
        df=seasonal_transform(df,season)
        df=df.groupby(df.index.year).sum()
        df.index=pd.to_datetime(df.index,format='%Y')
        df.index.name='%s_%s'%(field,season)
        store[df.index.name]=df

###----------------------------------------------------------------------------
store=pd.HDFStore(Dir_CA+'/data.h5')
for field in ['vpdmax']:
    for season in ['sum','win']:
        df=store[field]
        df=seasonal_transform(df,season)
        df=df.groupby(df.index.year).mean()
        df.index=pd.to_datetime(df.index,format='%Y')
        df.index.name='%s_%s'%(field,season)
        store[df.index.name]=df
store=pd.HDFStore(Dir_CA+'/data.h5')
###----------------------------------------------------------------------------
for field in ['tmax']:
    for season in ['sum','win']:
        df=store[field]
        df=seasonal_transform(df,season)
        df=df.groupby(df.index.year).mean()
        df.index=pd.to_datetime(df.index,format='%Y')
        df.index.name='%s_%s'%(field,season)
        store[df.index.name]=df             
           