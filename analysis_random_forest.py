# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:32:07 2017

@author: kkrao
"""

from dirs import *
from sklearn.ensemble import RandomForestRegressor
store=pd.HDFStore('data.h5')
LAI=store['LAI]
X=data_anomaly.copy()
y=mort.copy()
y.fillna(0,inplace=True)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
print(regr.feature_importances_)
