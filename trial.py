import urllib
from dirs import*
from netCDF4 import Dataset

os.chdir(MyDir+'/PET')
data=['sph','pr','tmmn','tmmx','pdsi','pet']
year_range=range(2005,2017)
my_example_nc_file = './'+data[5]+'/'+data[5]+'_'+'%d'%year_range[0]+'.nc'
fh = Dataset(my_example_nc_file, mode='r')

#lons = fh.variables['lon'][:]
#lats = fh.variables['lat'][:]
#tmax = fh.variables[data[5]][:]
                         

                   