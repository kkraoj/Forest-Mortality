import urllib
from dirs import*


data=['ppt','tmax','tmean']
year_range=range(2005,2017)
link='http://services.nacse.org/prism/data/public/4km/'
for file in data:
    os.chdir(MyDir+'/PRISM')
    if not(os.path.isdir(MyDir+'/PRISM/'+file)):        
        os.mkdir(file)
    os.chdir(MyDir+'/PRISM/'+file)
    for year in year_range:
        linkname=link+file+'/%d'%year
        filename='PRISM_%s_stable_4kmM2_%d_bil.zip'%(file,year)
        if not(os.path.isfile(filename)):
            urllib.urlretrieve(linkname,filename)

                   