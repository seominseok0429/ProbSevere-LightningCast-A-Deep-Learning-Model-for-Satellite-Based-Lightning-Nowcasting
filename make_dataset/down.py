import s3fs

# Python Standard Lib
import os
import itertools
from datetime import datetime, timedelta
import collections
import glob
import errno
import pickle
import shutil

# Scientific packages
import netCDF4
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import sklearn
import sklearn.metrics

def download_glm_files(bucketpatt):
  #bucketpatt, e.g.: 's3://noaa-goes16/GLM-L2-LCFA/2022/240/20/OR_GLM-L2-LCFA_G16_s2022240200*.nc'
  file_list = fs.glob(f'{bucketpatt}')

  for ff in file_list:
    parts = ff.split('/')
    satellite = parts[0]
    prefix = [f"https://{satellite}.s3.amazonaws.com"]
    url = ('/').join(prefix + parts[1:])
    #print(url)
    os.system(f'wget -q {url}')

fs = s3fs.S3FileSystem(anon=True) #connect to s3 bucket!

download_glm_files('s3://noaa-goes16/GLM-L2-LCFA/2022/240/2[0,1]/OR_GLM-L2-LCFA_G16_s*.nc')

