import s3fs

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

def save_numpy_patches(outdir, abidt, Y, X, ch02_patch, ch05_patch, ch13_patch, ch15_patch, gt_patch):
    # Save each patch as a .npy file

    if isinstance(ch02_patch, np.ma.MaskedArray):
        ch02_patch = ch02_patch.filled(0)
    if isinstance(ch05_patch, np.ma.MaskedArray):
        ch05_patch = ch05_patch.filled(0)
    if isinstance(ch13_patch, np.ma.MaskedArray):
        ch13_patch = ch13_patch.filled(0)
    if isinstance(ch15_patch, np.ma.MaskedArray):
        ch15_patch = ch15_patch.filled(0)
    if isinstance(gt_patch, np.ma.MaskedArray):
        gt_patch = gt_patch.filled(0)

    ch02_filename = os.path.join(outdir, f'CH02_{abidt.strftime("%Y%m%d-%H%M%S")}_Y{Y}_X{X}.npy')
    ch05_filename = os.path.join(outdir, f'CH05_{abidt.strftime("%Y%m%d-%H%M%S")}_Y{Y}_X{X}.npy')
    ch13_filename = os.path.join(outdir, f'CH13_{abidt.strftime("%Y%m%d-%H%M%S")}_Y{Y}_X{X}.npy')
    ch15_filename = os.path.join(outdir, f'CH15_{abidt.strftime("%Y%m%d-%H%M%S")}_Y{Y}_X{X}.npy')
    gt_filename = os.path.join(outdir, f'FED_{abidt.strftime("%Y%m%d-%H%M%S")}_Y{Y}_X{X}.npy')

    # Save the .npy files
    np.save(ch02_filename, ch02_patch)
    np.save(ch05_filename, ch05_patch)
    np.save(ch13_filename, ch13_patch)
    np.save(ch15_filename, ch15_patch)
    np.save(gt_filename, gt_patch)

    print(f"Saved patches: {ch02_filename}, {ch05_filename}, {ch13_filename}, {ch15_filename}, {gt_filename}")


# We're going to scale our data int uint8 (bytes).
# bsinfo will contain instructions on min and max values to perform the scaling.
def bytescale(data_arr,vmin,vmax):
    assert(vmin < vmax)
    DataImage = np.round((data_arr - vmin) / (vmax - vmin) * 255.9999)
    DataImage[DataImage < 0] = 0
    DataImage[DataImage > 255] = 255
    return DataImage.astype(np.uint8)

def unbytescale(scaled_arr,vmin,vmax):
  assert(vmin < vmax)
  scaled_arr = scaled_arr.astype(np.float32)
  unscaled_arr = scaled_arr / 255.9999 * (vmax - vmin) + vmin
  return unscaled_arr

fs = s3fs.S3FileSystem(anon=True)

bsinfo = {} #bytescaling info
bsinfo['CH02'] = {'vmin':0, 'vmax':1}
bsinfo['CH05'] = {'vmin':0, 'vmax':0.75}
bsinfo['CH13'] = {'vmin':190, 'vmax':320}
bsinfo['CH15'] = {'vmin':190, 'vmax':320}
bsinfo['FED_accum_60min_2km'] = {'vmin':0, 'vmax':255}

glmpatt = 'glmL3/60min/agg/%Y%m%d-%H%M00.netcdf'
glmvar = "flash_extent_density"

sector = 'RadC'
ltgthresh = 3 #should be in byte-scaled space.
              #We set a minimum threshold of 3 flashes, so that we
              #don't get any patches with very sparse lightning.

outpatt = 'tfrecs/%Y/%Y%m%d/'
output_base_path = 'output_npy'


# This information is valid at **2-km** resolution
NY,NX = (1500,2500) # grid size
ny,nx = (480,480)   # patch size
NorthwestXs = np.arange(50,NX-nx,nx,dtype=int) # start at 50 to minimize missing data (space-look pixels)
NorthwestYs = np.arange(50,NY-ny,ny,dtype=int) # start at 50 to minimize missing data (space-look pixels)

truth_files = np.sort(glob.glob('glmL3/60min/agg/*netcdf'))

# For each target/truth file, find the corresponding ABI files and create
# the TFRecord for that time
for truth_file in truth_files:

  # This datetime is the *end* dt of the accumulation. We need to subtract an hour
  # to get the correct ABI time.
  glmdt = datetime.strptime(os.path.basename(truth_file),'%Y%m%d-%H%M00.netcdf')
  abidt = glmdt - timedelta(hours=1)

  # Now get the truth/target data
  nc = netCDF4.Dataset(truth_file)
  glm_agg = nc.variables[glmvar][:]
  nc.close()

  # Here, we bytescale the GLM data and then add the "channels" dimension at the end
  glm_agg = np.expand_dims(bytescale(glm_agg,bsinfo['FED_accum_60min_2km']['vmin'],bsinfo['FED_accum_60min_2km']['vmax']), axis=-1)

  # Now get the ABI data

  #CH02
  file_location = fs.glob(abidt.strftime('s3://noaa-goes16/ABI-L1b-RadC/%Y/%j/%H/*C02*_s%Y%j%H%M*.nc'))
  file_ob = [fs.open(file) for file in file_location]
  ds = xr.open_mfdataset(file_ob,combine='nested',concat_dim='time')
  ch02 = ds['Rad'][0].data.compute()
  ch02 = ch02 * ds['kappa0'].data[0] # Convert to reflectance
  ch02 = np.expand_dims(bytescale(ch02,bsinfo['CH02']['vmin'],bsinfo['CH02']['vmax']), axis=-1) #bytescale and add channels dim

  #CH05
  file_location = fs.glob(abidt.strftime('s3://noaa-goes16/ABI-L1b-RadC/%Y/%j/%H/*C05*_s%Y%j%H%M*.nc'))
  file_ob = [fs.open(file) for file in file_location]
  ds = xr.open_mfdataset(file_ob,combine='nested',concat_dim='time')
  ch05 = ds['Rad'][0].data.compute()
  ch05 = ch05 * ds['kappa0'].data[0] # Convert to reflectance
  ch05 = np.expand_dims(bytescale(ch05,bsinfo['CH05']['vmin'],bsinfo['CH05']['vmax']), axis=-1) #bytescale and add channels dim

  #CH13
  file_location = fs.glob(abidt.strftime('s3://noaa-goes16/ABI-L1b-RadC/%Y/%j/%H/*C13*_s%Y%j%H%M*.nc'))
  file_ob = [fs.open(file) for file in file_location]
  ds = xr.open_mfdataset(file_ob,combine='nested',concat_dim='time')
  ch13 = ds['Rad'][0].data.compute()
  # Convert to brightness temperature
  # First get some constants
  planck_fk1 = ds['planck_fk1'].data[0]; planck_fk2 = ds['planck_fk2'].data[0]; planck_bc1 = ds['planck_bc1'].data[0]; planck_bc2 = ds['planck_bc2'].data[0]
  ch13 = (planck_fk2 / (np.log((planck_fk1 / ch13) + 1)) - planck_bc1) / planck_bc2
  ch13 = np.expand_dims(bytescale(ch13,bsinfo['CH13']['vmin'],bsinfo['CH13']['vmax']), axis=-1) #bytescale and add channels dim

  #CH15
  file_location = fs.glob(abidt.strftime('s3://noaa-goes16/ABI-L1b-RadC/%Y/%j/%H/*C15*_s%Y%j%H%M*.nc'))
  file_ob = [fs.open(file) for file in file_location]
  ds = xr.open_mfdataset(file_ob,combine='nested',concat_dim='time')
  ch15 = ds['Rad'][0].data.compute()
  # Convert to brightness temperature
  # First get some constants
  planck_fk1 = ds['planck_fk1'].data[0]; planck_fk2 = ds['planck_fk2'].data[0]; planck_bc1 = ds['planck_bc1'].data[0]; planck_bc2 = ds['planck_bc2'].data[0]
  ch15 = (planck_fk2 / (np.log((planck_fk1 / ch15) + 1)) - planck_bc1) / planck_bc2
  ch15 = np.expand_dims(bytescale(ch15,bsinfo['CH15']['vmin'],bsinfo['CH15']['vmax']), axis=-1) #bytescale and add channels dim


  # Now make patches out of the data and write TFRecords.
  # Recall that GOES-16 ABI and GOES-16 GLM are on the same grids.
  # However, CH02 has 0.5-km resolution, CH05 has 1-km resolution, and
  # CH13 and CH15 have 2-km spatial resolution. That is why the patch sizes are different.
  for Y in NorthwestYs:
    for X in NorthwestXs:
      if(np.max(glm_agg[Y:Y+ny,X:X+nx]) >= ltgthresh): #check to make sure there is more than just 1 or 2 flashes in the patch.
          ch02_patch = ch02[Y*4:(Y+ny)*4, X*4:(X+nx)*4]
          ch05_patch = ch05[Y*2:(Y+ny)*2, X*2:(X+nx)*2]
          ch13_patch = ch13[Y:Y+ny, X:X+nx]
          ch15_patch = ch15[Y:Y+ny, X:X+nx]
          gt_patch = glm_agg[Y:Y+ny, X:X+nx]
          
          outdir = os.path.join(output_base_path, abidt.strftime('%Y/%Y%m%d'))
          os.makedirs(outdir, exist_ok=True)
          save_numpy_patches(outdir, abidt, Y, X, ch02_patch, ch05_patch, ch13_patch, ch15_patch, gt_patch)
