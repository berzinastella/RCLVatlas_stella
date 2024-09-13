# Lexi Jones
# Date Created: 07/15/21
# Last Edited: 12/20/22

# Run an OceanParcels simulation
## this is the script that looks at config file and calculates lavd from lexi jones

import time,sys
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime
from parcels import FieldSet,Variable,JITParticle, Field
from config import *


################# figure out which of these are actually useful



# from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator

from numpy import arange, ones
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
from matplotlib import colors 
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from datetime import datetime, timedelta
from netCDF4 import Dataset

import io
import os
import warnings
warnings.filterwarnings("ignore")

import gsw
from xgcm.grid import Grid

# from concurrent.futures import ProcessPoolExecutor
# from concurrent.futures import ThreadPoolExecutor

import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,LatitudeLocator)
import cartopy.feature as cfeature

import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
import glob
import intake
from pathlib import Path
import dask
import cmocean.cm as cmo #also for pretty color palettes
import pandas as pd
dask.config.set({"array.slicing.split_large_chunks": True}) 
import sys

# to access intake catalog of eerie
eerie_cat=intake.open_catalog("https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml")
data_oce = eerie_cat["dkrz"]["disk"]["model-output"]["icon-esm-er"]["eerie-control-1950"]["v20231106"]["ocean"]["gr025"]["2d_daily_mean"].to_dask()

sys.path.append('./RCLVatlas/')
from functions_for_parcels import *

#this finds the date from what you write in the command line
date_input = sys.argv[1] # user input: particle intialization date

start_year,start_month,start_day = int(str(date_input)[0:4]),int(str(date_input)[4:6]),int(str(date_input)[6:8])
start_date = datetime(start_year,start_month,start_day) # format datetime
print(start_year, start_month)
data_sub=data_oce.sel(time=str(start_year))
data_sub=data_sub[["u", "v"]]

data_sub['v']=data_sub['v'][:,0,:,:]
data_sub['u']=data_sub['u'][:,0,:,:]


#### this code chunk is from xlcs package
ds0=data_sub
ds = ds0.copy(deep=True)
lon, lat = np.meshgrid(ds.lon, ds.lat)
ds = ds.assign_coords(
    {
        "dx": xr.DataArray(
            gsw.distance(lon, lat, axis=1),
            dims=["lat", "longitude_g"],
            coords={
                "lat": ds.lat.data,
                "longitude_g": 0.5 * (ds.lon.data[1:] + ds.lon.data[:-1]),
            },
        ),
        "dy": xr.DataArray(
            gsw.distance(lon, lat, axis=0),
            dims=["latitude_g", "lon"],
            coords={
                "latitude_g": 0.5 * (ds.lat.data[1:] + ds.lat.data[:-1]),
                "lon": ds.lon.data,
            },
        ),
    }
)

coords = {
    "X": {"center": "lon", "inner": "longitude_g"},
    "Y": {"center": "lat", "inner": "latitude_g"},
}
grid = Grid(ds, periodic=[], coords=coords)


variables = {"U": "u", "V": "v"}

dimensions = {
    "time": "time",
    "lon": "lon",
    "lat": "lat",
}

fs = FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh="spherical")
print('Fieldset created.')

#### now calculate vorticity
vg_x = grid.diff(ds.v, "X", boundary="extend") / ds.dx
ug_y = grid.diff(ds.u, "Y", boundary="extend") / ds.dy
ds["vorticity"] = grid.interp(vg_x, "X", to="center", boundary="extend") - grid.interp(
    ug_y, "Y", to="center", boundary="extend"
)
# add the field to the FS object
field1 = Field.from_xarray(ds["vorticity"], "vorticity", dimensions)
fs.add_field(field1)



print("Vorticity added")

### end of xlcs package code chunk



####lexis original code
### Create particleset ###
class SpinnyParticle(JITParticle):
    u = Variable('u',dtype=np.float64)
    v = Variable('v',dtype=np.float64)
    vort = Variable('vort',dtype=np.float64)
    
pset_dynamic,num_particles = particle_grid2d(fs,SpinnyParticle,
                                             [grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],grid_bounds['lag_grid_res']],
                                             [grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res']],
                                             start_date)


### stella trying to fix it











### Execute particle simulation ###
print("Running Lagrangian simulation ...")
traj_output_file_path = lag_traj_dir + str(date_input) + '_' + filename_str + '.zarr'
simulate_particles2d(pset_dynamic,traj_output_file_path,
                     sim_params['runtime'],sim_params['runtime_unit'],
                     sim_params['timestep'],sim_params['output_freq'],
                     sim_params['backwards'])
print('Trajectory output file: %s'%(traj_output_file_path))

### Calculate Lagrangian average vorticity deviation (LAVD) ###
print("Calculating the LAVD ...")
traj_ds = xr.open_dataset(traj_output_file_path) # open the Lagrangian trajectory dataset that was just produced
vort_premask = traj_ds.variables["vort"]
vort = np.array(vort_premask.where(vort_premask != 0)) #filters out land values
LAVD = calc_LAVD(vort,sim_params['output_freq'],sim_params['runtime'])
LAVD_output_file_path = LAVD_dir + str(date_input) + '_LAVD_' + filename_str + '.npy'
np.save(LAVD_output_file_path,LAVD)
print('LAVD output file: %s'%(LAVD_output_file_path))





# #### version 2 original 
# # Lexi Jones
# # Date Created: 07/15/21
# # Last Edited: 12/20/22

# # Run an OceanParcels simulation

# import time,sys
# import numpy as np
# import xarray as xr
# from glob import glob
# from datetime import datetime
# from parcels import FieldSet,Variable,JITParticle
# from config import *

# sys.path.append('./RCLVatlas/')
# from functions_for_parcels import *

# date_input = sys.argv[1] # user input: particle intialization date

# start_year,start_month,start_day = int(str(date_input)[0:4]),int(str(date_input)[4:6]),int(str(date_input)[6:8])
# start_date = datetime(start_year,start_month,start_day) # format datetime

# ### Create Parcels fieldset ###
# parcels_input_files = sorted(glob(gos_vel_dir+'dt_global_allsat_phy_l4_*.nc'))
# filenames = {'U': parcels_input_files,'V': parcels_input_files}
# variables = {'U': 'ugos','V': 'vgos'} #name of the velocity variables in the netCDF file
# dimensions = {'U': {'lon':'longitude','lat':'latitude','time':'time'},
#               'V': {'lon':'longitude','lat':'latitude','time':'time'}}
# fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
# print('Fieldset created.')

# ### Create particleset ###
# class SpinnyParticle(JITParticle):
#     u = Variable('u',dtype=np.float64)
#     v = Variable('v',dtype=np.float64)
#     vort = Variable('vort',dtype=np.float64)
    
# pset_dynamic,num_particles = particle_grid2d(fieldset,SpinnyParticle,
#                                              [grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],grid_bounds['lag_grid_res']],
#                                              [grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res']],
#                                              start_date)

# ### Execute particle simulation ###
# print("Running Lagrangian simulation ...")
# traj_output_file_path = lag_traj_dir + str(date_input) + '_' + filename_str + '.nc'
# simulate_particles2d(pset_dynamic,traj_output_file_path,
#                      sim_params['runtime'],sim_params['runtime_unit'],
#                      sim_params['timestep'],sim_params['output_freq'],
#                      sim_params['backwards'])
# print('Trajectory output file: %s'%(traj_output_file_path))

# ### Calculate Lagrangian average vorticity deviation (LAVD) ###
# print("Calculating the LAVD ...")
# traj_ds = xr.open_dataset(traj_output_file_path) # open the Lagrangian trajectory dataset that was just produced
# vort_premask = traj_ds.variables["vort"]
# vort = np.array(vort_premask.where(vort_premask != 0)) #filters out land values
# LAVD = calc_LAVD(vort,sim_params['output_freq'],sim_params['runtime'])
# LAVD_output_file_path = LAVD_dir + str(date_input) + '_LAVD_' + filename_str + '.npy'
# np.save(LAVD_output_file_path,LAVD)
# print('LAVD output file: %s'%(LAVD_output_file_path))

