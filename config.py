#stellas version
# Configuration file
# The user should define directory paths and adjust parameters here.

import numpy as np

### Directory paths ### 
# # gos_vel_dir = './CMEMS_data/' # Geostrophic velocity directory
# lag_traj_dir = '/work/bk1377/b382618/RCLVs/32day/lang_traj/' # Lagrangian trajectory directory
# LAVD_dir = '/work/bk1377/b382618/RCLVs/32day/LAVD/' # LAVD directory 
# RCLV_dir = '/work/bk1377/b382618/RCLVs/32day/RCLV_dir/' #RCLVatlas directory

# lag_traj_dir = '/work/bk1377/b382618/RCLVs/f8day/lang_traj/' # Lagrangian trajectory directory
# LAVD_dir = '/work/bk1377/b382618/RCLVs/f8day/LAVD/' # LAVD directory 
# RCLV_dir = '/work/bk1377/b382618/RCLVs/f8day/RCLV_dir/' #RCLVatlas directory

lag_traj_dir = '/work/bk1377/b382618/RCLVs/8day/lang_traj/' # Lagrangian trajectory directory
LAVD_dir = '/work/bk1377/b382618/RCLVs/8day/LAVD/' # LAVD directory 
RCLV_dir = '/work/bk1377/b382618/RCLVs/8day/RCLV_dir_positive/' #RCLVatlas directory



#path = "/work/bk1377/b382618/RCLVs/8day/RCLV_dir/RCLV_20100501_20100906_atlas.csv"

grid_bounds = {'lon_bound_west':180.0,
               'lon_bound_east':210.0,
               'lat_bound_south':10.0,
               'lat_bound_north':40.0,
               'lag_grid_res':0.03125}


## NW quadrant
# grid_bounds = {'lon_bound_west':0.1,
#                'lon_bound_east':10.0,
#                'lat_bound_south':0.1,
#                'lat_bound_north':10.0,
#                'lag_grid_res':0.03125}


sim_params = {'runtime':32, 
              'runtime_unit':'days',
              'timestep':20, 
              'output_freq':6,
              'backwards':'y'} 

# NOTE: min_dist and min_area are in units of pixels, relative to the LAVD field
RCLV_params = {'min_dist':24,
               'min_area':104,
               'init_contour_step_frac':0.1,
               'convex_def_tol':0.001}

filename_str = '%sdays_runtime_%smin_timestep_particle_start_lat_%s_%s_lon_%s_%s_spatial_step_%s_%shr_output_freq'%(
    sim_params['runtime'],sim_params['timestep'],grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],
    grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res'],sim_params['output_freq'])

traj_lon_array = np.arange(grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res'])
traj_lat_array = np.arange(grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],grid_bounds['lag_grid_res'])





### version 2 original

# import numpy as np

# ### Directory paths ### 
# gos_vel_dir = './CMEMS_data/' # Geostrophic velocity directory
# lag_traj_dir = './lag_trajs/' # Lagrangian trajectory directory
# LAVD_dir = './LAVD/' # LAVD directory 
# RCLV_dir = './RCLVs/' #RCLVatlas directory

# grid_bounds = {'lon_bound_west':199.0,
#                'lon_bound_east':204.0,
#                'lat_bound_south':17.0,
#                'lat_bound_north':22.0,
#                'lag_grid_res':0.03125}

# sim_params = {'runtime':32, 
#               'runtime_unit':'days',
#               'timestep':20, 
#               'output_freq':6,
#               'backwards':'y'} 

# # NOTE: min_dist and min_area are in units of pixels, relative to the LAVD field
# RCLV_params = {'min_dist':24,
#                'min_area':104,
#                'init_contour_step_frac':0.1,
#                'convex_def_tol':0.001}

# filename_str = '%sdays_runtime_%smin_timestep_particle_start_lat_%s_%s_lon_%s_%s_spatial_step_%s_%shr_output_freq'%(
#     sim_params['runtime'],sim_params['timestep'],grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],
#     grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res'],sim_params['output_freq'])

# traj_lon_array = np.arange(grid_bounds['lon_bound_west'],grid_bounds['lon_bound_east'],grid_bounds['lag_grid_res'])
# traj_lat_array = np.arange(grid_bounds['lat_bound_south'],grid_bounds['lat_bound_north'],grid_bounds['lag_grid_res'])