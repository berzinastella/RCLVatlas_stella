# Lexi Jones
# Last edited: 12/09/22

import numpy as np
import math,os,itertools
from datetime import timedelta
from parcels import ParticleSet,AdvectionRK4,StatusCode
from config import *

def CalcVort(particle, fieldset, time):  
    """
    Calculates vorticity & velocity at the OceanParcels particle location.
    Utilizes auxillary coords surrounding the particle (*), where rho is the buffer distance from (*):
    
        |------x[1]------|
        |                |
        |                |
        |                |
        x[2]     *      x[0]
        |                |
        |                |
        |------x[3]------|
    
    """ 

    # Find auxillary coords where we will calculate velocities surrounding the particle
    rho = 0.01 #small distance away from the point to calculate the velocity -> used to calculate the gradients
    lat1_aux = particle.lat + rho
    lat3_aux = particle.lat - rho
    lon0_aux = particle.lon + rho
    lon2_aux = particle.lon - rho

    # Calculate velocities at auxillary coords
    # U & V velocities are converted to degree/sec under the hood, so we have to convert them back to m/s here
    u1_aux = fieldset.U[time,particle.depth,lat1_aux,particle.lon]*1852*60*math.cos(particle.lat*(math.pi/180))
    u3_aux = fieldset.U[time,particle.depth,lat3_aux,particle.lon]*1852*60*math.cos(particle.lat*(math.pi/180))
    v0_aux = fieldset.V[time,particle.depth,particle.lat,lon0_aux]*1852*60
    v2_aux = fieldset.V[time,particle.depth,particle.lat,lon2_aux]*1852*60

    # Calculate distance in meters between the auxillary coords
    earth_radius = 6.3781*(10**6) #meters
    deg_to_radians = math.pi/180
    lat_dist = earth_radius*(2*rho)*deg_to_radians
    lon_dist = lat_dist*math.cos(particle.lat*deg_to_radians) #depends on the latitude of the particle
    grad_xv = ((v0_aux-v2_aux)/lon_dist) #dv/dx
    grad_yu = ((u1_aux-u3_aux)/lat_dist) #du/dy
    
    ## Add the vorticity to the output  
    particle.vort = grad_xv - grad_yu
    
    #U & V velocities are converted to degree/sec under the hood, so we have to convert them back to m/s here
    particle.u = fieldset.U[time,particle.depth,particle.lat,particle.lon]*1852*60*math.cos(particle.lat*(math.pi/180))
    particle.v = fieldset.V[time,particle.depth,particle.lat,particle.lon]*1852*60

def DeleteParticle(particle, fieldset, time):
    """
    Stop a particle from running through the simulation and avoid an error that cancels the entire simulation.
    Used if a particle runs on land or out of bounds of the grid.
    """
    particle.delete()

def particle_grid2d(fieldset,custom_particle,lat_params,lon_params,time):
    """
    Create lats and lons arrays to create a 2D grid of particles in the correct format to feed into ParticleSet.from_list() method.
    
    Input
        custom_particle: The class for custom particle outputs 
        _params: list of the form [_start,_stop,_step]        
    Ouput
        pset_grid: Particle set to run through OceanParcels
        len(lats): equivalent to the number of particles in the grid
    """
    lon_array = np.arange(lon_params[0],lon_params[1],lon_params[2])
    lat_array = np.arange(lat_params[0],lat_params[1],lat_params[2])
    
    # Check 
    if (not len(lon_array)):
        print('Warning: Lon_array is empty.')
    if (not len(lat_array)):
        print('Warning: Lat_array is empty.')
    
    # Format desired coords for reading into `from_list` function
    lats = list(itertools.chain.from_iterable([list(lat_array)]*len(lon_array)))
    lons = list(itertools.chain.from_iterable([[l]*len(lat_array) for l in lon_array]))
    print('Number of particles initialized: %s'%(len(lats)))

### works until here    
    pset_grid = ParticleSet.from_list(
        fieldset = fieldset, 
        pclass = custom_particle, 
        lon = lons, 
        lat = lats, 
        depth = [0]*len(lons), 
        time = time
    )
    
    return pset_grid,len(lats)

def simulate_particles2d(pset,output_file_path,runtime,runtime_unit,timestep_mins,output_hrs,backwards_flag):
    """
    Simulate particles through the trajectory field.
    
    Input
        runtime_unit: 'hours' or 'days'
        runtime: amount of time (with the given units) to simulate particle trajectories
        timestep_mins: Time between when particle location is calculated
        output_hrs: Time between when particle location is outputed
        backwards_flag: 'y' (run backwards in time from the initialization) or 'n' (run forwards in time) 
    Output
        output_file will write the data if requested, otherwise the pset will now contain particles in their new locations
    """

    ## Create output file & remove contents of the existing file (if it already exists) to ensure the code writes the new data to the file

    ### stella- change this to work for .zarr otherwise you have to delete the folder every time
    if os.path.exists(output_file_path):
        f = open(output_file_path, 'w')
        f.close()      
    output_file = pset.ParticleFile(name=output_file_path,outputdt=timedelta(hours=output_hrs))

    ## Assign the custom kernel behavior to the particle set  
    custom_kernel = pset.Kernel(CalcVort)    
        
    ## Set up the multipliers based on user input
    if (runtime_unit == 'hours'):
        runtime_multiplier = 1
    elif (runtime_unit == 'days'):
        runtime_multiplier = 24
    else:
        print('Warning: Output should be in units of hours or days.')
    
    if (backwards_flag == 'y'): #run backwards in time
        timedelta_multiplier = -1
        print('Particles are being simulated backward in time.')
    else: #run forwards in time 
        timedelta_multiplier = 1
        print('Particles are being simulated forward in time.')
    
    ## Run the parcels simulation
    pset.execute(AdvectionRK4+custom_kernel,
                runtime=timedelta(hours=runtime*runtime_multiplier),
                dt=timedelta_multiplier*timedelta(minutes=timestep_mins),
                output_file=output_file
                # recovery={StatusCode.ErrorOutOfBounds: DeleteParticle}
                )        
    # output_file.close() #deletes the Parcels output folder with npy files we don't need anymore
    
def calc_LAVD(vort,output_freq,runtime):
    vort_avg_t = np.nanmean(vort,axis=0)[1:] #Find average vorticity over the entire spatial domain at each time step
    LAVD = np.trapz(np.absolute(vort[:,1:] - vort_avg_t), dx=output_freq*60*60, axis=1)/(runtime*24*60*60-output_freq*60*60) #trapz does the integration; convert data to be in seconds units
    return LAVD



def calc_LAVD_spec_time(vort,output_freq,runtime, LAVD_days):
    vort_avg_t = np.nanmean(vort,axis=0)[1:] #Find average vorticity over the entire spatial domain at each time step
    LAVD = np.trapz(np.absolute(vort[:,0:LAVD_days] - vort_avg_t[:LAVD_days]), dx=output_freq*60*60, axis=1)/(LAVD_days*24*60*60-output_freq*60*60) #trapz does the integration; convert data to be in se
    return LAVD



def trajectories_to_specific_day_LAVD(traj_path, LAVD_days):
    print("Calculating the LAVD ...")
    traj_ds = xr.open_dataset(traj_path) # open the Lagrangian trajectory dataset that was just produced
    vort_premask = traj_ds.variables["vort"]
    vort = np.array(vort_premask.where(vort_premask != 0)) #filters out land values
    LAVD = calc_LAVD_spec_time(vort,sim_params['output_freq'],sim_params['runtime'], LAVD_days)
    LAVD_output_file_path = traj_path + str(LAVD_days) +'days.npy'
    np.save(LAVD_output_file_path,LAVD)
    print('LAVD output file: %s'%(LAVD_output_file_path))
