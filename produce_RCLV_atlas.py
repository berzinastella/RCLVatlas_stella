# RCLVatlas Pipeline:
# 1. Identify RCLVS using floater package
#      - Checks multiple CDs (0.01,0.02,0.03)
#      - Sets dispersal threshold with CI (>=-0.5)
#      - 85% of particles must have the same sign of the vorticity
# 2. Track the RCLVs through time and ID them
# 3. QC: Check if any RCLVs skipped a date (or 2)
#      - Interpolate over the missing timestep using particle trajectories
# 4. Interpolate eddy bounds for first 3 timesteps
#      - Tracks the RCLV birth 
# 5. QC: Address instances of overlapping contours
# 6. Give RCLVs an age

# Step 1 takes the longest amount of time, followed by step 4. It is recommended to run step 1 in parallel on subsets
# of the dates of interest. The remaining functions need to be run linearly with all of the RCLV data. 
#
# This script is set up so that each step can be run one at a time, if desired.

# Lexi Jones
# Last Edited: 12/20/22

import os
import numpy as np
from config import *

sys.path.append('./RCLVatlas/')
from subfunctions_for_RCLV_atlas import read_RCLV_CSV_untracked,read_RCLV_CSV_tracked,save_RCLV_CSV
from mainfunctions_for_RCLV_atlas import *

# Retrieve initialization dates from LAVD directory
dates = []
for filename in os.listdir(LAVD_dir):
    dates.append(filename[0:8].replace('-',''))    
date_list = np.sort(np.unique(dates)).tolist()[::-1] #Reverse the order because we will iterate from the latest date to the earliest

# Edit this if you want to create an atlas for a subset of years, or run the set_up_RCLV_atlas() in parallel on a subset of years.
# I typically run one year at a time for set_up_RCLV_atlas() because this step takes several hours to run 
start_year,end_year = 2010,2011 
date_list = [i for i in date_list if ((int(i[0:4]) >= start_year) and (int(i[0:4]) <= end_year))]

# Notes about weird RCLVs are written to the log file 
#log_file = open('%sRCLV_%s_%s_log_file.txt'%(RCLV_dir,date_list[-1],date_list[0]),'a')

####################################### 1. Identify RCLVS #######################################
RCLV_data = set_up_RCLV_atlas(date_list) 
save_RCLV_CSV(RCLV_data,'%sRCLV_%s_%s_untracked.csv'%(RCLV_dir,date_list[-1],date_list[0]))

####################################### 2. Track the RCLVs through time and ID them #######################################

# The commented lines show an example of running the set_up_RCLV_atlas() on individual years, and then combining them for the
# tracking step. Cannot run the track_and_ID_RCLVs() function in parallel, all of the data must be read in together. 
#file1 = read_RCLV_CSV_untracked('%sRCLV_2010_2011_untracked.csv'%(output_dir),1)
#file2 = read_RCLV_CSV_untracked('%sRCLV_2012_2013_untracked.csv'%(output_dir),0)
#file3 = read_RCLV_CSV_untracked('%sRCLV_2014_2015_untracked.csv'%(output_dir),0)
#file4 = read_RCLV_CSV_untracked('%sRCLV_2016_2017_untracked.csv'%(output_dir),0)
#file5 = read_RCLV_CSV_untracked('%sRCLV_2018_2019_untracked.csv'%(output_dir),0)
#RCLV_data = np.array(np.concatenate((file1,file2,file3,file4,file5)))
    
#RCLV_data = np.array(read_RCLV_CSV_untracked('%sRCLV_%s_%s_untracked.csv'%(RCLV_dir,date_list[-1],date_list[0]),1),dtype=object)
#RCLV_data = track_and_ID_RCLVs(RCLV_data,date_list)
#save_RCLV_CSV(RCLV_data,'%sRCLV_%s_%s_tracked_with_ID.csv'%(RCLV_dir,date_list[-1],date_list[0])) # Save the tracked data as a CSV

####################################### 3. QC: Check if any RCLVs skipped a date (or 2) #######################################
#RCLV_data = read_RCLV_CSV_tracked('%sRCLV_%s_%s_tracked_with_ID.csv'%(RCLV_dir,date_list[-1],date_list[0]))   
#RCLV_data = interpolate_skipped_contours(RCLV_data,log_file,date_list)
#save_RCLV_CSV(RCLV_data,'%sRCLV_%s_%s_skips_interpolated.csv'%(RCLV_dir,date_list[-1],date_list[0]))

####################################### 4. Interpolate eddy bounds for first 3 timesteps #######################################
#RCLV_data = read_RCLV_CSV_tracked('%sRCLV_%s_%s_skips_interpolated.csv'%(RCLV_dir,date_list[-1],date_list[0])
#RCLV_data = interpolate_first_3timesteps(RCLV_data,log_file,date_list)
#save_RCLV_CSV(,'%sRCLV_%s_%s_first_3timesteps_interpolated.csv'%(RCLV_dir,date_list[-1],date_list[0])) 

####################################### 5. QC: Address instances of overlapping contours #######################################
#RCLV_data = np.array(read_RCLV_CSV_tracked('%sRCLV_%s_%s_first_3timesteps_interpolated.csv'%(RCLV_dir,date_list[-1],date_list[0])),dtype=object)
#RCLV_data = overlapping_RCLV_QC(RCLV_data,log_file,date_list)
#save_RCLV_CSV(RCLV_data,'%sRCLV_%s_%s_overlap_QC.csv'%(RCLV_dir,date_list[-1],date_list[0]))

####################################### 6. Give RCLVs an age #######################################
#RCLV_data = read_RCLV_CSV_tracked('%sRCLV_%s_%s_overlap_QC.csv'%(RCLV_dir,date_list[-1],date_list[0]))
#RCLV_data = age_RCLVs(RCLV_data)
#save_RCLV_CSV(RCLV_data,'%sRCLV_%s_%s_atlas.csv'%(RCLV_dir,date_list[-1],date_list[0])) # Save the final dataset

#####################################################################################################################                                  
#log_file.close()
