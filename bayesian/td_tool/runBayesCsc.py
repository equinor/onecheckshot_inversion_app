# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:33:52 2020

@author: disch
"""

import time
import numpy as np
import pandas as pd
import pickle
import td_lib 
import os
import matplotlib.pyplot as plt
import bayes_csc
from IPython import embed
from pandasgui import show

dataframe = pd.read_csv('td_tool/test_well.csv')

#####################
# START: PARAMETERS #
#####################

# Turn on/off processes
extend2zero = 1 # extend well log to z = 0 using seabed and analytical expression
doBayes = 1 # run bayesian check-shot correction 
plotWell = 1 # plot result
doExportPkl = 1 # export result as numerical pkl file
doExportPng = 1 # export result as numerical pkl file

# Input data
root_folder = 'demo_example/'
time_depth_folder = root_folder + 'input/td/'
well_pkl_folder = root_folder + 'input/pkl/'
water_depth_file = root_folder + 'input/UNIQUE_WELLBORE_IDENTIFIER_WATER_DEPTH.csv'

td_output_folder = root_folder + 'output/bayes_csc/'
png_output_folder = root_folder + 'output/bayes_csc/png/'
pkl_output_folder = root_folder + 'output/bayes_csc/pkl/'

# parameters
depth_log_name = 'TVDMSL'
velocity_log_name = 'LFP_VP_V'
water_velocity = 1480

###################
# END: PARAMETERS #
###################


# water depths
print('Reading water depth file ... ')
wd = pd.read_csv(water_depth_file)
wd.columns = ['WELL', 'DEPTH']
wd.WELL = [ww.replace(' ','_').replace('/','_') for ww in wd.WELL] # change formtting of well name
print('...done.')


# get list of all wells in folder
allfiles = os.listdir(well_pkl_folder)    
well_all = [ww[:-4] for ww in allfiles if os.path.isfile(os.path.join(well_pkl_folder, ww))] 
    
# Loop over all wells in folder and perform bayesian check-shot correction
for ww in well_all:  

    runWell = 1 # flag to skip well
    
    print('----')
    print('WELL: ' + ww)
    
    # read time-depth data
    td_file = time_depth_folder + ww + '.dat'
    print('Reading TD file ' + td_file + ' ...')
    try:
        td = pd.read_csv(td_file, header = 2)
    except:
        td = []
    print('...done')   
    
    if isinstance(td, pd.DataFrame): 
        
        # get data
        td_z = td['TVDSS'].values
        td_t = td['TWT'].values   
    else:
        print('ERROR: empty time depth - skipping well')
        runWell = 0 # skip this well
        
    if runWell:
            
        # read well data from file    
        pkl_file = well_pkl_folder + ww + '.pkl'        
        print('Reading file ' + pkl_file + ' ...')
        wlog0 = pd.read_pickle(pkl_file)        
        print('... done.')
        
        # get log data
        well_z = wlog0[depth_log_name].values
        well_vp = wlog0[velocity_log_name].values
        
        # stop if empty intput
        if len(well_z) >= 2:
            
            # create output dataframe
            df_well = pd.DataFrame({'TVDMSL': well_z, 
                                    'VP_IN': well_vp})
        else:
            print('ERROR: empty well log data - skipping well')
            runWell = 0 # skip this well
            
    # get water depth            
    if runWell:    
        
        water_depth = wd[wd['WELL'].str.startswith(ww)]['DEPTH'].values[0]                
        water_twt = 2 * water_depth / water_velocity
        ii = (td_z > water_depth)
        td_z = np.union1d([0, water_depth], td_z[ii])
        td_t = np.union1d([0, water_twt], td_t[ii])
          
    # add top part of logs
    if extend2zero and runWell:
        
        print('EXTEND logs to z = 0')
        print('   #log: ' + str(len(well_z)))
        well_z, well_vp = td_lib.extendLogsToZero(well_z, well_vp, water_depth, water_velocity)        
        print('   #log: ' + str(len(well_z)))
        print('...done')
        
        
        # merge output with existing dataframe
        df_ext = pd.DataFrame({'TVDMSL': well_z, 
                               'VP_EXT': well_vp})
        df_well = pd.merge(df_well, df_ext, how = 'outer', on=['TVDMSL'])    
        df_well = df_well.sort_values('TVDMSL').reset_index(drop = True)       


    if doBayes and runWell:
        
        # set parameters for bayesian check-shot correction
        par = bayes_csc.getDefaultPar()        
        par['apply_corr'] = 0
        par['istep_bayes'] = 1
        par['std_vp_mode'] = 2
        par['std_vp_const'] = 500
        par['std_t_const'] = 0.005            
        par['zstart'] = well_z[1] # start at seabed 
        try:
        #if 1:
            
            print('BAYESIAN check-shot correction...')
            embed()
            well_vp, well_z = bayes_csc.runCsc(well_z, well_vp, td_z, td_t, par)
            print('...done')
            # merge output with existing dataframe
            #well_vp = well_vp
            df_bayes = pd.DataFrame({'TVDMSL': well_z, 
                                     'VP_BAYES': well_vp})
            
            df_well = pd.merge(df_well, df_bayes, how = 'outer', on=['TVDMSL'])   
            df_well = df_well.sort_values('TVDMSL').reset_index(drop = True)                
            
            # plot well                
            fig = bayes_csc.bayes_well_plot(df_well, td_z, td_t, ww, water_depth = water_depth, water_vel = water_velocity)
            
            # save output data
            bayes_csc_out = {'well_name': ww, 'td' : td, \
                             'water_depth': water_depth, 'water_vel': water_velocity, \
                             'df_well': df_well}
            
            # export numerical output data
            if doExportPkl:
                
                output_pkl_file = pkl_output_folder + ww + '.pkl'                    
                print('Exporting PKL: ' + output_pkl_file + ' ...')
                with open(output_pkl_file, 'wb') as handle:
                    pickle.dump(bayes_csc_out, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print ('...done')                                
            
            # export image of plot
            if doExportPng:                                    
                output_png_file = png_output_folder + ww + '.png'
                print('Exporting PNG: ' + output_png_file + ' ...')     
                fig.savefig(output_png_file)           
                print('...done')
                
            
            
            
        except Exception as e:
            print(e.args)
            print('ERROR: Bayesian step failed')
                        