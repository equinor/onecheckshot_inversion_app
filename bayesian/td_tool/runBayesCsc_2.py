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
from pathlib import Path
import bayes_csc
from IPython import embed
import csv
import sys
import json

def get_data():
    directory = Path(__file__).parents[2]/'data/checkshot_sonic_table_well.csv'
    dataframe_sonic_checkshot = pd.read_csv(directory)
    return dataframe_sonic_checkshot
#dataframe = pd.read_csv('td_tool/test_well_8.csv')
dataframe_sonic_checkshot = get_data()

df_output = []
n=0

#####################
# START: PARAMETERS #
#####################

# Turn on/off processes
extend2zero = 1 # extend well log to z = 0 using seabed and analytical expression
doBayes = 1 # run bayesian check-shot correction 
plotWell = 1 # plot result
doExportPkl = 1 # export result as numerical pkl file
doExportPng = 1 # export result as numerical pkl file
doExportCsv = 1

# Input data
root_folder = Path(__file__).parents[1]/'demo_example/'
time_depth_folder = str(root_folder/'input/td/')
well_pkl_folder = str(root_folder/'input/pkl/')
water_depth_file = str(root_folder/'input/UNIQUE_WELLBORE_IDENTIFIER_WATER_DEPTH.csv')

td_output_folder = str(root_folder/'output/bayes_csc/')
png_output_folder = str(root_folder/'output/bayes_csc/png/')
pkl_output_folder = str(root_folder/'output/bayes_csc/pkl/')
csv_output_folder = str(root_folder/'output/bayes_csc/csv/')
files_in_folder = str(os.listdir(pkl_output_folder))
files_in_folder = [well.replace('.pkl','') for well in files_in_folder]
total_wells = dataframe_sonic_checkshot.uwi.unique()
total_wells = [well.replace(' ','_').replace('/','_') for well in total_wells]
wells_not_loaded = [x for x in total_wells if x not in files_in_folder]

# parameters
depth_log_name = 'tvd_ss'
velocity_log_name = 'vp'
water_velocity = 1478

###################
# END: PARAMETERS #
###################

for i in dataframe_sonic_checkshot.uwi.unique():
    n = 1 + n
    if n > 5:
        break
    else:
        pass
    try:
        dataframe = dataframe_sonic_checkshot[dataframe_sonic_checkshot.uwi == i]
        df_checkshot = dataframe[['tvd_ss','twt picked', 'average velocity', 'interval velocity']].dropna(subset='twt picked')
        df_sonic = dataframe[['tvd_ss','vp', ]].dropna(subset='vp')

        # water depths
        print('Reading water depth file ... ')
        wd = dataframe

        #wd.WELL = [ww.replace(' ','_').replace('/','_') for ww in wd.WELL] # change formtting of well name
        print('...done.')


        # get list of all wells in folder
        allfiles = os.listdir(well_pkl_folder)    
        well_all = [ww[:-4] for ww in allfiles if os.path.isfile(os.path.join(well_pkl_folder, ww))] 

        # Loop over all wells in folder and perform bayesian check-shot correction

        ww = dataframe.uwi.unique()[0]
        ww = ww.replace(' ','_').replace('/','_')

        runWell = 1 # flag to skip well

        print('----')
        print('WELL: ' + ww)

        # read time-depth data
        td = df_checkshot
        if isinstance(td, pd.DataFrame): 
            
            # get data
            td_z = td['tvd_ss'].values
            td_t = td['twt picked'].values
            td_t = td_t*0.001
        else:
            print('ERROR: empty time depth - skipping well')
            runWell = 0 # skip this well
            

        print('... done.')

        # get log data
        well_z = df_sonic['tvd_ss'].values
        well_vp = df_sonic['vp'].values

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
            water_depth = df_checkshot.loc[df_checkshot['average velocity'] == 1478]['tvd_ss']    
            water_depth = float(water_depth.iloc[0])    
            water_twt = df_checkshot.loc[df_checkshot['average velocity'] == 1478]['twt picked']
            water_twt = float(water_twt.iloc[0])

            ii = (td_z > water_depth)
            #td_z = np.union1d([0, water_depth], td_z[ii])
            #td_t = np.union1d([0, water_twt], td_t[ii])
    
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
            #df = pd.DataFrame({'tvdss': well_z, 'vp': well_vp})
            
        if doBayes and runWell:
            
            # set parameters for bayesian check-shot correction
            par = bayes_csc.getDefaultPar()        
            par['apply_corr'] = 0
            par['istep_bayes'] = 1
            par['std_vp_mode'] = 2
            par['std_vp_const'] = 500
            par['std_t_const'] = 0.005
            par['apply_corr'] = 1
            par['corr_order'] = 1.8
            par['corr_range'] = 100            
            par['zstart'] = well_z[1] # start at seabed 
            try:
            #if 1:

                print('BAYESIAN check-shot correction...')
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
                df_output.append(bayes_csc_out)
                embed()
                # export numerical output data
                if doExportPkl:
                    
                    output_pkl_file = pkl_output_folder + ww + '.pkl'                    
                    print('Exporting PKL: ' + output_pkl_file + ' ...')
                    with open(str(output_pkl_file), 'wb') as handle:
                        pickle.dump(bayes_csc_out, handle, protocol = pickle.HIGHEST_PROTOCOL)
                    print ('...done')                                
                
                # export image of plot
                if doExportPng:                                   
                    output_png_file = png_output_folder + ww + '.png'
                    print('Exporting PNG: ' + output_png_file + ' ...')     
                    fig.savefig(str(output_png_file))           
                    print('...done')

                
            except Exception as e:
                print(e.args)
                print('ERROR: Bayesian step failed')
    except:
        print(f'Bayesian inference could not be applied for well {i}')
        pass
print('finished')
                        