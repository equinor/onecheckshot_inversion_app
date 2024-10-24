# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:33:52 2020

@author: disch
"""

import time
import numpy as np
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.getcwd())
from bayesian.td_tool.td_lib import extendLogsToZero
import matplotlib.pyplot as plt
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython import embed
from bayesian.td_tool.bayes_csc import Run_Bayesian, getDefaultPar,bayes_well_plot
import csv
import sys
import json
import streamlit as st
import time

class Bayesian_Inference():
    def __init__(self):
        pass

    def get_data(self, uwi):
        self.directory = Path(__file__).parents[2]/'data/checkshot_sonic_table.csv'
        
        dataframe_sonic_checkshot = pd.read_csv(self.directory)
        dataframe_sonic_checkshot = dataframe_sonic_checkshot[dataframe_sonic_checkshot['uwi']==uwi]
        return dataframe_sonic_checkshot

    def run(self, df_checkshot, df_sonic, std_sonic, std_checkshot, apply_covariance, inversion_start_depth, decimation_step, uwi):    
    #dataframe = pd.read_csv('td_tool/test_well_8.csv')

        #dataframe_sonic_checkshot = self.get_data(uwi)

        print(df_checkshot.columns)
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
        #total_wells = dataframe_sonic_checkshot.uwi.unique()
        #total_wells = [well.replace(' ','_').replace('/','_') for well in total_wells]
        #wells_not_loaded = [x for x in total_wells if x not in files_in_folder]

        # parameters
        depth_log_name = 'tvd_ss'
        velocity_log_name = 'vp'
        water_velocity = 1478

        ###################
        # END: PARAMETERS #
        ###################

        #dataframe = dataframe_sonic_checkshot
        df_sonic = df_sonic.rename(columns={'interval_velocity_sonic': 'vp'})

        df_checkshot = df_checkshot[['tvd_ss','time', 'average_velocity', 'interval_velocity', 'depth_source']].dropna(subset='time')
        water_velocity = float(df_checkshot[df_checkshot['depth_source']=='seabed from smda']['average_velocity'].iloc[0])   

        df_sonic = df_sonic[['tvd_ss','vp']].dropna(subset='vp')

        try:


            
            # water depths
            print('Reading water depth file ... ')
            #wd = dataframe

            #wd.WELL = [ww.replace(' ','_').replace('/','_') for ww in wd.WELL] # change formtting of well name
            print('...done.')


            # Loop over all wells in folder and perform bayesian check-shot correction

            ww = uwi
            ww = ww.replace(' ','_').replace('/','_')

            runWell = 1 # flag to skip well

            print('----')
            print('WELL: ' + ww)

            # read time-depth data
            td = df_checkshot

            if isinstance(td, pd.DataFrame): 

                # get data
                td_z = td['tvd_ss'].values
                td_t = td['time'].values
                td_t = td_t*0.001
            else:
                print('ERROR: empty time depth - skipping well')
                runWell = 0 # skip this well
                

            print('... done.')

            # get log data
            
            well_z = df_sonic['tvd_ss'].values.astype(float)
            well_vp = df_sonic['vp'].values.astype(float)

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

                water_depth = df_checkshot.loc[df_checkshot['depth_source'] == 'seabed from smda']['tvd_ss']
                   
                water_depth = float(water_depth.iloc[0])    
                water_twt = df_checkshot.loc[df_checkshot['depth_source'] == 'seabed from smda']['time']
                water_twt = float(water_twt.iloc[0])

                #ii = (td_z > water_depth)
                ##td_z = np.union1d([0, water_depth], td_z[ii])
                #td_t = np.union1d([0, water_twt], td_t[ii])
            
            # add top part of logs
            if extend2zero and runWell:
                print('EXTEND logs to z = 0')
                print('   #log: ' + str(len(well_z)))
                

                well_z, well_vp = extendLogsToZero(well_z, well_vp, water_depth, water_velocity)       
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
                par = getDefaultPar()        
                par['apply_corr'] = 0
                par['istep_bayes'] = decimation_step
                par['std_vp_mode'] = 2
                par['std_vp_const'] = std_sonic
                par['std_t_const'] = std_checkshot
                if apply_covariance == 'Apply':
                    par['apply_corr'] = 1
                elif apply_covariance == 'Do not apply':  
                    par['apply_corr'] = 0
                else:
                    par['apply_corr'] = 0
                    #these three were not activated
                par['corr_order'] = 1.8
                par['corr_range'] = 100

                par['zstart'] = inversion_start_depth             
                #par['zstart'] = well_z[1] # start at seabed
                 
                try:
                #if 1:

                    st.write(f'Calculating Bayesian time-depth correction for {uwi}... This might take some time.')

                    #dataframe_sonic_checkshot = self.get_data(uwi)


                    
                    class_bayes = Run_Bayesian()

                    well_vp, well_z, C = class_bayes.runCsc(well_z, well_vp, td_z, td_t, par)
                    
                    print('...done')
                    # merge output with existing dataframe
                    #well_vp = well_vp

                    df_bayes = pd.DataFrame({'TVDMSL': well_z,'VP_BAYES': well_vp})
                    
                    df_well = pd.merge(df_well, df_bayes, how = 'outer', on=['TVDMSL'])   
                    df_well = df_well.sort_values('TVDMSL').reset_index(drop = True)                
 
                    # plot well                
                    #
                    #fig = bayes_well_plot(df_well, td_z, td_t, ww, water_depth = water_depth, water_vel = water_velocity)
                    # save output data
                    bayes_csc_out = {'well_name': ww, 'td' : td, \
                                        'water_depth': water_depth, 'water_vel': water_velocity, \
                                        'df_well': df_well}
                    print("Bayesian Inversion could be applied")

                except Exception as e:
                    print(e.args)
                    print('ERROR: Bayesian step failed')
                
        except:
            print(f'Bayesian inference could not be applied for well')
            pass
        return df_well, td_z, td_t, ww, water_depth, water_velocity, C

#clas = Bayesian_Inference()
#df_well, td_z, td_t, ww, water_depth, water_velocity = clas.run(uwi='NO 34/7-22') #'NO 1/9-7 T3'
#print('finish')