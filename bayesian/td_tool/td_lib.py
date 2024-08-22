# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:23:51 2020

@author: eidi
"""

from collections import namedtuple
import time
import numpy as np
import pandas as pd
import bruges as br
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
#import synthmod1d as smod
from PIL import Image


#import holoviews as hv
#import holoviews.plotting.mpl
#from holoviews.operation.datashader import datashade
#from colorcet import fire


#import holoviews as hv
#from holoviews.element.tiles import EsriImagery
#from holoviews.operation.datashader import datashade

#import datashader as ds
#from datashader import transfer_functions as tf
#import bokeh.plotting
#from bokeh.plotting import figure, output_file, show
#import seaborn as sns

#hv.extension('bokeh')
#import panel as pn
#pn.extension()



#print('')
#print('START')

#root_folder = 'C:/Users/eidi.STATOIL-NET/OneDrive - Equinor/Python/src/bravos/'
#root_folder = 'D:/Users/disch/OneDrive/--Statoil--/python/c-drive-work/bravos/'
#td_file_folder = root_folder + 'INPUT/'
#td_file = td_file_folder + 'TIME_CKS_QUAD_1_to_36.dat'
#input_pkl_folder = root_folder + 'INPUT/PKL/'
#cwn0 = 'NO_16/1-29_ST2'

def extendLogsToZero(well_z, well_vp, water_depth, water_vel):
    
    # get time
    well_t = get_time(well_z, well_vp)
    
    # remove any logs values within water layer
    ii = (well_z > water_depth)
    well_z = well_z[ii]
    well_vp = well_vp[ii]
    well_t = well_t[ii]
            
#    # make water layer
#    water_twt = 2 * water_depth / water_vel
#    well_water_z = [0, water_depth]
#    well_water_vp = [water_vel, water_vel]
    
    
    ###############################    
    # velocity trend below seabed #
    ###############################    
    
    # parameters for depth trend
    dt = 0.001 # time step for velocity trend
    tmax = 1e4
    tt = np.arange(0, tmax, dt)
    v_sb = 1700 # velocity at seafloor (1800 in paper)
    v_inf = 5000 # velocity at infinite depth (max)
    kk = .5 # s^-1 # exponential rate of velocity increase
    exp0 = -kk * tt * v_inf / (v_inf - v_sb)
    v_trend = v_sb * v_inf / (v_sb + (v_inf - v_sb) * np.exp(exp0))
    
    # convert from time to depth
    dz_trend = v_trend * dt
    zz_trend = np.cumsum(dz_trend) + water_depth
        
    # keep part between seabed and first log depth
    jj = (zz_trend < well_z[0])
    #print(jj)
    well_trend_z = zz_trend[jj]
    well_trend_vp = v_trend[jj]
    
    # scale velocities to match log velocities
    
    # resample in depth to log sampling
    
    ###################
    # merge all parts #
    ###################
    
    well_z = np.concatenate(([0, water_depth], well_trend_z, well_z))
    well_vp = np.concatenate(([water_vel, water_vel], well_trend_vp, well_vp))
    
    # return result    
    return well_z, well_vp 

def displayWell(zz, vv, tt = 0):
    
    
    plt.figure('TD from well')
    nrow = 1
    ncol = 2
    isub = 1
    zmax = max(zz) * 1.1
    
    plt.subplot(nrow, ncol, isub)
    isub = isub + 1
    plt.plot(vv, zz, '-b')
    plt.ylim(zmax, 0)
    plt.grid()
    
    plt.subplot(nrow, ncol, isub)
    isub = isub + 1
    plt.plot(tt, zz, '-b')
    plt.ylim(zmax, 0)    
    plt.grid()

def applyTimeShift(well_z, well_t, td_z, td_t):
    
    
    # merge at first td-value after first well data            
    td_z0 = td_z[td_z >= well_z[0]][0]
    td_t0 = td_t[td_z >= well_z[0]][0]    
    well_t0 = smod.getTimeFromTD(td_z0, well_z, well_t) # time at first log value            
    
    # special case, no check-shots within log range
    if np.isnan(well_t0):

        # merge at start of well data 
        t_shift = smod.getTimeFromTD(well_z[0], td_z, td_t) # time at first log valuetd_drift
        
    t_shift = td_t0 - well_t0    
    
    # apply shift
    well_t = well_t + t_shift
    
    # return
    return well_t

def extendTDwithLogs(td_z, td_t, well_z, well_vp, dz = 0):

    z_ext = []
    t_ext = []
    
    # check input
    if (len(td_z) == 0) or (len(well_z) == 0):
        return z_ext, t_ext
    
    # check that well extends deeper than td-data
    if td_z[-1] < well_z[-1]:
    
        # check that there is some overlap (well starts before last td-depth)
        ii = (well_z <= td_z[-1])
        if any(ii):
            
            # remove part of well shallower than last td-value
            well_z = well_z[ii == False]            
            well_vp = well_vp[ii == False]            
            
            # get time
            z0 = td_z[-1]
            t0 = td_t[-1] 
            zz = np.insert(well_z, 0, z0) # add start depth of first interval
            vp = np.insert(well_vp, 0, well_vp[0]) 
            tt = td_lib.get_time(zz, vp) # get time from z/vp                                        
            tt = tt + t0 # add start time
            
            # resample
            print(dz)
            if dz > 0:
                                    
                zz_new = np.arange(zz[0], zz[-1], dz)
                zz_new = np.union1d(zz_new, zz[-1]) # ensure laste depth sample is included                
                ff = interp1d(zz, tt, kind = 'linear', bounds_error = False, fill_value = np.nan)
                tt = ff(zz_new)                
                zz = zz_new
                

            # (remove first sample since it equals last td sample)
            t_ext = tt[1:]
            z_ext = zz[1:]
        
    
    # return output time-depth values 
    return z_ext, t_ext
    
def strictlyIncreasing(vv):
    
    
        
    ii = np.where(np.diff(vv) <= 0)
    if len(ii[0]) == 0:
        ilast = len(vv) - 1 # last valid integer
    else:
        ilast = ii[0][0]
        
    vv = vv[:(ilast + 1)]        
    
    return vv, ilast

def get_time(z, vp):
    
    dz = np.diff(z) # depth in meters
    dt = 2 * dz / vp[:-1] # velocity in m/s, time in s
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    
    return t

def getDrift(td_z, td_t, well_z, well_t):
    
    # get well times at each check-shot depth
    ff = interp1d(well_z, well_t, kind = 'linear', bounds_error = False, fill_value = np.nan)
    td_well_t = ff(td_z)
    
    # drift (TWT --> OWT)
    td_drift_t = .5 * (td_well_t - td_t)
    
    # get td times at each well depth
    ff = interp1d(td_z, td_t, kind = 'linear', bounds_error = False, fill_value = np.nan)
    well_td_t = ff(well_z)
    
    # drift (TWT --> OWT)
    well_drift_t = .5 * (well_t - well_td_t)    

    # fix NaN
    #ii = ~np.isnan(td_drift_t)
    ii = np.arange(len(td_z))
    td_drift_t = td_drift_t[ii]
    td_drift_z = td_z[ii]
    
    #ii = ~np.isnan(well_drift_t)
    ii = np.arange(len(well_z))
    well_drift_t = well_drift_t[ii]
    well_drift_z = well_z[ii]
    
    
    return td_drift_t, td_drift_z, well_drift_t, well_drift_z

def crossplot(df, x_curve_name = "LFP_PHIT", y_curve_name = "LFP_SWT"):

#    x_range=(df[x_curve_name].quantile(q=0.02), df[x_curve_name].quantile(q=0.98))
#    y_range=(df[y_curve_name].quantile(q=0.02), df[y_curve_name].quantile(q=0.98))
    x_range=(df[x_curve_name].quantile(q=0), df[x_curve_name].quantile(q=0.98))
    y_range=(df[y_curve_name].quantile(q=0), df[y_curve_name].quantile(q=1))
    
    x_range = (-4000, 8000)
    x_range = (-4000, 10000)
    y_range = (0, 6000)
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=900, plot_width=900)
    agg = cvs.points(df, x_curve_name, y_curve_name)
    res = tf.set_background(tf.shade(agg, cmap=fire),"black")
    
    ds.transfer_functions.Image.to_pil(res,'test')
    
    
    #renderer = hv.Store.renderers['matplotlib'].instance(fig='svg', holomap='gif')
    #renderer.save(res, 'test200127')
    #agg = cvs.points(df, x_curve_name, y_curve_name, 'LFP_VSH')
    #res = tf.set_background(tf.shade(agg, cmap=['fire', 'lightblue']),"black")    
    #res = tf.shade(agg, cmap=['fire', 'lightblue'])#,'black')    
    #print(res.shape)
    display(res)
    
    return res

def removeNegTimeDepth(td):
    
    bb = ' ' * 4
    
    print('Remove negative depths...') 
    
    wells_neg_depth = td[td.TVDSS < 0]['WELL'].unique()
    jj = td[td.TVDSS < 0].index
    td = td.drop(jj)    
    print(bb + 'Wells with negative depths: ' + str(len(wells_neg_depth)))
    print(bb + 'Rows removed: ' + str(len(jj)))
    
    print('Remove negative time...')
    wells_in = td['WELL'].unique()
    wells_neg_time = td[td.TWT < 0]['WELL'].unique()
    jj = td[td.TWT < 0].index
    td = td.drop(jj)    
    wells_out = td['WELL'].unique()    
    wells_removed = list(set(wells_out) - set(wells_in))
        
    print(bb + 'Wells with negative times: ' + str(len(wells_neg_time)))          
    if len(wells_neg_time) > 0:
        print(bb + bb + ', '.join(np.sort(wells_neg_time)))                
    print(bb + 'Rows removed: ' + str(len(jj)))
    if wells_removed:
        print(bb + '#Wells removed: ' + str(len(wells_removed)))                    
          
    
    return td #, wells_neg_depth, wells_neg_time

def removeZeroTimeDepth(td):
    
    
    print('Remove zero depths...') 
    jj = td[td.TVDSS == 0].index
    td = td.drop(jj)    
    
    print('Remove zero time...')
    jj = td[td.TWT == 0].index
    td = td.drop(jj)    
    
    return td

def addZeroDepth(td):
    
    return td
    


def setTopBase(td):
    

            
    print('Set top/base depth and time...')
    
    # get unique wells
    wu = td['WELL'].unique()
    
    # initiate columns
    nrow, ncol = td.shape
    td['Z_TOP'] = np.zeros(nrow)
    td['Z_BASE'] = np.zeros(nrow)
    td['Z_MID'] = np.zeros(nrow)
    td['DZ'] = np.zeros(nrow)
    td['TWT_TOP'] = np.zeros(nrow)
    td['TWT_BASE'] = np.zeros(nrow)
    td['TWT_MID'] = np.zeros(nrow)
    td['DTWT'] = np.zeros(nrow)    
    
    
    #- loop through wells and calculate interval velocities
    for w0 in wu:

        #- get well row index
        jj = td[td['WELL'] == w0].index
        
        #- get base
        z_base = td.loc[jj]['TVDSS'].values
        twt_base = td.loc[jj]['TWT'].values
                    
        #- create top depth
        z_top = np.zeros(len(z_base))
        z_top[1:] = z_base[:-1]
        twt_top = np.zeros(len(twt_base))
        twt_top[1:] = twt_base[:-1]
        
        
        #- thickness
        dz = z_base - z_top
        dtwt = twt_base - twt_top
        
        #- midpoint
        z_mid = 0.5 * (z_base + z_top)
        twt_mid = 0.5 * (twt_base + twt_top)
        
        #- assign values
        td.loc[jj, 'Z_TOP'] = z_top
        td.loc[jj, 'Z_BASE'] = z_base
        td.loc[jj, 'Z_MID'] = z_mid
        td.loc[jj, 'DZ'] = dz
        td.loc[jj, 'TWT_TOP'] = twt_top
        td.loc[jj, 'TWT_BASE'] = twt_base
        td.loc[jj, 'TWT_MID'] = twt_mid
        td.loc[jj, 'DTWT'] = dtwt
        
    
    return td

    tt1 = time.time()
    etime = tt1 - tt0    
    print(etime)


def ensureIncreasing(td):
    
    tt0 = time.time()    
    
    bb = ' ' * 4
    
    print('Ensure increasing time/depth...')        
    
    # get unique wells
    wu = td['WELL'].unique()
    
    # initiate columns
    nrow, ncol = td.shape
    td['keep_flag'] = np.ones(nrow)
    
    
    #- loop through wells and calculate interval velocities
    for w0 in wu:

        #- get well row index
        jj = td[td['WELL'] == w0].index
        
        #- get time and depth
        zz = td.loc[jj]['TVDSS'].values
        tt = td.loc[jj]['TWT'].values
        
        #- get step
        dz = np.diff(zz)
        dt = np.diff(tt)
        
        #- get index of first element <= 0
        kk = np.where((dt <= 0) | (dz <= 0))        
        
        #- only use portion of ddata up to this element
        if len(kk[0]) > 0:
                        
            k0 = kk[0][0] # index of first elements where time or depth is not strictly increasing                        
            td.loc[jj[(k0+1):], 'keep_flag'] = 0
    
    
    #- remove all rows not marked with keep
    wells_in = td['WELL'].unique()    
    jj = td[td.keep_flag == 0].index
    wells_in_selection = td[td.keep_flag == 0]['WELL'].unique()
    td = td.drop(jj) 
    wells_out = td['WELL'].unique()
    wells_removed = list(set(wells_out) - set(wells_in))
    
    #- print info
    print(bb + 'Wells affected: ' + str(len(wells_in_selection)))
    print(bb + 'Rows removed: ' + str(len(jj)))
    if wells_removed:
        print(bb + '#Wells removed: ' + str(len(wells_removed)))       

    #- print time
    tt1 = time.time()
    etime = tt1 - tt0    
    print(bb + 'Time elapsed (s): ' + str(etime))
    
    return td
    
        

def setIntVel(td):
    
    print('Set interval velocity')
    
    # initiate columns
    nrow, ncol = td.shape
    v_int = np.zeros(nrow)
    
    # calculate
    dz = td['DZ'].values
    dtwt = td['DTWT'].values    
    ii = np.where(dtwt != 0)    
    v_int[ii] = 2 * dz[ii] / dtwt[ii]
    
    # assign
    td['V_INT'] = v_int
    
    return td

def getVel(zz, tt):    
        
    # calculate
    dzz = np.diff(zz)    
    dtt = np.diff(tt)
    #ii = np.where(dtwt != 0)        
    
    # get velocity
    vp = 2 * dzz / dtt
    
    # add top value to ensure same dimension
    vp = np.insert(vp, 0, vp[0])
    
    # return velocity
    return vp

def getStat(td):
    
    df = pd.DataFrame
    
    
    
    

def filterTDdata(td):
    
    # create error column
    td['error'] = np.zeros(len(td['WELL']))
    
    # get unique wells
    wu = td['WELL'].unique()
    
    #- loop through wells and detect errors
    for w0 in wu:
                
        #- get  well check-shots
        tt = td[td['WELL'] == w0]['TWT'].values
        zz = td[td['WELL'] == w0]['TVDSS'].values
        
        #- detect negative values
        if any(tt) < 0:
            
            # assign values
            td.loc[td['WELL'] == w0, 'error'] = 1
            
        #else if any(zz) < 0:
            
            
            
            
            
        
        print(w0)
        
        
    return td

def scatterPlot_old(x, y, fig_name = 'test', xlim = (-1,1), ylim = (-1,1), nbin = 100):
    
    # create figure
    fig = plt.figure(fig_name,figsize = (10,14))
    fig.clf()    
    
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    # size of axes
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    
    ax_scatter = plt.axes(rect_scatter)    
    ax_scatter.tick_params(direction='in', top=True, right=True)
    
    #sns.scatterplot(x = x_s, y = y_s, data = td, alpha = 0.5, legend = False)
    
    #sns.scatterplot(x , y, alpha = 0.5, legend = False)
    sns.scatterplot(x , y, alpha = 0.5, legend = False)
    #sns.scatterplot(x = x_s, y = y_s, hue = 'WELL', data = td, alpha = 0.5, legend = False)
    plt.plot([-1e4, 1e4], [0, 0], '-k', linewidth = 2)
    plt.plot([0, 0], [0, 1e4], '-k', linewidth = 2)
    
    #ax.invert_yaxis()
    plt.grid()        
    
    ax_histx = plt.axes(rect_histx)
    plt.grid()
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    plt.grid()
    ax_histy.tick_params(direction='in', labelleft=False)
    
    # the scatter plot:
    
    #ax_scatter.scatter(x, y)
    
    # axes limits
    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    
    # bin size    
    xbinwidth = np.abs(xlim[1] - xlim[0]) / nbin
    ybinwidth = np.abs(ylim[1] - ylim[0]) / nbin
        
    # define bins        
    xbins = np.arange(min(xlim), max(xlim) + xbinwidth, xbinwidth)
    ybins = np.arange(min(ylim), max(ylim) + ybinwidth, ybinwidth)
    
    # create histograms    
    ax_histx.hist(x, bins = xbins)
    
    ax_histy.hist(y, bins = ybins, orientation='horizontal')
    ax_histy.invert_yaxis()
    #plt.grid()        

    
    # set histogram axes
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    
    
    plt.show()    
    
def scatterPlot(td, x_s, y_s, fig_name = 'test', xlim = (-1,1), ylim = (-1,1), nbin = 100):
    
    x = td[x_s].values
    y = td[y_s].values
    
    # create figure
    fig = plt.figure(fig_name,figsize = (10,14))
    fig.clf()    
    
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    # size of axes
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    
    ax_scatter = plt.axes(rect_scatter)    
    ax_scatter.tick_params(direction='in', top=True, right=True)
    
    #sns.scatterplot(x = x_s, y = y_s, data = td, alpha = 0.5, legend = False)
    
    #sns.scatterplot(x , y, alpha = 0.5, legend = False)
    
    sns.scatterplot(x = x_s, y = y_s, hue = 'WELL', data = td, alpha = 0.5, legend = False)
    plt.plot([-1e4, 1e4], [0, 0], '-k', linewidth = 2)
    plt.plot([0, 0], [0, 1e4], '-k', linewidth = 2)
    
    #ax.invert_yaxis()
    plt.grid()        
    
    ax_histx = plt.axes(rect_histx)
    plt.grid()
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    plt.grid()
    ax_histy.tick_params(direction='in', labelleft=False)
    
    # the scatter plot:
    
    #ax_scatter.scatter(x, y)
    
    # axes limits
    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    
    # bin size    
    xbinwidth = np.abs(xlim[1] - xlim[0]) / nbin
    ybinwidth = np.abs(ylim[1] - ylim[0]) / nbin
        
    # define bins        
    xbins = np.arange(min(xlim), max(xlim) + xbinwidth, xbinwidth)
    ybins = np.arange(min(ylim), max(ylim) + ybinwidth, ybinwidth)
    
    # create histograms    
    ax_histx.hist(x, bins = xbins)
    
    ax_histy.hist(y, bins = ybins, orientation='horizontal')
    ax_histy.invert_yaxis()
    #plt.grid()        

    
    # set histogram axes
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    
    
    plt.show()        



        
#td = smod.getTDfromPetrelFileMulti(td_file)

    

    
    
    
    
    

def preProcMain(td):
    #td = td0.copy()
    
    #nn = 20000
    #td = td.iloc[nn:(2 * nn),:]
    
    #td = filterTDdata(td)
    print('Input size...')
    print(td.shape)
    
    #- remove rows with NaN
    #td = removeNaNrows(td)
    #print(td.shape)
    
    #- remove negative time/depth
    td = removeNegTimeDepth(td)
    print(td.shape)
    
    #- remove zero time/depth
    td = removeZeroTimeDepth(td)
    print(td.shape)
    
    #- ensure strictly increasing
    td = ensureIncreasing(td)
    print(td.shape)
    
    #- add various depth columns
    td = setTopBase(td)
    print(td.shape)
    
    #- get interval velocity
    td = setIntVel(td)
    print(td.shape)
    
    return td

if 0:
    # TIME VS: DEPTH plot
    nbin = 100
    n1 = 30000
    nn = 10000  
    n2 = n1 + nn
    
    ii = np.arange(n1, n2)
    x_s = 'TWT'
    y_s = 'TVDSS'
    xx = td[x_s].values[ii]
    yy = td[y_s].values[ii]
    wells_unique = td.loc[ii,'WELL'].unique()
    print('UNIQUE WELLS: ' + str(len(wells_unique)))
    print(wells_unique)
    #print('   ' + ', '.join(wells_unique))
    #print('    ' + ', '.join(list(wells_unique)))
    xlim = (min(xx), max(xx))
    ylim = (max(yy), min(yy))
    #scatterPlot(xx, yy, fig_name = 'time_vs_depth', xlim = xlim, ylim = ylim, nbin = nbin)
    scatterPlot(td, x_s, y_s, fig_name = 'time_vs_depth', xlim = xlim, ylim = ylim, nbin = nbin)
    
    
    
    # INT VEL plot
    x_s = 'V_INT'
    y_s = 'Z_MID'
    xx = td[x_s].values[ii]
    yy = td[y_s].values[ii]
    xlim = (-4e3, 1e4)
    ylim = (6000, 0)
    scatterPlot(xx, yy, fig_name = 'int_vel_vs_depth', xlim = xlim, ylim = ylim)
    
    #crossplot(td, x_curve_name = 'V_INT', y_curve_name = 'Z_MID')
    
    
            
if 0:
    #- get time-deph of specific well
    td = smod.getTDfromPKL(td_file_folder, cwn0)
    
    
    #- get well log data
    w0 = smp.cwn0[3:].replace('/','_')
    wlog_pkl_file = input_pkl_folder + w0 + '.pkl'
    wlog0 = smod.getWell(wlog_pkl_file, cwn0) 

if 0:    
    td = smod.getTDfromPetrelFileMulti(td_file)
    
if 0:
    print(td.head())
    #print(wlog0.columns)
    #print(wlog0.head())
    
    nrow = 2
    ncol = 2
    iax = 1
    
    fig = plt.figure('test')
    fig.clf()
    ax = plt.subplot(nrow, ncol, iax)
    iax = iax + 1
    #tips = sns.load_dataset("tips")
    plt.plot([0, 0], [0, 1e4], '-k', linewidth = 2)
    plt.plot([-10, 10], [0, 0], '-k', linewidth = 2)
    #sns.scatterplot(x = 'TWT', y = 'TVDSS', data = td, alpha = 0.5)
    x_s = 'TWT'
    y_s = 'TVDSS'
    xx = td[x_s].values
    yy = td[y_s].values    
    sns.scatterplot(x = x_s, y = y_s, hue = 'WELL', data = td, alpha = 0.5, legend = False)
    
    
    pp = 1.1
    plt.xlim(pp * min(xx), pp * max(xx))    
    plt.ylim(pp * min(yy), pp * max(yy))
    
    ax.invert_yaxis()
    plt.grid()    
    plt.xlabel('TWT (s)')
    plt.ylabel('TVDSS (m)')
    
    ax = plt.subplot(nrow, ncol, iax)
    iax = iax + 1
    plt.plot([0, 0], [0, 1e4], '-k', linewidth = 2)
    plt.plot([-10, 10], [0, 0], '-k', linewidth = 2)
    x_s = 'V_INT'
    y_s = 'Z_MID'
    xx = td[x_s].values
    yy = td[y_s].values
    sns.scatterplot(x = x_s, y = y_s, hue = 'WELL', data = td, alpha = 0.5, legend = False)
    
    
    ax = plt.subplot(nrow, ncol, iax)
    iax = iax + 1
    
    
    #nn = 1e3
    #sns.jointplot(x = x_s, y = y_s, data = td, xlim = (-4000, 10000))
    
        
    plt.xlim(pp * min(xx), pp * max(xx))
    plt.xlim(-4000,10000)
    plt.ylim(pp * min(yy), pp * max(yy))
    
    ax.invert_yaxis()
    plt.grid()    
    plt.xlabel(x_s)
    plt.ylabel(y_s)    
    
    #sns.set_style("white")
    #sns.kdeplot(td['TWT'], td['TVDSS'])
    
    


if 0:
    res = crossplot(td, x_curve_name = 'TWT', y_curve_name = 'TVDSS')
    im = res.to_pil()
    im.save('test.png')
    
    
    
    im = Image.open('test.png')
    
    
    
    fig = plt.figure('test')
    fig.clf()
    ax = plt.subplot(1,1,1)
    
    #xx = res.TVD
    xv = res.TWT.values
    yv = res.TVDSS.values
    xv[0]
    print(xv[0], xv[-1])
    MM = res.T.values
    vmin = np.min(MM)
    vmax = np.max(MM)
    vmax = vmin + 0.5*(vmax - vmin)
    #plt.imshow(MM, aspect='auto', cmap = 'fire', vmin = vmin, vmax = vmax, extent = [xv[0], xv[-1], yv[-1], yv[0]])
    #plt.imshow(, aspect='auto', cmap = 'fire', extent = [xv[0], xv[-1], yv[-1], yv[0]])
    plt.imshow(im, aspect='auto', cmap = 'fire', extent = [xv[0], xv[-1], yv[-1], yv[0]])
    plt.xlabel('TWT (s)')
    plt.ylabel('TVDSS (m)')
    #plt.imshow(res.TVDSS.values, cmap = 'fire')



#print('END')
