# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:01:07 2020

@author: disch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage, misc
from IPython import embed
import sys
import os
sys.path.append(os.getcwd())
#import bayesian.td_tool.post_gauss.PostGauss as pg

from bayesian.td_tool.td_lib import getVel, getDrift
import plotly.express as px
import plotly.graph_objects as go


def mean_filter(val, N):
    
    nval = len(val)
    val_out = np.zeros(nval)
    
    for ii in np.arange(0, nval):
        
        # half length of average window
        min_edge_distance = np.min([ii, nval - 1 - ii])
        nn = np.min([N, min_edge_distance])
                        
        # set indexes for average window
        imin = ii - nn
        imax = ii + nn    
                
        # ensure idnex within range
        imin = np.max([0, imin])
        imax = np.min([nval, imax])
        
        # calculate average        
        val_out[ii] = np.mean(val[imin:imax + 1])
        
    return val_out
#    
#    
#    # padding
#    val = np.insert(val, 0, val[0] * np.ones((N-1,))) 
#    val = np.insert(val, -1, val[-1] * np.ones((N-1,))) 
#    
#    # convolution
#    val = np.arange(1,12); np.convolve(aa, np.ones((N,))/N, mode='valid')
#     
#    return val

    
def redatum(well_z, well_vp, td_z, td_t, zstart = 0):    
    
    # get well time
    well_t = getTime(well_z, well_vp)    
            
    # get corresponding start time
    ff = interp1d(well_z, well_t, kind = 'linear', bounds_error = False, fill_value = 0)    
    tstart = ff(zstart)        
        
    # get top part of well and td data
    itop_well = (well_z <= zstart)
    well_z_top = well_z[itop_well]
    well_vp_top = well_vp[itop_well]    
    
    itop_td = (td_z <= zstart)
    td_z_top = td_z[itop_td]
    td_t_top = td_t[itop_td]
    
    # select bottom part to use   
    ibot_well = (well_z > zstart)
    well_z = well_z[ibot_well]
    well_vp = well_vp[ibot_well]
    well_t = well_t[ibot_well]
            
    ibot_td = (td_z > zstart)
    td_z = td_z[ibot_td]
    td_t = td_t[ibot_td]
    
#    # insert exact start depth in well data
#    well_z = np.insert(well_z, 0, zstart)
#    well_vp = np.insert(well_vp, 0, well_vp[0])
    # perform redatuming
    well_z = well_z - zstart    
    td_z = td_z - zstart
    td_t = td_t - tstart
     
    # return output
    return well_z, well_vp, td_z, td_t, well_z_top, well_vp_top, td_z_top, td_t_top
    
class Run_Bayesian():
    def runCsc(self,well_z_in, well_vp_in, td_z_in, td_t_in, par):
        from bayesian.td_tool.td_lib import getVel, getDrift   
        from bayesian.td_tool.bayes_csc import getTime
        #run_2 = self.test(self, well_z_in, well_vp_in, td_z_in, td_t_in, par)
        ##########################
        # REDATUM to start depth #
        # SPLIT at start depth   #
        ##########################

        zstart = par['zstart']    
        well_z, well_vp, td_z, td_t, well_z_top, well_vp_top, td_z_top, td_t_top = redatum(well_z_in, well_vp_in, td_z_in, td_t_in, zstart)
        
        # derive time
        well_t = getTime(well_z, well_vp)
        #import ctypes
        #clibrary = ctypes.CDLL("clibrary.c")

   
        ##################################################
        # DECIMATE input logs to speed up bayesian step # 
        ##################################################
        
        # decimate according to bayes step
        istep_bayes = par['istep_bayes']    
        ibayes = np.arange(istep_bayes - 1, len(well_z), istep_bayes)    
        ibayes[-1] = len(well_z)-1 # ensure that last sample is included
        well_z_dec = well_z[ibayes]
        well_t_dec = well_t[ibayes]
        
        well_vp_dec = getVel(well_z_dec, well_t_dec)

        ###############
        # PRIOR model #        
        ###############
        
        # set model uncertainty
        if par['std_vp_mode'] == 1: # percentage
            well_vp_std_dec = well_vp_dec * par['std_vp_perc'] 
        elif par['std_vp_mode'] == 2: # constant velocity
            well_vp_std_dec = well_vp_dec * 0 + par['std_vp_const']
            
        # prior model expectation value
        mu_m = 1 / well_vp_dec
        
        # prior model standard deviation    
        #std_m = (1/2) * ((well_vp_dec - well_vp_std_dec)**(-1.0) - (well_vp_dec + well_vp_std_dec)**(-1.0))
        #Trick to calculate standard deviation for slowness
        s_max = 1/(well_vp_dec-well_vp_std_dec)
        s_min = 1/(well_vp_dec+well_vp_std_dec)

        std_m = (1/well_vp_dec)-s_min #standard deviation prior model for slowness

        
        # add spatial correlation
        dz_median = np.median(np.diff(well_z_dec))
        ncorr_range = par['corr_range'] / dz_median
        nc = len(std_m)
        if par['apply_corr']:
            C = corr_exp_mat(nc, ncorr_range, par['corr_order'])
        else:
            C = np.eye(nc)
        # prior covariance matrix
        Sigma_m = np.outer(std_m, std_m.T) * C

        ########
        # DATA #        
        ########

        # data is check-shot times
        d = td_t
        
        # set data uncertainty
        if par['std_t_mode'] == 1:
            std_e = td_t * par['std_t_perc']
        else:
            std_e = 0 * td_t + par['std_t_const']
    
        # testing....
        std_e_min = par['std_t_const']
        std_e_max = 10 * std_e_min
        rr = 500 # length (m) for which exponential is reduced to 10% of initial value
        std_e = std_e_min + (std_e_max - std_e_min) * np.exp(- np.log(10) * td_z / rr)       
    #     print(std_e[:10])
            
        # data covariance matrix
        Sigma_e = np.diag(std_e * std_e)

        #################
        # BAYESIAN step #
        #################
        
        # get G matrix  
        G = makeG(well_z_dec, td_z)

        # get solution
        mu_post,Sigma_post = PostGauss(G, d, Sigma_e, mu_m, Sigma_m)    

        #start_time_2 = time.time()
        #mu_post_cpp,Sigma_post_cpp = pg.PostGauss(G, d, Sigma_e, mu_m, Sigma_m)
        #end_time_2 = time.time()
        #print("Execution time CPP:", end_time_2 - start_time_2, "seconds")
        
        # derive posterior data
        well_vp_dec_post = 1 / mu_post
        sigma_post_d = np.diag(Sigma_post)
        # derive posterior covariance for velocity (from slowness):
        v_min = 1/(mu_post + sigma_post_d )
        v_max = 1/(mu_post - Sigma_post)

        std_vel = well_vp_dec_post - v_min

 
        

        ####################################
        # RESAMPLE to undecimated sampling #
        ####################################
        
        
        # resample back to original depth sampling via time curve    
        well_t_dec_post = getTime(well_z_dec, well_vp_dec_post)
        ff = interp1d(np.insert(well_z_dec, 0, 0), np.insert(well_t_dec_post, 0, 0), kind = 'linear', bounds_error = False, fill_value = np.nan)                          
        well_t_post = ff(well_z)

        #resample back standard deviation for velocity
        std_function = interp1d(np.insert(well_z_dec, 0, 0), np.insert(std_vel, 0, 0), kind = 'linear', bounds_error = False, fill_value = np.nan)
        
        

        
        # median filtering of time curve
        #well_t_post = ndimage.median_filter(well_t_post, size = 5)    
            
        # get output velocity from interpolated time
        well_vp_post = getVel(well_z, well_t_post)
        well_vp_diff = well_vp_post - well_vp

    #     print(well_vp_diff[:10])
        
        # final smoothing of velocity update
        if 1:
            #N = 50
            #well_vp_diff = np.convolve(well_vp_diff, np.ones((N,))/N, mode='same')        
            N = 25
            well_vp_diff = mean_filter(well_vp_diff, N)
            well_vp_post = well_vp + well_vp_diff
    #     print(well_vp_diff[:10])
                
        ###########################
        # REDATUM back to initial #
        ###########################
        
            
        # add start depth 
        well_z_out = well_z_in
        well_vp_out = np.insert(well_vp_post, 0, well_vp_top)
        std_function = interp1d(well_z_dec, std_vel, kind = 'linear', bounds_error = False, fill_value = np.nan)
        std_total_depth = std_function(well_z_out)
        print(std_total_depth)

    
        return well_vp_out,well_z_out, C, std_total_depth#,well_t_out
    
    
    #%%%%%%%%%%%%%%%%%%%%%
    #%- Resample output -%
    #%%%%%%%%%%%%%%%%%%%%%
    #
    #vp_b_diff=vp_b_post-vp_b;
    #if istep_bayes==1 % in resampling was applied
    #    vp_diff=vp_b_diff;
    #else % input was reampled before bayesian step
    #    
    #    %- interpolate velocity difference to original log sampling    
    #    vp_diff=interp1(zlog_b,vp_b_diff,zlog,'linear',0);    
    #end
    #
    #%- derive new velocity
    #vp_out=vp_in; % initiate output
    #vp_out(iuse)=vp_out(iuse)+vp_diff;
    #
    #%- derive other output parameters
    #tlog_out=[0; 2*cumsum(dzlog_in./vp_out(2:end))];
    #
    #%- drift
    #tlog_in_td=interp1(zlog_in,tlog_in,z_td,'linear',NaN);
    #tlog_out_td=interp1(zlog_in,tlog_out,z_td,'linear',NaN);
    #drift_in=t_td-tlog_in_td;
    #drift_out=t_td-tlog_out_td;
    #
    #%%%%%%%%%%%%%%%%%%%
    #%- Assign output -%
    #%%%%%%%%%%%%%%%%%%%
    #
    #%- return new velocity
    #bv.logs.vp=vp_out;
    #
    #%- return additional parameters
    #csc.depth_inv=zlog_b; % depth used in inversion
    #csc.vp_in=vp_in;
    #csc.vp_out=bv.logs.vp;
    #csc.vp_prior=vp_b;
    #csc.vp_post=vp_b_post;
    #csc.std_vp_in=interp1(zlog_b,std_vp_b,zlog_in,'linear',0);
    #csc.std_vp_out=interp1(zlog_b,std_vp_b_post,zlog_in,'linear',0);
    #csc.std_vp_prior=std_vp_b;
    #csc.std_vp_post=std_vp_b_post;
    #csc.t_in=tlog_in;
    #csc.t_out=tlog_out;
    #csc.dvp_out=csc.vp_out-csc.vp_in;
    #csc.dvp_post=csc.vp_post-csc.vp_prior;
    #csc.t_td=t_td;
    #csc.z_td=z_td;
    #csc.std_t_td=std_e;
    #csc.drift_in=drift_in;
    #csc.drift_out=drift_out;
    #bv.func.csc=csc;
    #
    #
    #%- return status
    #status=1;

#%--------------------------------------------------------------------------
#function G=makeG(zlog,z_td)
#%--------------------------------------------------------------------------
#
#% ensure row 
#zlog=zlog(:)';
#z_td=z_td(:)';
#
#G=zeros(numel(z_td),numel(zlog));
#dz=diff([0 zlog]); % Depth step
#
#for i=1:numel(z_td)   
#    k=find(zlog<z_td(i),1,'last');
#    G(i,1)=2*z_td(1);
#    G(i,1:k)=2*dz(1:k);
#    G(i,min(k+1,numel(zlog)))=2*(z_td(i)-zlog(k));        
#end

    
def makeG(well_z, td_z):
    
    # get input dimensions
    nwell = len(well_z)
    ntd = len(td_z)

    # initiate matrix
    G = np.zeros((ntd, nwell))
    # depth step
    well_dz = np.diff(np.insert(well_z, 0, 0)) # assumes well_z starts depth > 0, inserts zero depth
    # loop through time-depth data
    for ii in range(ntd): # loop over time-depth data
        
        G[ii, 0] = 2 * td_z[0] # default        
        dz = well_dz[well_z < td_z[ii]]
        G[ii,:len(dz)] = 2 * dz
        G[ii,np.min((len(dz), nwell-1))] = 2 * (td_z[ii] - np.sum(dz))

    return G



def corr_exp_mat(n, d, nu):
    
    tau = np.arange(0, n)
    rho_s = np.exp(-3 * (tau / d)**nu)
    
    A = np.tile(tau, [n, 1])
    itau = np.abs(A - np.transpose(A))
    C = rho_s[itau]
    
    return C    

def getDefaultPar():
    
    # create dictionary with parameters
    par = {}
    
    # log parameters
    par['std_vp_mode'] = 1 # 1 = percentage, 2 = constant
    par['std_vp_perc'] = .5 # percentage (fraction)
    par['std_vp_const'] = 500 # m/s 
    
    # spatial correlation      
    par['apply_corr'] = 1
    par['corr_order'] = 1.8
    par['corr_range'] = np.finfo(float).eps
    
    # other parameters
    par['istep_bayes'] = 10 # sampling step used to speed up bayesian part        
    par['zstart'] = 0 # start depth for bayesian inversion
    
    # check-shot parameters        
    par['std_t_mode'] = 2 # 1 = percentage, 2 = constant
    par['std_t_perc'] = .05 # percentage (fraction)
    par['std_t_const'] = 0.005 # s 
    
    
    return par

def getVel(zz, tt): 
    
    # assumes first depth is > 0
        
    # calculate differences
    dz = np.diff(zz)    
    dt = np.diff(tt)
    
    # add first time/depth pair (assumes it starts from 0)

    dz = np.insert(dz, 0, zz[0])
    dt = np.insert(dt, 0, tt[0])
    dt = np.where(dt == 0, np.nan, dt)
    #ii = np.where(dtwt != 0)        

    # get velocity
    vp = 2 * dz / dt
    
    # add top value to ensure same dimension
    #vp = np.insert(vp, 0, vp[0])
    
    # return velocity
    return vp

def getTime(z, vp):
    
    # assumes first depth is > zero

    dz = np.diff(z) # depth step in meters
    
    dz = np.insert(dz, 0, z[0])

    dt = 2 * dz / vp # velocity in m/s, time in s
    t = np.cumsum(dt)    

    
#    if z[0] == 0 # check if first element is zero        
#    
#        dz = np.diff(z) # depth in meters
#        dt = 2 * dz / vp[:-1] # velocity in m/s, time in s
#        t = np.cumsum(dt)
#        t = np.insert(t, 0, 0)
#        
#    else: # first element is non zero
#        
#        dz = np.diff(z) # depth in meters
#        dz = np.insert(dz, 0, z[0])
#        dt = 2 * dz / vp[:-1] # velocity in m/s, time in s
#        t = np.cumsum(dt)                
    
    return t

def PostGauss(G,d,Sigma_e,mu_m,Sigma_m):
    
    #FUNCTION PostGauss: Compute posterior mean and covarianece
    #
    #Input: 
    #   G        : Linear forward model operator
    #   d        : Data
    #   Sigma_e  : Noise
    #   mu_m     : Prior model
    #   Sigma_m  : Prior model covariance
    #
    #Output
    #   mu_post    : Posterior mean
    #   Sigma_post : Posterior covarince
    #   
    #Programmed by Arild Buland
    #Ported to python, Eirik Dischler, 2020
    
    Sigma_d = G.dot(Sigma_m).dot(G.T) + Sigma_e

    m = np.linalg.inv(Sigma_d).dot(d - G.dot(mu_m))

    mu_post = mu_m + Sigma_m.dot(G.T).dot(m)

    N = np.linalg.inv(Sigma_d).dot(G.dot(Sigma_m))

    Sigma_post = Sigma_m - Sigma_m.dot(G.T).dot(N)


    return mu_post, Sigma_post

#def bayes_well_plot_old(well_z, well_vp_in, well_vp, td_z, td_t, well_s, water_depth = np.nan, water_vel = 1480):

def bayes_well_plot(df_well, td_z, td_t, well_s, water_depth = np.nan, water_vel = 1480):
    # extract columns from dataframe
    well_z = df_well['TVDMSL'].values
    #well_vp_in = df_well['VP_IN'].values
    well_vp_ext = df_well['VP_EXT'].values
    well_vp = df_well['VP_BAYES'].values
    
    # derive data

    td_vp = getVel(td_z, td_t)

    td_df = pd.DataFrame({'depth':td_z,'twt':td_t,'vp':td_vp})
    td_df = td_df.dropna()
    td_z = td_df['depth'].values
    td_t = td_df['twt'].values
    td_vp = td_df['vp'].values


    well_t_ext = getTime(well_z, well_vp_ext)
    well_t = getTime(well_z, well_vp)
    td_drift_t_ext, td_drift_z_ext, well_drift_t_ext, well_drift_z_ext = getDrift(td_z, td_t, well_z, well_t_ext)    
    td_drift_t, td_drift_z, well_drift_t, well_drift_z = getDrift(td_z, td_t, well_z, well_t)    

    
    fig = plt.figure('Time-Depth QC-plot')
    fig.clf()
    plt.gcf().set_size_inches(14, 8)        
    ax1 = plt.subplot(151)
    ax2 = plt.subplot(152, sharey = ax1)     
    ax3 = plt.subplot(153, sharey = ax1)    
    ax4 = plt.subplot(154, sharey = ax1)   
    ax5 = plt.subplot(155, sharey = ax1)   
    yr = [np.max(np.union1d(well_z, td_z)) * 1.1, 0]

    # velocity plot        
    plt.sca(ax1)
    ax1.invert_yaxis()
    plt.plot([-1e4, 1e4], [water_depth, water_depth], '--k')
    plt.plot([water_vel, water_vel], [-1e4, 1e4], ':g')    
    plt.plot(well_vp_ext, well_z, '-g')
    plt.step(td_vp, td_z, '-r', where = 'post')
    plt.grid()
    plt.xlabel('m/s')
    plt.ylabel('TVDMSL')
    plt.title(well_s + '\n VELOCITY')
    plt.xlim(0, np.max(np.union1d(well_vp, td_vp)) * 1.2)
    plt.ylim(yr)
    
    # velocity difference
    plt.sca(ax2)
    ax1.invert_yaxis()
    plt.plot([-1e4, 1e4], [water_depth, water_depth], '--k')
    plt.plot([0, 0], [0, 1e4], '-k')    
    xx = well_vp - well_vp_ext
    plt.plot(xx, well_z, '-b')
    plt.grid()
    plt.xlabel('m/s')    
    plt.title('VELOCITY DIFFERENCE') 
    xmin = np.min(xx) - 10
    xmax = np.max(xx) + 10
    plt.xlim([xmin, xmax])
    plt.ylim(yr)    
    
    
    # time plot    
    plt.sca(ax3)        
    plt.plot([-1e4, 1e4], [water_depth, water_depth], '--k')
    plt.plot([0, 0], [0, np.max(td_z)], '-k')    
    
     
    plt.plot(1e3 * well_t_ext, well_z, '-g')
    plt.plot(1e3 * well_t, well_z, '-b')
    plt.plot(1e3 * td_t, td_z, '.r')   
    
    plt.grid()
    plt.xlabel('ms')
    plt.title('TWT')
    plt.xlim(-100, np.max(1e3 * well_t) * 1.1)
    plt.ylim(yr)
    
    # drift plot
    plt.sca(ax4)
    plt.plot([-1e4, 1e4], [water_depth, water_depth], '--k')
    plt.plot([0, 0], [0, np.max(td_z)], '-k')     
    plt.plot(1e3 * td_drift_t_ext, td_z, '-g')
    plt.plot(1e3 * td_drift_t, td_z, '-b')                   
    plt.grid()        
    plt.xlabel('ms')
    
    #well_drift_t = well_drift_t[~np.isnan(well_drift_t)]    
    xx = np.insert(well_drift_t, 0, well_drift_t_ext)
    xx = xx[~np.isnan(xx)]    
    plt.xlim(1e3 * np.min(xx) - 3, 1e3 * np.max(xx) + 3)    
    plt.title('DRIFT (OWT)')    
    plt.ylim(yr)
    
    # output velocity only
    plt.sca(ax5)
    ax1.invert_yaxis()
    plt.plot([-1e4, 1e4], [water_depth, water_depth], '--k')
    plt.plot([water_vel, water_vel], [-1e4, 1e4], ':g')    
    plt.plot(well_vp, well_z, '-b')
    plt.step(td_vp, td_z, '-r', where = 'post')
    plt.grid()
    plt.xlabel('m/s')
    plt.title('VELOCITY BAYESIAN')
    xmin = 1000
    xmax = np.min((np.max(np.union1d(well_vp, td_vp)) * 1.2, 6000))
    plt.xlim(xmin, xmax)
    plt.ylim(yr)    
    
    return fig
    
    
#    ax = plt.subplot(1, ncol, isub); isub = isub + 1    
#    ax.cla()
#    plt.plot(well_vp, well_t, '-b')    
#    
#    
#    # time drift
#    ax = plt.subplot(1, ncol, isub); isub = isub + 1
#    ax.cla()
#    #plt.plot(well_vp, well_z, '-b')        
