from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import math
import os
sys.path.append(os.getcwd())
import plotly.express as px
from IPython import embed
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
import git
import plotly.graph_objects as go
from bayesian.td_tool.runBayesDixInv import Bayesian_Inference
from bayesian.td_tool.bayes_csc import getTime, getDrift
from bayesian.td_tool.td_lib import getVel, resample_tvdss_md
import time
import lasio
from bayesian.td_tool.export_las import to_las, resample_log
from bayesian.td_tool.smda_api.smda_api import get_wellbore_trajectory
#from pages._3_Checkshot_Data import get_data, filter_data
plots_posteriori = False
st.write('# Bayesian Inversion')
col1, col2 = st.columns(2)
with col1:
    st.write('## Application')
    st.write('Welcome to the Bayesian Inversion section! Bayesian inference can be a method to apply drift on the sonic log using checkshot data. The aim is to generate a time-depth relationship that keeps the high-resolution from sonic log, but that also matches the full coverage from checkshot data.\
            The output is a ready-to-use velocity trend for multipliple purposes: well-tie; depth conversion; seismic depth processing; etc. You can either run the bayesian inference with Standard values or select some of the parameters yourself.')
try:
    df_sonic = st.session_state['Sonic_log']
    df_checkshot = st.session_state['Checkshot']
    seabed = st.session_state['seabed']
    uwi = st.session_state['uwi']
except:
    df_sonic = pd.DataFrame()
    df_checkshot = pd.DataFrame()
    pass

if df_checkshot.empty:
    st.write(f"No data available. Please select and save some well in section '3 Data Visualization'")
    exit(0)
else:
    pass    
if df_sonic.empty:
    st.write(f"No sonic log available for Well {uwi}. In order to apply the Bayesian Inversion Method it is necessary to have both Checkshot data and Sonic Log data.")
    exit(0)
else:
    pass 


with col2:
    st.write('## Select parameters for Bayesian Inversion')
    st.write(f"On section '3 Data Visualization' you selected well {uwi}. You can now select parameters to run the bayesian inversion yourself.")
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        decimation_step = st.text_input(f"Define a decimation step. The lowest the value more accurate your inversion will be.", 10)
        decimation_step = int(decimation_step)
    with col1_2:
        inversion_start_depth = st.text_input(f"Assign from which depth the inversion is starting from..\
                                                Standard value for well {uwi} is seabed depth: {float(df_checkshot[(df_checkshot['depth_source'] == 'seabed from smda') | (df_checkshot['depth_source'] == 'seabed detected')]['tvd_ss'])} m", float(df_checkshot[(df_checkshot['depth_source'] == 'seabed from smda') | (df_checkshot['depth_source'] == 'seabed detected')]['tvd_ss']))
        inversion_start_depth = float(inversion_start_depth)

    st.write("### Exponential Correlation Matrix")
    st.write("An exponential correlation matrix is a mathematical tool used to model spatial autocorrelation between data points. In the context of Bayesian inversion, this matrix is employed to introduce spatial correlation into the prior model. This correlation assumes that closely spaced depth samples exhibit higher correlation than those that are more distant, with the correlation decaying exponentially as the distance increases.")
    col1_3_1, col1_3_2 = st.columns(2)
    with col1_3_1:
        apply_covariance = st.radio(
            "Select Depth Covariance:",
            options=["Do not apply", "Apply"],
        )
    with col1_3_2:
        if apply_covariance == 'Apply':
            corr_order = st.text_input(f"Define a Correlation order between 1 and 2. It determines the shape of the correlation function. A higher correlation order implies a slower decay in correlation as the distance between data points increases, indicating stronger correlation between more distant points.", 1.8)
            corr_order = float(corr_order)
        else:
            corr_order = None

    st.write('### Definition of uncertainties')
    st.write('Uncertainty associated to Sonic log shall be higher than the one associated to Checkshot Data')
    col1_4, col1_5 = st.columns(2)
    
    with col1_4:
        std_sonic = st.text_input(f"Enter standard deviation for Sonic:", 500)
        std_sonic = float(std_sonic)
    with col1_5:
        std_checkshot = st.text_input(f"Enter standard deviation for Checkshot:", 0.005)
        std_checkshot = float(std_checkshot)    
with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Line(x=df_checkshot['interval_velocity'], y=df_checkshot['tvd_ss'], line_shape='hv',line=dict(color='red'),name='Vp Checkshot'))
    
    try:
        #std_sonic = st.slider("Standard deviation: Vp", 400, 600)
        df_sonic = df_sonic.dropna(subset=['interval_velocity_sonic'])
        df_sonic['std_sonic'] = std_sonic   
        df_sonic['u+std'] = df_sonic['interval_velocity_sonic'] + df_sonic['std_sonic']
        df_sonic['u-std'] = df_sonic['interval_velocity_sonic'] - df_sonic['std_sonic']           
        
        fig2.add_trace(go.Line(x=df_sonic['interval_velocity_sonic'],y=df_sonic['tvd_ss'],name="Sonic Log", marker_color='blue'))
        
        fig2.add_trace(go.Line(x=df_sonic['u+std'],y=df_sonic['tvd_ss'],fill=None,name="Uncertainty",line_color="rgba(0,0,0,0)", showlegend=False))
        fig2.add_trace(go.Line(x=df_sonic['u-std'],y=df_sonic['tvd_ss'],fill='tonexty',name="Uncertainty - Sonic Log", line_color="rgba(0,0,0,0)"))
        #fig2.add_trace(go.Scatter(x=[-1e4, 1e4], y=[floatwater_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed', legendgroup = '1'), row=1, col=1)
        fig2.add_trace(
            go.Scatter(
                x=[fig2['data'][0]['x'][0], fig2['data'][0]['x'][-1]],  # Extend the line across the x-axis
                y=[float(st.session_state['seabed']), float(st.session_state['seabed'])],  # Set the y-coordinate for the horizontal line
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Seabed'  # Customize the line style
            )
        )
        # Filling the area between the upper and lower bounds
    except:
        pass  
    fig2.update_traces(connectgaps=True) 
    fig2.update_layout(
    title=f'Velocity plot for Well {uwi}',
    xaxis_title="Vp (m/s)",
    yaxis_title='TVDSS (m)',
    autosize=False,
    width=600,
    height=1200,
    yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])],
    showlegend=True)
        
    st.plotly_chart(fig2)
    st.write("Figure 1. Checkshot velocities (Vp Checkshot), Sonic Log Velocities, and uncertaintity associated to Sonic Log (mean - standard deviation).")

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
if "button_clicked2" not in st.session_state:
    st.session_state.button_clicked2 = False
if "button_resampling" not in st.session_state:
    st.session_state.button_resampling = False    

def callback():
    st.session_state.button_clicked = True
def callback2():
    st.session_state.button_clicked2 = True
def callback3():
    st.session_state.button_clicked = True
    st.session_state.button_clicked2 = True
def callback4():
    st.session_state.button_clicked = True
    st.session_state.button_clicked2 = True



with col2:
    st.write(st.session_state.button_clicked, st.session_state.button_clicked2, st.session_state.button_resampling)
    st.session_state.button_clicked = False
    if st.session_state.button_resampling:
        callback()
    if (st.button("Run Bayesian Dix Inversion") or st.session_state.button_clicked2):
        st.session_state.button_clicked = True
        st.session_state.button_clicked2 = False
        start_time = time.time()
        clas = Bayesian_Inference()
        df_well, td_z, td_t, ww, water_depth, water_velocity, C, std_total_depth = clas.run(df_checkshot, df_sonic, std_sonic, std_checkshot, apply_covariance, corr_order, inversion_start_depth, decimation_step, uwi)
        df_well = df_well.rename(columns={'TVDMSL':'tvd_ss'})
        #st.write(df_checkshot, td_z)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Time taken to run Bayesian Dix Inversion: {elapsed_time:.2f} seconds")
        
        #well_z_md = df_well['md'].values
        #st.write(df_well)
        well_z_md = df_well['md'].values
        well_z = df_well['tvd_ss'].values
        well_vp_ext = df_well['VP_EXT'].values
        well_vp = df_well['VP_BAYES'].values
        
        td_vp = getVel(td_z, td_t)

        td_df = pd.DataFrame({'depth':td_z,'twt':td_t,'vp':td_vp})
        td_df = td_df.dropna()
        td_md = df_checkshot['md']
        td_z = td_df['depth'].values
        td_t = td_df['twt'].values
        td_vp = td_df['vp'].values

        well_t_ext = getTime(well_z, well_vp_ext)

        well_t = getTime(well_z, well_vp)

        td_drift_t_ext, td_drift_z_ext, well_drift_t_ext, well_drift_z_ext = getDrift(td_z, td_t, well_z, well_t_ext)  

        td_drift_t, td_drift_z, well_drift_t, well_drift_z = getDrift(td_z, td_t, well_z, well_t)    
        #st.write(well_t_ext)

        yr = [np.max(np.union1d(well_z, td_z)) * 1.1, 0]
        plots_posteriori = True        
    else:
        st.write("Inversion not running.")
    
    st.write(st.session_state.button_clicked, st.session_state.button_clicked2)

#df_checkshot, df_sonic, df_merged = filter_data(df_checkshot, df_sonic, raw_cks_df, uwi)

if (st.session_state.button_clicked == True) or (st.session_state.button_clicked2 == True):
   

    st.write("## Velocity plots")
    col1_plot, col2_plot, col3_plot, col4_plot = st.columns(4)
    with col1_plot:
        st.write("### Velocity Input (Prior)")

    with col2_plot:
        st.write("### Velocity Output (Posterior)")
    with col3_plot:
        st.write("### Comparison")
    with col4_plot:
        st.write("### Velocity Difference")
        #if st.session_state.button_clicked:
        #    answer_plot = st.radio("Should the velocity difference be shown in:", ("Vp Output - Vp Input", "Vp Input - Vp Output"))
        #    callback3()
        #    if answer_plot == "Vp Output - Vp Input":
        #        xx = well_vp - well_vp_ext
        #    elif answer_plot == "Vp Input - Vp Output":
        #        xx =  well_vp_ext - well_vp

    fig = make_subplots(rows=1, cols=4, shared_yaxes=True)
    fig.add_trace(go.Scatter(x=well_vp_ext, y=well_z, mode='lines', name='Vp Prior', line=dict(color='green'), legendgroup = '1'), row=1, col=1)
    fig.add_trace(go.Line(x=td_vp, y=td_z, line_shape='hv',line=dict(color='red'), name='Vp Checkshot', legendgroup = '1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed', legendgroup = '1'), row=1, col=1)

    fig.add_trace(go.Scatter(x=well_vp, y=well_z, mode='lines', name='Vp Posterior', line=dict(color='blue'), legendgroup = '2'), row=1, col=2)
    fig.add_trace(go.Line(x=td_vp, y=td_z, line_shape='hv',line=dict(color='red'),name='Vp Checkshot', legendgroup = '2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed', legendgroup = '2'), row=1, col=2)


    fig.add_trace(go.Scatter(x=well_vp_ext, y=well_z, mode='lines', name='Vp Prior', line=dict(color='green'), legendgroup = '3'), row=1, col=3)
    fig.add_trace(go.Scatter(x=well_vp, y=well_z, mode='lines', name='Vp Posterior', line=dict(color='blue'), legendgroup = '3'), row=1, col=3)
    fig.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed', legendgroup = '3'), row=1, col=3)
    fig.add_trace(go.Line(x=td_vp, y=td_z, line_shape='hv',line=dict(color='red'),name='Vp Checkshot', legendgroup = '3'), row=1, col=3)    
    xx = well_vp - well_vp_ext
    fig.add_trace(go.Scatter(x=xx, y=well_z, mode='lines', line=dict(color='blue'), name="Vp Posterior - Vp Prior", legendgroup = '4'), row = 1, col=4)
    
    xmin = np.min(xx) - 10
    xmax = np.max(xx) + 10
    fig.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed', legendgroup = '4'), row = 1, col=4)

    fig.add_vline(x=0, line_width=0.5, line_color="black", row = 1, col=4, legendgroup = '4')
    fig.update_xaxes(title_text="Interval Velocity (m/s)", range =[0, np.max(np.union1d(well_vp, td_vp)) * 1.2], row=1, col=1)
    fig.update_xaxes(title_text="Interval Velocity (m/s)", range=[0, np.max(np.union1d(well_vp, td_vp)) * 1.2], row=1, col=2)
    fig.update_xaxes(title_text="Interval Velocity (m/s)", range=[0, np.max(np.union1d(well_vp, td_vp)) * 1.2], row=1, col=3)
    fig.update_xaxes(title_text="Interval Velocity Difference (m/s)", range=[xmin, xmax], row=1, col=4)

    fig.update_layout(height=1300, 
                    width=1300,
                    yaxis=dict(autorange="reversed", title="TVDSS  (m)"),
                    legend=dict(orientation="h", xanchor="center", x=0.5, tracegroupgap=1000))
    st.plotly_chart(fig, use_container_width=True)
    col1_legend, col2_legend, col3_legend, col4_legend = st.columns(4)
    with col1_legend:
        st.write("Figure 2. On the first plot it is shown the interval velocity from both checkshot and sonic log. The velocity from sonic was interpolated up to the seabed as the starting point for the bayesian inversion method.")
    with col2_legend:
        st.write("Figure 3. The second plot shows the updated time-depth relationship plotted against checkshot values.")
    with col3_legend:
        st.write("Figure 4. The third plot shows the comparison between the updated velocity trend and the one only using sonic log data interpolated to the seabed.")
    with col4_legend:
        st.write("Figure 5. plot shows the difference between the updated velocity trend and the one only using sonic log data interpolated to the seabed.")


    col4, col5 = st.columns(2)
    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Line(x=1e3 * well_t_ext, y=well_z, line=dict(color='green'), name='TWT Prior'))
        fig4.add_trace(go.Line(x=1e3 * well_t, y=well_z, line=dict(color='blue'),  name='TWT Posterior'))
        fig4.add_trace(go.Scatter(x=1e3 * td_t, y=td_z, mode='markers', line=dict(color='red'),  name='TWT Checkshot'))
        fig4.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed'))
        fig4.update_xaxes(title_text='ms')

        fig4.update_layout(
        title=f'Time Domain',
        xaxis_title="TWT (ms)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=900,
        height=1200,
        xaxis_range=[-100, np.max(1e3 * well_t) * 1.1],
        yaxis_range=[0,yr],
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h",xanchor = "center",x = 0.5),legend_tracegroupgap=300)
        st.plotly_chart(fig4)

    with col5:
        fig5 = go.Figure()
        fig5.add_trace(go.Line(x=1e3 * td_drift_t_ext, y=td_z, line=dict(color='green'), legendgroup="4", name='Drift Prior'))
        fig5.add_trace(go.Line(x=1e3 * td_drift_t, y=td_z, line=dict(color='blue'), legendgroup="4", name='Drift Posterior'))
        xx = np.insert(well_drift_t, 0, well_drift_t_ext)
        xx = xx[~np.isnan(xx)]
        fig5.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash'), name='Seabed'))
        fig5.update_yaxes(title_text="Y-axis 1")
        fig5.update_xaxes(title_text="DRIFT (OWT)")
        fig5.add_vline(x=0, line_width=1, line_color="black")

        fig5.update_layout(
        title=f"DRIFT",
        xaxis_title="TWT (ms)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=900,
        height=1200,
        xaxis_range=[1e3 * np.min(xx) - 3, 1e3 * np.max(xx) + 3],
        yaxis_range=[0,yr],
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h",xanchor = "center",x = 0.5),legend_tracegroupgap=300)
        st.plotly_chart(fig5)

    col4_legend, col5_legend = st.columns(2)
    with col4_legend:
        st.write("Figure 6. Plot showing the cumulative two-way travel times for checkshot data, posterior velocity model, and prior velocity model.")
    with col5_legend:
        st.write("Figure 7. The difference between the integrated sonic log and the checkshot in time is known as the drift curve. It is supposed that the drift between the new time-depth curve from bayesian inversion is smaller than the one coming from sonic log alone.")

    import plotly.figure_factory as ff
    with col2:
            if st.button("Display Advanced Parameters", on_click=callback2):
                col2_5, col2_6 = st.columns(2)
                std_total_depths = [[float(value) for value in std_total_depth]]
                df_standard_deviation_vel = pd.DataFrame({"std":std_total_depth,"tvd_ss":well_z})   
                with col2_5:
                    fig5 = go.Figure()
                    fig5.add_trace(go.Scatter(x=df_standard_deviation_vel["std"],y=df_standard_deviation_vel['tvd_ss'],name="Standard Deviation Velocity",line_color="blue"))

                    fig5.update_traces(connectgaps=True) 
                    fig5.update_layout(
                    xaxis_title="Standard Deviation (m/s)",
                    yaxis_title='TVDSS (m)',
                    autosize=False,
                    width=400,
                    height=600,
                    #showlegend=True,
                    yaxis=dict(autorange='reversed'))        
                    st.plotly_chart(fig5)
                    st.write("Figure 8. Standard deviation from time-depth relationship a posteriori along depth.")
                with col2_6:
            
                    fig6 = px.box(df_standard_deviation_vel["std"])
                    fig6.update_layout(
                    xaxis_title="Standard Deviation (m/s)",
                    yaxis_title='TVDSS (m)',
                    autosize=False,
                    width=400,
                    height=600,
                    #showlegend=True
                    )                  
                    st.plotly_chart(fig6)
                    st.write("Figure 9. Boxplot for the standard deviation of velocity posteriori. This is a direct measure of the uncertainty associated to the time-depth relationship")    
                    #fig, ax = plt.subplots()

                    #ax.hist(df_standard_deviation_vel["std"])
                    #st.plotly_chart(fig)
            else:
                pass
from scipy.interpolate import interp1d  
st.write("### Export Las")

with st.form("my_form"):
    col1, col2 = st.columns(2)
    st.write("Inside the form")
    with col1:
        
        answer_step = st.radio("Should depth resampling be performed to generate a curve with uniform depth intervals?", ("True", "False"))
        st.write(answer_step)
        depth_step = st.text_input(f"Depth step", 0.2)
        
        depth_export = st.radio("Should depth be generated in MD or TVDSS?", ("MD", "TVDSS"))
        st.write("** As the estimation of the velocity trend is done in TVDSS, it is necessary to fetch SMDA wellbore trajectory to convert TVDSS to MD for missing values.")

    with col2:
        answer_petrel = st.radio("Should the file be exported in Petrel format", ("True", "False"))
        if answer_petrel == "True":
            petrel_format = True  
        else:
            petrel_format = False    
    

    # Every form must have a submit button.
    if answer_step == 'True':
        depth_step = float(depth_step)
    else:
        depth_step = False
    submitted = st.form_submit_button("Generate LAS file")      
    if submitted:
        st.write("answer_step", depth_step, "petrel_format", petrel_format)

st.write("Outside the form")
st.write(answer_step, answer_petrel)
col1, col2 = st.columns(2)
with col1:
    st.write("#### Export Bayesian velocity updated in MD")
    st.write("##### * As the estimation of the velocity trend is done in TVDSS, it is necessary to fetch SMDA wellbore trajectory to convert TVDSS to MD.")
with col2:
    st.write("#### Export Bayesian velocity updated in TVDSS")
col1, col2 = st.columns(2)

with col1:
    if st.button("Generate LAS file in MD", on_click=callback3):
        results_api, msg = get_wellbore_trajectory(uwi=uwi)
        st.write(msg)
        if results_api:
            if pd.DataFrame(results_api['data']['results']).empty is False:
                

                df_wellbore = pd.DataFrame(results_api['data']['results'])
                df_wellbore = df_wellbore.sort_values(by='md')
                md_interp = interp1d(df_wellbore['md'], df_wellbore['tvd_msl'], kind='linear', fill_value=np.nan, bounds_error=False)
                missing_md_indices = df_well['md'].isnull()
                df_well.loc[missing_md_indices, 'md'] = md_interp(df_well.loc[missing_md_indices, 'tvd_ss'])
                if df_well['md'].isnull().any():
                    st.write(f"Warning: The following TVDSS values could not be calculated because these are out of range from wellbore trajectory in SMDA: {df_well[df_well['md'].isnull()]['tvd_ss'].values}")
                df_well = df_well.dropna(subset=['md'])
                st.write(f"Trajectory in SMDA for well {uwi} goes from MD:{df_wellbore['md'].min()}m to MD:{df_wellbore['md'].max()}m and MD missing points were interpolated from TVDSS within this interval")
                to_las(depth_path="MD", uwi=uwi,output_file="las_export/test", depth_in=np.array(df_well['md']), vp_input =np.array(df_well['VP_IN']), vp_ext=np.array(df_well['VP_EXT']),vp_output=np.array(df_well['VP_BAYES']), depth_step=depth_step)
                st.write(f"Well {uwi} correctly exported as LAS file")
                st.write(df_well)
            else:
                st.write(f"No wellbore trajectory data from SMDA plan survey samples for file {uwi}. It is not possible to convert TVDSS to MD. No LAS file is outputted")
        else:
            st.write(f"Due to an unsuccessful API connection for file {uwi}, no LAS file was generated.")
            pass

with col2:
    if st.button("Generate LAS file in TVDSS", on_click=callback3):
        to_las(depth_path="TVDSS", uwi=uwi,output_file="las_export/test", depth_in=np.array(df_well['tvd_ss']), vp_input =np.array(df_well['VP_IN']), vp_ext=np.array(df_well['VP_EXT']),vp_output=np.array(df_well['VP_BAYES']), depth_step=depth_step)
        

        #df_well = resample_tvdss_md(df_well, df_wellbore)
        #to_las(uwi=uwi,output_file="test", depth_in=np.array(df_well['md']), vp_input=np.array(df_well['VP_EXT']),vp_output=np.array(df_well['VP_BAYES']))
        st.write(f"Well {uwi} correctly exported as LAS file")
        

