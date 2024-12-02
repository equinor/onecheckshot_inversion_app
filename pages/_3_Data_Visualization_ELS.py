from pathlib import Path

import sys
import streamlit as st
import pandas as pd
import math

import plotly.express as px
from IPython import embed
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
import git
import os
sys.path.append(os.getcwd())
from bayesian.td_tool.bayes_csc import getTime
from bayesian.td_tool.td_lib import getVel
#import bayesian.td_tool.ELS_log
import pandas as pd
from bayesian.td_tool.data_management_smda.connect_smda import Connection_Database, get_connect_database, generate_df
from scipy.interpolate import interp1d
from bayesian.td_tool.ELS_log import Connection_ELS_LOG, load_els_data


#raw_cks_df, df_checkshot, df_sonic = get_data()
host, dbname, user, password, sslmode = get_connect_database()
connect = Connection_Database(host,dbname,user,password,sslmode)

database_checkshot = "smda.smda_workspace.wellbore_checkshot_data_qc"
wells = connect.get_wells(database_checkshot)

connect.close_connection()

st.write("## Data Visualization")
def select_well():
    selected_value = st.selectbox(f"Select Checkshot Wellbore", options=sorted(wells))
    return selected_value

uwi = select_well()
st.write(f"Well selected: {uwi}")

host, dbname, user, password, sslmode = get_connect_database()

columns = 'md, tvd, tvd_ss, depth_reference_elevation, depth_source, time, source_file, unique_wellbore_identifier, average_velocity, interval_velocity, qc_description, md_increasing, tvd_ss_increasing, time_increasing, average_velocity_qc, trajectory_checked, mae_soniclog_checkshot, comparison_sonic_log_qc, preference_checkshotfile'
df = generate_df(host, dbname, user, password, sslmode, columns, database_checkshot, uwi)

df.loc[df['depth_source']=='seabed from smda','average_velocity'] = 1478.2
df.loc[df['depth_source']=='seabed from smda','interval_velocity'] = 1478.2

col1, col2 = st.columns(2)
with col1:
    selected_source = st.selectbox(f"Select Checkshot File", options=df['source_file'].unique())

    df = df[(df['source_file'] == selected_source) & (df['md_increasing'] == 'true')]
    df.sort_values(by=['md'], ascending=[True], inplace=True)
    st.write(df[['md', 'tvd', 'tvd_ss','depth_source','depth_reference_elevation', 'time', 'average_velocity', 'average_velocity_qc', 'interval_velocity', 'mae_soniclog_checkshot','qc_description']])


with col2:
    selected_source_well_log = st.selectbox(f"Select Sonic log File Source", options=['LFP', 'FMB'])

    if "connection_ELS" not in st.session_state:
        st.session_state.connection_ELS = False
    try:

        @st.cache_data
        def load_els_log(uwi):
            connection_els = Connection_ELS_LOG()
            df = connection_els.kusto_query_LFP(uwi)
            st.session_state.connection_ELS = True
            return df
        df_sonic_els = load_els_log(uwi)

    except Exception as e:
        st.write(f'Problem with connection to ELS API. Error: {e}')
    
    #st.write(load_els_log(uwi, 'LFP_DT', connection_els))
    options_log = ['LFP_DT', 'LFP_DT_O', 'LFP_DT_G', 'LFP_DT_B']
    selected_log_curve = st.selectbox(f"Select Sonic Log Curve", options=options_log)
    df_sonic = load_els_data(df_sonic_els, selected_log_curve)
    st.write(df_sonic)




    #df_sonic = load_els_log(uwi, selected_curve_welllog, connection_els)
    #if st.session_state.connection_ELS == True:
    #selected_source_welllog = st.selectbox(f"Select Sonic Log File", options=df_filtered['source'].unique())
        #st.write(df_sonic)    
    


#df_sonic = load_els_log(uwi, selected_curve_welllog, connection_els)

td = df[['md','tvd','tvd_ss', 'depth_reference_elevation', 'time', 'qc_description', 'average_velocity', 'interval_velocity', 'md_increasing', 'tvd_ss_increasing', 'time_increasing', 'average_velocity_qc', 'trajectory_checked', 'depth_source']]






col1, col2 = st.columns(2)
with col1:
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        #st.write("Plot Checkshot data. It was performed a quality control for this data following https://github.com/equinor/qc_and_clean_cks.")
        fig1_1 = go.Figure()
        fig1_1.add_trace(go.Scatter(y=td["tvd_ss"],x=td["time"],mode="markers",marker=dict(color='red',size=10),name='Checkshot data'))

        fig1_1.update_layout(
        title=f'Checkshot data : Well {uwi}',
        xaxis_title="TWT (ms)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=800,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        )
        st.plotly_chart(fig1_1)

    with col1_2:
        fig1_2 = go.Figure()
        qc = ['md_increasing', 'tvd_ss_increasing', 'time_increasing', 'average_velocity_qc', 'trajectory_checked']
        for col in qc:
            colors = ['green' if (b=='true' or b=='passed') else 'gray' if (b=='not checked') else 'red' for b in td[col]]
            fig1_2.add_trace(go.Scatter(
                y=td["tvd_ss"],
                x=[col for i in range(len(td["tvd_ss"]))],
                mode="markers",
                marker=dict(
                    color=colors,
                    size=10),
                name=col
            ))

        #fig1_2 = px.scatter(td, x="md_increasing", y="tvd_ss")
        fig1_2.update_layout(
        xaxis=dict(tickvals=[0, 1, 2, 3, 4, 5], ticktext=['MD', 'TVDSS', 'Time', 'Average vel', 'Trajectory']),
        autosize=False,
        width=10,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        showlegend=False
        )
        fig1_2.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig1_2)
with col2:
    try:
        fig2 = px.line(df_sonic, x="interval_velocity_sonic", y="tvd_ss")  # Replace "x_column" and "y_column" with the appropriate column names from df_2   
        fig2.update_traces(connectgaps=False) 
        fig2.update_layout(
        title=f'Sonic Log data : Well {uwi}',
        xaxis_title="Vp (m/s)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=500,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])])
        
        st.plotly_chart(fig2)
    except:
        pass

    df_checkshot_plot2 = td.copy(deep=True)

#continue from here
#time_sonic = getTime(df_sonic['tvd_ss'], df_sonic['interval_velocity_sonic'])

#st.write([type(x) for x in np.array(df_sonic['tvd_ss']).astype(float)])
#st.write([type(x) for x in df_sonic['interval_velocity_sonic']])

def decimate_dataframe(df, decimate_step):
    """Decimates a DataFrame by selecting every `decimate_step`-th row.

    Args:
        df: The input DataFrame.
        decimate_step: The decimation step size.

    Returns:
        The decimated DataFrame.
    """

    ibayes = np.arange(decimate_step - 1, len(df), decimate_step)
    ibayes[-1] = len(df) - 1  # Ensure the last row is included

    df_decimated = df.iloc[ibayes]
    return df_decimated

col1, col2 = st.columns(2)
with col1:
    decimate_step = int(st.text_input(f"Enter Decimation Step for Checkshot Data:", 0))
    if decimate_step == 0:
        pass
    else:
        
        td_seabed_sealevel = td[(td['depth_source'].str.contains('seabed') | (td['depth_source'].str.contains('sealevel')))]
        td_to_decimate = td[~(td['depth_source'].str.contains('seabed') | (td['depth_source'].str.contains('sealevel')))]
        td_decimate = decimate_dataframe(td_to_decimate, decimate_step)
        td = (pd.concat([td_seabed_sealevel, td_decimate]))


#embed()
col3, col4 = st.columns(2)
with col3:
    container1 = st.container()
    with container1:
        # Create your plot here
        st.write('## Time Domain')
        fig1 = go.Figure()

        #fig1 = px.scatter(df_checkshot_plot2, x="twt picked", y="tvd_ss")

      
        fig1 = go.Figure()
        #fig1.add_trace(go.Line(x=df_checkshot_plot2["time"],y=df_checkshot_plot2['tvd_ss'],name="Checkshot", marker_color='red'))
        fig1.add_trace(go.Scatter(y=td["tvd_ss"],x=td["time"],mode="markers",marker=dict(color='red',size=10),name='Checkshot data'))
        #fig1.add_trace(go.Line(x=df_checkshot_plot2['u+std'],y=df_checkshot_plot2['tvd_ss'],fill=None,name="u+std",line_color="rgba(0,0,0,0)"))
        #fig1.add_trace(go.Line(x=df_checkshot_plot2['u-std'],y=df_checkshot_plot2['tvd_ss'],fill='tonexty',name="u+std", line_color="rgba(0,0,0,0)"))
        try:
            #time = getTime(np.array(df_sonic['tvd_ss'].astype(float)), np.array(df_sonic['interval_velocity_sonic']))*1000
            #TWT for sonic data 
            td_tointerpolate = td[['tvd_ss','time']].dropna()
            interp_func = interp1d(td_tointerpolate['tvd_ss'].astype(float), td_tointerpolate['time'], kind='linear')
            first_depth_sonic = float(np.array(df_sonic['tvd_ss'])[0])
            first_time_sonic = float(interp_func(first_depth_sonic))
            dz = np.diff(np.array(df_sonic['tvd_ss'].astype(float)))*1000
            dz = np.insert(dz, 0, 0)
            dt = 2 * dz / np.array(df_sonic['interval_velocity_sonic'])
            t = np.cumsum(dt)+first_time_sonic
            fig1.add_trace(go.Line(x=t,y=df_sonic['tvd_ss'].astype(float),name="Sonic Log", marker_color='blue'))
        except:
            st.write('No Sonic log available for this well')

        fig1.update_layout(
        title=f'#Time Domain',
        xaxis_title="TWT (ms)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=900,
        height=1800,
        yaxis_range=[max(df_checkshot_plot2["tvd_ss"]), min(df_checkshot_plot2["tvd_ss"])])
        
        st.plotly_chart(fig1)

df_sonic_plot2 = df_sonic.copy(deep=True)


if isinstance(td, pd.DataFrame): 
    
    # get data
    td_z = td['tvd_ss'].values
    td_t = td['time'].values
    td_t = td_t*0.001
    td_vp = td['interval_velocity'].values
    

else:
    print('ERROR: empty time depth - skipping well')
    runWell = 0 # skip this well

with col4:
    container2 = st.container()
    with container2:
        st.write('## Velocity Domain')
        fig2 = go.Figure()
        fig2.add_trace(go.Line(x=td_vp,y=td_z, line_color='red', name='Vp Checkshot', line_shape='hv'))


        try:

            df_sonic_plot2 = df_sonic_plot2.dropna(subset=['interval_velocity_sonic'])

        
            
            fig2.add_trace(go.Line(x=df_sonic_plot2['interval_velocity_sonic'],y=df_sonic_plot2['tvd_ss'],name="Sonic Log", marker_color='blue'))
            



            # Filling the area between the upper and lower bounds
        except:
            pass
            
        fig2.update_traces(connectgaps=True) 
        fig2.update_layout(
        title=f'Velocity Domain',
        xaxis_title="Vp (m/s)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=900,
        height=1800,
        yaxis_range=[max(df_checkshot_plot2["tvd_ss"]), min(df_checkshot_plot2["tvd_ss"])],
        showlegend=True)


            #fig2 = px.line(df_sonic, x="vp", y="tvd_ss")  # Replace "x_column" and "y_column" with the appropriate column names from df_2   
            #fig2.update_traces(connectgaps=True) 
            #fig2.update_layout(
            #title=f'Sonic Log data : Well {uwi}',
            #xaxis_title="Vp (m/s)",
            #yaxis_title='TVDSS (m)',
            #autosize=False,
            #width=400,
            #height=900,
            #yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])])
            
        st.plotly_chart(fig2)



if st.button("Save Checkshot and Sonic data:"):
    

    try:
        st.session_state['Checkshot'] = td
        st.write(f"Checkshot file for well {uwi} saved")
    except:
        st.write(f'No Checkshot file for well {uwi}')
    try:
        st.session_state['Sonic_log'] = df_sonic
        if not df_sonic.empty:
            st.write(f"Sonic log saved for well {uwi}")
        else:
            pass
            #st.write(f'No sonic log file for well {uwi}')
    except:
        print('d')
    st.session_state['uwi'] = uwi
else:
    st.write("Files not yet saved...")


