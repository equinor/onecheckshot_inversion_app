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
import pandas as pd
import psycopg2
import yaml

class Connection_Database:
  def __init__(self,host,dbname,user,password,sslmode):
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
    self.conn = psycopg2.connect(conn_string)
    self.conn.set_client_encoding('UTF8')
    print("Connection established")

  def connect_database(self,database, columns):
    self.query = "SELECT {} from {}".format(columns, database)
    self.df = pd.read_sql_query(self.query, self.conn)
    return self.df

  def close_connection(self):
    self.conn.close()

def chksht_welldb_smda():
  
  config_file = os.path.join(os.getcwd(),"bayesian","td_tool", "smda_password","config.yaml")
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  host = config['host']
  dbname = config['dbname']
  user = config['user']
  password = config['password']
  sslmode = config['sslmode']   
  database_wellbore_checkshot = "smda.smda_workspace.wellbore_checkshot_data"
  connect = Connection_Database(host,dbname,user,password,sslmode)
  #columns_wellbore_checkshot = "id, unique_wellbore_identifier, source_file, tvd_ss, time, time_unit, tvd, tvd_unit, md, md_unit"
  columns_wellbore_checkshot = "*"
  df = connect.connect_database(database=database_wellbore_checkshot,columns=columns_wellbore_checkshot)
  database_smda = "smda.smda_master.v_wellbore_time_depth_data"
  columns_smda = "*"
  df_smda = connect.connect_database(database=database_smda,columns=columns_smda)
  connect.close_connection()
  return df, df_smda


@st.cache_data

def get_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
    
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parents[1]/'data/checkshot_sonic_table.csv'
    raw_cks_df = pd.read_csv(DATA_FILENAME, index_col=False)
    df, df_smda = chksht_welldb_smda()
    #df.rename(columns={'unique_wellbore_identifier': 'uwi', 'time': 'twt picked', 'average_velocity': 'average velocity', 'interval_velocity': 'interval velocity'}, inplace=True)
    #df_checkshot = df[['uwi','tvd_ss','tvd_unit','twt picked','time_unit', 'average velocity', 'interval velocity','source_file']].dropna(subset='twt picked')

    df_checkshot = raw_cks_df[['uwi','tvd_ss','twt picked', 'average velocity', 'interval velocity']].dropna(subset='twt picked')
    df_sonic = raw_cks_df[['uwi','tvd_ss','vp', ]].dropna(subset='vp')
    df_sonic['vp'] = df_sonic['vp'].fillna(False)

    return raw_cks_df, df_checkshot, df_sonic #gdp_df

raw_cks_df, df_checkshot, df_sonic = get_data()

def filter_data(df_checkshot, df_sonic, raw_cks_df, uwi):
    df_checkshot = df_checkshot[df_checkshot['uwi'] == uwi]
    df_sonic = df_sonic[df_sonic['uwi'] == uwi]
    df_merged = raw_cks_df[raw_cks_df['uwi'] == uwi]
    return df_checkshot, df_sonic, raw_cks_df


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Checkshot Data

SMDA Platform for Time-Depth relationship data extraction. (https://opus.smda.equinor.com/) SMDA website. 
'''

# Add some spacing
''

wells = raw_cks_df.uwi.unique()
st.write("## Data Visualization")
def select_well():
    selected_value = st.selectbox(f"Select Well", options=wells)
    return selected_value

uwi = select_well()

df_checkshot, df_sonic, df_merged = filter_data(df_checkshot, df_sonic, raw_cks_df, uwi)
td = df_checkshot
def return_well():
    return uwi

if __name__ == '__main__':


    if isinstance(td, pd.DataFrame): 
        
        # get data
        td_z = td['tvd_ss'].values
        td_t = td['twt picked'].values
        td_t = td_t*0.001
        td_vp = getVel(td_z, td_t)
        

    else:
        print('ERROR: empty time depth - skipping well')
        runWell = 0 # skip this well
        

    print('... done.')

    # get log data
    well_z = df_sonic['tvd_ss'].values
    well_vp = df_sonic['vp'].values
    well_t = getTime(well_z, well_vp)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Checkshot Data")
        st.table(df_checkshot[['uwi','twt picked','tvd_ss', 'average velocity']].set_index('uwi').head(10))

    with col2:
        st.write("Sonic Log data")
        st.table(df_sonic.set_index('uwi').head(10))

    #min_value = df_checkshot['tvd_ss'].min()
    #max_value = df_checkshot['tvd_ss'].max()


    wells = df_checkshot['uwi'].unique()

    if not len(wells):
        st.warning("Select at least one well")


    col1, col2 = st.columns(2)
    with col1:
        container1 = st.container()
        with container1:
            # Create your plot here
            st.write("Plot Checkshot data. It was performed a quality control for this data following https://github.com/equinor/qc_and_clean_cks.")
            fig1 = px.scatter(df_checkshot, x="twt picked", y="tvd_ss")
            fig1.update_layout(
            title=f'Checkshot data : Well {uwi}',
            xaxis_title="TWT (ms)",
            yaxis_title='TVDSS (m)',
            autosize=False,
            width=400,
            height=900,
            yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])])
            
            st.plotly_chart(fig1)

    with col2:
        container2 = st.container()
        with container2:
            st.write("Plot Sonic Log data. It was performed a quality control for this data following xxx.")
            fig2 = px.line(df_sonic, x="vp", y="tvd_ss")  # Replace "x_column" and "y_column" with the appropriate column names from df_2   
            fig2.update_traces(connectgaps=True) 
            fig2.update_layout(
            title=f'Sonic Log data : Well {uwi}',
            xaxis_title="Vp (m/s)",
            yaxis_title='TVDSS (m)',
            autosize=False,
            width=400,
            height=900,
            yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])])
            
            st.plotly_chart(fig2) 
    
    st.write("## Definition of uncertainties")
    
    df_checkshot_plot2 = df_checkshot.copy(deep=True)

    col3, col4 = st.columns(2)
    with col3:
        container1 = st.container()
        with container1:
            # Create your plot here
            fig1 = go.Figure()
            std_checkshot = st.slider("Standard deviation: TWT", 0.005, 0.05)
            df_checkshot_plot2['std_checkshot'] = std_checkshot
            #fig1 = px.scatter(df_checkshot_plot2, x="twt picked", y="tvd_ss")

            df_checkshot_plot2['u+std'] = df_checkshot_plot2["twt picked"] + df_checkshot_plot2['std_checkshot']
            df_checkshot_plot2['u-std'] = df_checkshot_plot2["twt picked"] - df_checkshot_plot2['std_checkshot']           
            fig1 = go.Figure()
            fig1.add_trace(go.Line(x=df_checkshot_plot2["twt picked"],y=df_checkshot_plot2['tvd_ss'],name="Sonic Log", marker_color='blue'))
            fig1.add_trace(go.Line(x=df_checkshot_plot2['u+std'],y=df_checkshot_plot2['tvd_ss'],fill=None,name="u+std",line_color="rgba(0,0,0,0)"))
            fig1.add_trace(go.Line(x=df_checkshot_plot2['u-std'],y=df_checkshot_plot2['tvd_ss'],fill='tonexty',name="u+std", line_color="rgba(0,0,0,0)"))

            fig1.update_layout(
            title=f'Checkshot data : Well {uwi}',
            xaxis_title="TWT (ms)",
            yaxis_title='TVDSS (m)',
            autosize=False,
            width=900,
            height=1800,
            yaxis_range=[max(df_checkshot_plot2["tvd_ss"]), min(df_checkshot_plot2["tvd_ss"])])
            
            st.plotly_chart(fig1)

    df_sonic_plot2 = df_sonic.copy(deep=True)
    
    with col4:
        container2 = st.container()
        with container2:
            std_sonic = st.slider("Standard deviation: Vp", 400, 600)
            df_sonic_plot2 = df_sonic_plot2.dropna(subset=['vp'])
            df_sonic_plot2['std_sonic'] = std_sonic   
            df_sonic_plot2['u+std'] = df_sonic_plot2['vp'] + df_sonic_plot2['std_sonic']
            df_sonic_plot2['u-std'] = df_sonic_plot2['vp'] - df_sonic_plot2['std_sonic']           
            fig2 = go.Figure()
            fig2.add_trace(go.Line(x=df_sonic_plot2['vp'],y=df_sonic_plot2['tvd_ss'],name="Sonic Log", marker_color='blue'))
            fig2.add_trace(go.Line(x=df_sonic_plot2['u+std'],y=df_sonic_plot2['tvd_ss'],fill=None,name="u+std",line_color="rgba(0,0,0,0)"))
            fig2.add_trace(go.Line(x=df_sonic_plot2['u-std'],y=df_sonic_plot2['tvd_ss'],fill='tonexty',name="u+std", line_color="rgba(0,0,0,0)"))
            
            fig2.add_trace(go.Line(x=td_vp,y=td_z, line_color='red', name='Vp Checkshot'))

            # Filling the area between the upper and lower bounds
            
            
            fig2.update_traces(connectgaps=True) 
            fig2.update_layout(
            title=f'Sonic Log data : Well {uwi}',
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



get_data()
    ###
    #first_year = gdp_df[gdp_df['Year'] == from_year]
    #last_year = gdp_df[gdp_df['Year'] == to_year]