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
from bayesian.td_tool.runBayesCsc_2 import Bayesian_Inference
from bayesian.td_tool.bayes_csc import getTime, getDrift
from bayesian.td_tool.td_lib import getVel
from pages._2_Checkshot_Data import get_data, filter_data

raw_cks_df, df_checkshot, df_sonic = get_data()
wells = raw_cks_df.uwi.unique().tolist()
uwi = st.selectbox(f"Select Well", options=wells)
df_checkshot, df_sonic, df_merged = filter_data(df_checkshot, df_sonic, raw_cks_df, uwi)
st.write(f'Well Selected: {uwi}')
st.write('You can either run the bayesian inference with Standard values or select some of the parameters yourself.')
clas = Bayesian_Inference()
df_well, td_z, td_t, ww, water_depth, water_velocity = clas.run(uwi='NO 16/2-1')
well_z = df_well['TVDMSL'].values
well_vp_ext = df_well['VP_EXT'].values
well_vp = df_well['VP_BAYES'].values

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

fig = go.Figure()

#, "VELOCITY DIFFERENCE", "TWT", "DRIFT (OWT)", "VELOCITY BAYESIAN"]

fig = make_subplots(rows=1, cols=3, subplot_titles=["VELOCITY","VELOCITY DIFFERENCE","TIME"])

yr = [np.max(np.union1d(well_z, td_z)) * 1.1, 0]
# Subplot 1: Velocity
fig.add_trace(go.Scatter(x=well_vp_ext, y=well_z, mode='lines', name='Well VP Extension', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Line(x=td_vp, y=td_z, line_shape='hv',line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[-1e4, 1e4], y=[water_depth, water_depth], mode='lines',line=dict(color='black', dash='dash')), row=1, col=1)
fig.update_yaxes(title_text="Y-axis 1", row=1, col=1, autorange='reversed')
fig.update_xaxes(title_text="yaxis 2 title", range=[0, np.max(np.union1d(well_vp, td_vp)) * 1.2], row=1, col=1)


# Subplot 2: Velocity Difference
xx = well_vp - well_vp_ext
xmin = np.min(xx) - 10
xmax = np.max(xx) + 10
fig.add_trace(go.Scatter(x=xx, y=td_z, mode='lines', name='Well VP Extension', line=dict(color='blue')), row=1, col=2)
fig.update_yaxes(title_text="yaxis 2 title", row=1, col=2,autorange='reversed')
fig.update_xaxes(title_text="yaxis 2 title", range=[xmin, xmax], row=1, col=2)



# Subplot 3: Time Plots
fig.add_trace(go.Line(x=1e3 * well_t_ext, y=well_z, line=dict(color='green')), row=1, col=3)
st.plotly_chart(fig)