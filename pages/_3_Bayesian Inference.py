from pathlib import Path
bayesian_folder = str(Path(__file__).parents[1]/'bayesian'/'td_tool')
pages_folder = str(Path(__file__).parents[1]/'pages')
import sys
sys.path.append(bayesian_folder)
sys.path.append(pages_folder)
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
from runBayesCsc_2 import Bayesian_Inference
from bayes_csc import getTime


from _2_Checkshot_Data import get_data, return_well, filter_data

raw_cks_df, df_checkshot, df_sonic = get_data()
wells = raw_cks_df.uwi.unique().tolist()
selected_value = st.selectbox(f"Select Well", options=wells)
uwi = selected_value
df_checkshot, df_sonic, df_merged = filter_data(df_checkshot, df_sonic, raw_cks_df, uwi)

st.write(uwi)



clas = Bayesian_Inference()
bayes_csc_out, fig = clas.run(uwi)

st.pyplot(fig)

embed()