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
from bayesian.td_tool.runBayesCsc_2 import Bayesian_Inference
from bayesian.td_tool.bayes_csc import getTime

from _2_Checkshot_Data import get_data, filter_data

raw_cks_df, df_checkshot, df_sonic = get_data()
wells = raw_cks_df.uwi.unique().tolist()
uwi = st.selectbox(f"Select Well", options=wells)
df_checkshot, df_sonic, df_merged = filter_data(df_checkshot, df_sonic, raw_cks_df, uwi)
st.write(f'Well Selected: {uwi}')
st.write('You can either run the bayesian inference with Standard values or select some of the parameters yourself.')

if st.button("Calculate Bayesian Inference"):
    # Your command goes here

    clas = Bayesian_Inference()
    bayes_csc_out, fig = clas.run(uwi)
    st.pyplot(fig)

    st.write("You can download the new time-depth relationship here:")
    st.download_button(
        label='Download CSV',
        data=bayes_csc_out['df_well'].to_csv(index=False),
        file_name='output.csv',
        mime='text/csv'
    )


