import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
from IPython import embed
import matplotlib.pyplot as plt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='OneCheckshot',
    page_icon=':earth_americas:',
    layout="wide" # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

col1, col2 = st.columns(2)
with col1:
    st.write("# Welcome to SMDA OneCheckshot App!")
    st.write("SMDA Platform for Time-Depth relationship data extraction. (https://opus.smda.equinor.com/)")
    st.write("""
## OneCheckshot App

In this app you can compare checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. 
        It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.
         
""")
    st.write("""
## What are checkshots?

Checkshots are a geophysical technique used to determine the depth of a wellbore relative to the Earth's surface. 

Method: A geophone is lowered down the borehole while a seismic source at the surface generates a pulse of energy. The time it takes for the pulse to travel down the borehole and be recorded by the geophone is measured. This process is repeated at various depths.

Data: The checkshot provides direct measurements of the vertical travel time of seismic waves at specific depths.       
""")
    st.image('images/checkshot.jpg')
with col2:
    st.image('images/equinor_blue.jpg')
    st.write("""
    ## SMDA in Equinor

    In this app you can compare checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. 
            It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.
            
    """)

