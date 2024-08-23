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
with col2:
    st.image('images/illustration-shows-equinor-logo.jpg')

col1, col2 = st.columns(2)
with col1:
    container1 = st.container()
    with container1:
        st.write("""
## OneCheckshot App

In this app you can compare checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. 
        It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.
         
""")
        st.write("""
## What are checkshots?

Checkshots are a geophysical technique used to determine the depth of a wellbore relative to the Earth's surface. This is achieved by measuring the travel time of seismic waves that are generated at the surface and recorded by geophones placed downhole.
         
""")

with col2:
    container1 = st.container()
    with container1:

        st.write("""
        ## SMDA in Equinor

        In this app you can compare checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. 
                It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.
                
        """)