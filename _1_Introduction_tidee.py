#! \onecheckshot_inversion_app\.venv\Scripts\python.exe
import os
import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
from IPython import embed
import matplotlib.pyplot as plt

# py -m streamlit run _1_Introduction_tidee.py
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="tidee",
    page_icon=":earth_americas:",
    layout="wide",  # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

col1, col2 = st.columns(2)
with col1:
    st.write("# Welcome to Tidee App!")
    st.write(
        "SMDA Platform for Time-Depth relationship data extraction. (https://opus.smda.equinor.com/)"
    )
    st.write(
        """
## Tidee App

In this app you can compare Checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.

Apart from analyzing the quality procedures applied, you can also use an iterative platform to generate an automatic time-depth relationship from sonic log data calibrated with checkshot data. This method applies a Bayesian Dix Inversion approach that incorporates prior information (sonic log data) with new data (checkshot data) to estimate the interval velocity, a key parameter in time-depth conversion.
         
"""
    )
    st.write(
        """
## What are checkshots?
             
Checkshot is a geophysical technique employed to determine the seismic travel time from the surface to a specific depth within a borehole. Each individual measurement of the source-receiver travel time constitutes a checkshot.\
A checkshot survey can be defined as the collection and analysis of these travel time measurements to establish a time-depth calibration function.

Method: A geophone is lowered down the borehole while a seismic source at the surface generates a pulse of energy. The time it takes for the pulse to travel down the borehole and be recorded by the geophone is measured. This process is repeated at various depths.

Data: The checkshot provides direct measurements of the vertical travel time of seismic waves at specific depths.       
"""
    )
    st.image(os.path.join(os.getcwd(), "images/checkshot.jpg"), width=600)
with col2:
    st.image(os.path.join(os.getcwd(), "images/tidee_logo.jpg"))
    st.image(os.path.join(os.getcwd(), "images/equinor_blue.jpg"))
    st.write(
        """
    ## SMDA in Equinor

    In this app you can compare checkshot data with Sonic Log data available. This platform provides a centralized hub for visualizing and analyzing checkshot data sourced from SMDA and sonic log data. 
            It offers a comprehensive solution for researchers, engineers, and geologists to explore and interpret these geophysical datasets.
            
    """
    )
