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
    st.write("# Sonic log correction using checkshot data")
    st.write("Traditionally, sonic log data shift using checkshot data has been performed using relatively simple methods that rely on manual adjustments and assumptions.")

    st.write("## Bayesian Inference method")
    st.write("Bayesian inference can be a method to apply drift on the sonic log using checkshot data. The aim is to have a time-depth relationship at disposal for SMDA users.")
    st.write("""
## Generation of time-depth relationship

Combination of velocity logs with checkshots to provide high-resolution, full coverage and imply the corrected seismic time. The aim of this method is to perform Bayesian inference and to estimate posterior distribution of a model (time-depth relationship) given the data (checkshot data) and a prior distribution (sonic log data).""")
    st.image('images/bayesian_formula.png')        

with col2:
    
    st.write("""
## Advantages
             
Bayesian inference offers a robust and flexible framework for shifting sonic data using checkshot data. It allows for the quantification of uncertainty in the estimated shift, 
             incorporates prior knowledge, and can handle hierarchical structures in the data. Additionally, Bayesian methods are less sensitive to outliers, 
             providing more reliable results. The posterior distributions obtained can offer insights into the relationship between sonic and checkshot data, aiding in 
             interpretation. Finally, Bayesian inference can handle large datasets and complex models, making it a scalable and powerful approach for sonic data shift applications.

Uncertainty Quantification: Bayesian inference provides a posterior covariance matrix, which quantifies the uncertainty in the estimated sonic log values using checkshot data.​

Model Complexity: Bayesian methods can handle complex models with multiple parameters and non-linear relationships, making them more adaptable to various geological conditions.​

HIgh Impact: Ready-to-use velocity trends for multiple purposes: Synthetic well seismic, well-tie, depth conversion...

      
""")
st.write("""
## Methodology

Checkshots are a geophysical technique used to determine the depth of a wellbore relative to the Earth's surface. 

Method: A geophone is lowered down the borehole while a seismic source at the surface generates a pulse of energy. The time it takes for the pulse to travel down the borehole and be recorded by the geophone is measured. This process is repeated at various depths.

Data: The checkshot provides direct measurements of the vertical travel time of seismic waves at specific depths.         
""")
col1, col2, col3 = st.columns(3)
with col1:
    st.write('Step 1: Data quality verification')
    st.image('images/methodology_1.jpg') 
with col2:
    st.write('Step 2: Data extrapolation from Sonic log to seabed. Then, checkshot data is estimated in the velocity domain and sonic log is transformed to the time domain')
    st.image('images/methodology_2.jpg')
with col3:
    st.write('Step 3: Application of Bayesian Inference method. It is possible to verify that the drift has diminished as the new time-depth relationship contains information from checkshot data.')
    st.image('images/methodology_3.jpg')