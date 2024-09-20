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
st.write("# Sonic log calibration using checkshot data")
st.write("## Introduction")
st.write("""Checkshot data is used to calibrate sonic log data, which is essential for accurate depth conversion. By comparing the travel times recorded by the checkshot with those measured by the sonic log, geophysicists can determine the depth shift needed to align the sonic log data with the actual depth. The output of this calibration is a time-depth relationship (TDR) that will be essential to accurately tie the wellbore data in the time domain. This tie ensures that depth-related properties measured by the sonic log, such as porosity and lithology, are accurately correlated with the true depth in the borehole.
             
The traditional method for sonic log calibration involves a uniform drift correction applied to the entire log. To begin, sonic log data is plotted against checkshot data, allowing for a visual comparison. The difference between these two datasets at each depth is calculated to quantify the drift. This calculated drift is then applied to the sonic log data, either by adding or subtracting it from the readings. This adjustment aims to correct the sonic log for any deviations from the true formation velocity.

Despite its widespread use, the traditional sonic log calibration method assumes a constant drift throughout the well. However, factors like borehole conditions, temperature fluctuations, and tool calibration can introduce variations in drift. This can lead to inaccurate corrections in specific depth intervals. Moreover, complex drift patterns, such as those arising from non-linear changes in borehole conditions or tool response, may not be adequately addressed by a uniform correction, potentially resulting in residual errors in the corrected sonic log data.

To overcome these limitations, more advanced calibration techniques can be employed, such as statistical methods, wavelet-based approaches and machine learning algorithms. In the framework of this project we propose a sonic log calibration from checkshot data using a probabilistic approach. Bayesian inference provides a robust and flexible framework for shifting sonic logs using checkshot data. It allows for the incorporation of prior knowledge, uncertainty quantification, and the handling of complex geological scenarios.            
""")
col1, col2 = st.columns(2)
with col1:
    st.write("## Bayesian Inversion method") 
    st.write("""Bayesian inversion can be a method to apply drift on the sonic log using checkshot data. It provides a framework for combining an attributed prior knowledge (e.g., about the sonic log and checkshot uncertainties) with new data (the checkshot measurements) to estimate the posterior probability distribution of the sonic log's depth scale. This allows for a quantitative assessment of the uncertainty associated with the adjusted time-depth relationship curve.

In this case, both the the sonic log and checkshot data can be modeled as uncertain data, but the uncertainty on sonic log is higher. Sonic logs are typically more susceptible to noise and interference compared to checkshot data, which is often considered more reliable for depth calibration. By doing so, the sonic log curve can be shifted towards the checkshot data points, and an uncertainty can be estimated using bayesian inference.             

In the context of applying shift to sonic log data using checkshot data, the forward model is a simple linear relationship that assumes the checkshot data is equal to the sonic log data plus a constant shift.
""")
    
    st.write("""
## Generation of time-depth relationship

Combination of velocity logs with checkshots to provide high-resolution, full coverage and imply the corrected seismic time. The aim of this method is to perform Bayesian inference and to estimate posterior distribution of a model (time-depth relationship) given the data (checkshot data) and a prior distribution (sonic log data).""")
    st.image('images/bayesian_formula.png', width=400, use_column_width=False)
    st.write("""P(ϴ | data) is the posterior distribution of the shift (ϴ) parameter given the data.
             
P(data | ϴ) is the likelihood function, representing the probability of observing the data given a specific shift.
             
P(ϴ) is the prior distribution of the shift parameter.
             
P(data) is the marginal likelihood, which acts as a normalization constant.""")

with col2:

    st.write("""
## Advantages
             
Bayesian inversion offers a robust and flexible framework for shifting sonic data using checkshot data. It allows for the quantification of uncertainty in the estimated shift, 
             incorporates prior knowledge, and can handle hierarchical structures in the data. Additionally, Bayesian methods are less sensitive to outliers, 
             providing more reliable results. The posterior distributions obtained can offer insights into the relationship between sonic and checkshot data, aiding in 
             interpretation. Finally, Bayesian inference can handle large datasets and complex models, making it a scalable and powerful approach for sonic data shift applications.

Some direct advantages include:

Uncertainty Quantification: Bayesian inference provides a posterior covariance matrix, which quantifies the uncertainty in the estimated sonic log values using checkshot data.​

Model Complexity: Bayesian methods can handle complex models with multiple parameters and non-linear relationships, making them more adaptable to various geological conditions.​
             
Sensibility to outliers: Probabilistic methods can be less sensitive to outliers in the data, reducing the influence of erroneous checkshot measurements or anomalies in the sonic log.

High Impact: Ready-to-use velocity trends for multiple purposes: Synthetic well seismic, well-tie, depth conversion...

      
""")
st.write("""
## Methodology

       
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