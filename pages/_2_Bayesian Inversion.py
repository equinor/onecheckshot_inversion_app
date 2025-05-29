import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
from IPython import embed
import matplotlib.pyplot as plt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="OneCheckshot",
    page_icon=":earth_americas:",
    layout="wide",  # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.
st.write("# Sonic log calibration using checkshot data")
st.write("## Introduction")

st.write(
    """ #### Checkshot data is used to calibrate sonic log data, which is essential for accurate depth conversion. By comparing the travel times recorded by the checkshot with those measured by the sonic log, geophysicists can determine the depth shift needed to align the sonic log data with the actual depth. The output of this calibration is a time-depth relationship that will be essential to accurately tie the wellbore data in the time domain. This tie ensures that depth-related properties measured by the sonic log, such as porosity and lithology, are accurately correlated with the true depth in the borehole.
                
#### The traditional method for sonic log calibration involves a uniform drift correction applied to the entire log. To begin, sonic log data is plotted against checkshot data, allowing for a visual comparison. The difference between these two datasets at each depth is calculated to quantify the drift. This calculated drift is then applied to the sonic log data, either by adding or subtracting it from the readings. This adjustment aims to correct the sonic log for any deviations from the true formation velocity.

#### Despite its widespread use, the traditional sonic log calibration method assumes a constant drift throughout the well. However, factors like borehole conditions, temperature fluctuations, and tool calibration can introduce variations in drift. This can lead to inaccurate corrections in specific depth intervals. Moreover, complex drift patterns, such as those arising from non-linear changes in borehole conditions or tool response, may not be adequately addressed by a uniform correction, potentially resulting in residual errors in the corrected sonic log data.

#### To overcome these limitations, more advanced calibration techniques can be employed, such as statistical methods, wavelet-based approaches and machine learning algorithms. On this application it is proposed a probabilitic approach for sonic log calibration using checkshot data developed by Dischler et al. (2013), based on Buland et al. (2011). Bayesian Dix Inversion provides a robust and flexible framework for shifting sonic logs using checkshot data. It allows for the incorporation of prior knowledge, uncertainty quantification, and the handling of complex geological scenarios.            
    """
)

col1, col2 = st.columns(2)
with col1:
    st.write("## Bayesian Dix Inversion method")
    st.write(
        """ #### Bayesian inversion can be a method to apply sonic log calibration on the sonic log using checkshot data. It provides a framework for combining an attributed prior knowledge (e.g., interval velocity and uncertainty from sonic log) with new data (the checkshot measurements) to estimate a more accurate posterior velocity trend. This allows for a quantitative assessment of the uncertainty associated with the adjusted time-depth relationship curve.

#### In this case, both the the sonic log and checkshot data can be modeled as uncertain data, but the uncertainty on sonic log is higher. Sonic logs are typically more susceptible to noise and interference compared to checkshot data, which is often considered more reliable for depth calibration. By doing so, the sonic log curve can be shifted towards the checkshot data points, and an uncertainty can be estimated using bayesian inversion.             
"""
    )

    st.write(
        """
## Methodology
#### The workflow for generating the updated time-depth relationship is illustrated on Figure x (Source?). The first step is to extend to seabed, which can be detailed using velocity trends estimated from other log data (density log for example) and using general theoretical trends. Once the velocity trend is created, Bayes' theorem can be applied to correct it using sonic log. The output is a continuous high-resolution velocity log and time-depth curve from z=0."""
    )
    st.image("images/methodology.jpg")
    st.write(
        """#### The Bayes' Theorem can be described in the application of this workflow with equation x (Dischler et al. 2013). It involves formulating a prior probability distribution for the model, p(m), which is here interval velocity from sonic log, and then regard the measured checkshot points as a set of data, d. A likelihood function, p(d|m), that represents the conditional probability of measuring a given set of data d given the model m is defined by assuming a measurement error, e. By Bayes’rule the inversion solution is obtained as a posterior probability function

## p(m|d) ∝ p(d|m)p(m).
             
###### p(m|d) - A posteriori velocity trend given checkshot data
###### p(d|m) - Likelihood function
###### p(m) - Sonic Log interval Velocity interpolated up to the surface             

#### In the context of applying shift to sonic log data using checkshot data, the forward model is a simple linear relationship that assumes the checkshot data is equal to the sonic log data plus a constant shift. Because the solution is of an explicit analytic form, the bayesian inversion is computationally fast and does not require stochastic simulation (Dischler et al., 2013).
"""
    )


with col2:
    st.write(
        """
## Advantages
             
#### Bayesian inversion offers a robust and flexible framework for shifting sonic data using checkshot data. It allows for the quantification of uncertainty in the estimated shift and \
incorporates prior knowledge. Additionally, Bayesian methods are less sensitive to outliers,\
providing more reliable results. The posterior distributions obtained can offer insights into the relationship between sonic and checkshot data, aiding on \
interpretation. Finally, Bayesian inference can handle large datasets and complex models, making it a scalable and powerful approach for sonic data calibration applications.

### Some direct advantages include:

#### Uncertainty Quantification: Bayesian inference provides a posterior covariance matrix, which quantifies the uncertainty in the estimated sonic log values using checkshot data.​

#### Model Complexity: Bayesian methods can handle complex models with multiple parameters and non-linear relationships, making them more adaptable to various geological conditions.​
             
#### Sensibility to outliers: Probabilistic methods can be less sensitive to outliers in the data, reducing the influence of erroneous checkshot measurements or anomalies in the sonic log.

#### High Impact: Ready-to-use velocity trends for multiple purposes: Synthetic well seismic, well-tie, depth conversion...

      
"""
    )
    st.image("images/impact.jpg")


st.write(
    """ ## References\
         
Buland, A., Kolbjørnsen, O., & Carter, A. J. (2011). Bayesian dix inversion. Geophysics, 76(2), R15-R22.
         
Dischler, E., Hokstad, K., & Buland, A. (2013). Bayesian anisotropic Dix inversion. In SEG Technical Program Expanded Abstracts 2013 (pp. 4853-4857). Society of Exploration Geophysicists."""
)
