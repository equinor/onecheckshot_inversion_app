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
import time
#from pages._3_Checkshot_Data import get_data, filter_data

import streamlit as st

# This doesn't work, because button "pressed" state doesn't survive rerun, and pressing
# any button triggers a rerun.


import SessionState

button1 = st.empty()
text1 = st.empty()
button2 = st.empty()
text2 = st.empty()

ss = SessionState.get(button1 = False)

if button1.button('1') or ss.button2:
    ss.button1 = True

if ss.button1:
    text1.write('you clicked the first button')
    if button2.button('2'):
        text2.write('you clicked the second button')