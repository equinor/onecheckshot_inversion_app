import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
from IPython import embed
import plotly.graph_objects as go
import matplotlib.pyplot as plt


@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
    
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parents[1]/'data/checkshot_sonic_table_well.csv'
    raw_cks_df = pd.read_csv(DATA_FILENAME)
    df_checkshot = raw_cks_df[['uwi','tvd_ss','twt picked', 'average velocity', 'interval velocity']].dropna(subset='twt picked')
    df_sonic = raw_cks_df[['uwi','tvd_ss','vp', ]].dropna(subset='vp')

    return df_checkshot, df_sonic #gdp_df

df_checkshot, df_sonic = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Checkshot Data

SMDA Platform for Time-Depth relationship data extraction. (https://opus.smda.equinor.com/) SMDA website. 
'''

# Add some spacing
''

uwi = 'NO 15/6-9 S' #'NO 15/6-9 S', 'NO 16/2-1']
df_checkshot = df_checkshot[df_checkshot['uwi'] == uwi]


min_value = df_checkshot['tvd_ss'].min()
max_value = df_checkshot['tvd_ss'].max()


wells = df_checkshot['uwi'].unique()




if not len(wells):
    st.warning("Select at least one well")


fig = px.subplots(
    data_frame=data,
    subplot_titles=["First Subplot", "Second Subplot"],
    layout=dict(
        xaxis1_title="X-axis 1",
        yaxis1_title="Y-axis 1",
        xaxis2_title="X-axis 2",
        yaxis2_title="Y-axis 2"
    )
)



col1, col2 = st.columns([1, 2])

with col1:
    # Create your plot here
    fig1 = px.scatter(df_checkshot, x="twt picked", y="tvd_ss")
    fig1.update_layout(
    autosize=False,
    width=10,
    height=900,
    yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])])
     
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Create your plot here
    fig2 = px.scatter(df_sonic, x="vp", y="tvd_ss")  # Replace "x_column" and "y_column" with the appropriate column names from df_2   
    fig2.update_traces(connectgaps=False) 
    fig2.update_layout(
    autosize=False,
    width=10,
    height=900,
    yaxis_range=[max(df_checkshot["tvd_ss"]), min(df_checkshot["tvd_ss"])])
    
    st.plotly_chart(fig2, use_container_width=True)    


#st.plotly_chart(fig2, use_container_width=True)

''
#st.scatter_chart(df_checkshot, y='tvd_ss', x='twt picked',)

###
#first_year = gdp_df[gdp_df['Year'] == from_year]
#last_year = gdp_df[gdp_df['Year'] == to_year]