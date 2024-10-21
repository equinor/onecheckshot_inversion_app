import plotly.graph_objects as go
import numpy as np
from IPython import embed
import pandas as pd
# Assuming you have a DataFrame 'df' with columns 'data', 'depth', 'qc1', 'qc2', 'qc3'
embed()
data = {
    'Column1': np.array([i for i in range(0,10)]),  # Random numbers from a standard normal distribution
    'Column2': np.array([i*(-1) for i in range(0,10)]),  # Sample letters
    'Column3': np.random.choice([True, False], size=10)  # Dates starting from January 1, 2023
}

df = pd.DataFrame(data)


fig = go.Figure()

# Primary plot (modify marker mode as needed)
fig.add_trace(go.Scatter(x=df['Column1'], y=df['Column2'], mode='markers', name='Data'))



# QC plot (Histogram for Column3)
fig.add_trace(go.Bar(
    x=['True', 'False'],
    y=data['Column3'],
    marker=dict(color=['green', 'red'])
))

# Customize layout
fig.update_layout(
    xaxis2=dict(anchor='y2', position=0.7),  # Adjust position of secondary x-axis
    yaxis2=dict(anchor='x2', position=1),  # Adjust position of secondary y-axis
    title='Data with Aligned QC Plot (Histogram for Column3)',
    showlegend=True
)

fig.show()


col1, col2 = st.columns(2)
with col1:
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        #st.write("Plot Checkshot data. It was performed a quality control for this data following https://github.com/equinor/qc_and_clean_cks.")
        fig1_1 = go.Figure()
        fig1_1.add_trace(go.Scatter(y=td["tvd_ss"],x=td["time"],mode="markers",marker=dict(color='blue',size=10),name='Checkshot data'))

        fig1_1.update_layout(
        title=f'Checkshot data : Well {uwi}',
        xaxis_title="TWT (ms)",
        yaxis_title='TVDSS (m)',
        autosize=False,
        width=400,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])])
        st.plotly_chart(fig1_1)
        
        fig1_2 = go.Figure()
        qc = ['md_increasing', 'tvd_ss_increasing', 'time_increasing']
        for col in qc:
            colors = ['green' if b=='true' else 'red' for b in td[col]]
            fig1_2.add_trace(go.Scatter(
                y=td["tvd_ss"],
                x=[col for i in range(len(td["tvd_ss"]))],
                mode="markers",
                marker=dict(
                    color=colors,
                    size=10),
                name=col
            ))

        #fig1_2 = px.scatter(td, x="md_increasing", y="tvd_ss")
        fig1_2.update_layout(
        autosize=False,
        width=10,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        showlegend=False
        )
        fig1_2.update_yaxes(visible=False, showticklabels=False)

    with col1_2:
        fig1_2 = go.Figure()
        qc = ['md_increasing', 'tvd_ss_increasing', 'time_increasing']
        for col in qc:
            colors = ['green' if b=='true' else 'red' for b in td[col]]
            fig1_2.add_trace(go.Scatter(
                y=td["tvd_ss"],
                x=[col for i in range(len(td["tvd_ss"]))],
                mode="markers",
                marker=dict(
                    color=colors,
                    size=10),
                name=col
            ))

        #fig1_2 = px.scatter(td, x="md_increasing", y="tvd_ss")
        fig1_2.update_layout(
        autosize=False,
        width=10,
        height=900,
        yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        showlegend=False
        )
        fig1_2.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig1_2)