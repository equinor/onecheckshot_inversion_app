from pathlib import Path

import sys
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
import os

sys.path.append(os.getcwd())
from bayesian.td_tool.bayes_csc import getTime
from bayesian.td_tool.td_lib import getVel

# import bayesian.td_tool.ELS_log
import pandas as pd
from bayesian.td_tool.data_management_smda.connect_smda import (
    Connection_Database,
    get_connect_database,
    generate_df,
)
from scipy.interpolate import interp1d
from bayesian.td_tool.ELS_log_API import (
    Connection_ELS_LOG,
    load_els_data,
    load_els_data_fmb,
)
from bayesian.td_tool.functions import return_style_table, decimate_dataframe
from bayesian.td_tool.smda_api.els_api import get_api

st.set_page_config(layout="wide")
# raw_cks_df, df_checkshot, df_sonic = get_data()
host, dbname, user, password, sslmode = get_connect_database()
connect = Connection_Database(host, dbname, user, password, sslmode)

# database_checkshot = "smda.smda_workspace.wellbore_checkshot_data_qc"
database_checkshot = "smda.smda_workspace.wellbore_checkshot_data"
database_dsa = "smda.smda_workspace.dsa_wellbore_checkshot_header"
wells = connect.get_wells(database_checkshot)


st.write("## Select Data")


def select_well():
    selected_value = st.selectbox(f"Select Checkshot Wellbore", options=sorted(wells))
    return selected_value


uwi = select_well()

st.write(f"Well selected: {uwi}")

host, dbname, user, password, sslmode = get_connect_database()

columns = "md, md_unit, tvd, tvd_ss, tvd_unit, depth_reference_elevation, depth_source, time, time_unit, source_file, unique_wellbore_identifier, average_velocity, interval_velocity, qc_description, md_increasing, tvd_ss_increasing, time_increasing, average_velocity_qc, trajectory_checked, comparison_sonic_log_qc, preference_checkshotfile"
df = generate_df(
    host, dbname, user, password, sslmode, columns, database_checkshot, uwi
)
df_dsa = generate_df(host, dbname, user, password, sslmode, "*", database_dsa, uwi)
connect.close_connection()
seabed = df.loc[df["depth_source"].str.contains("seabed", case=False), "tvd_ss"].astype(
    float
)


col1, col2 = st.columns(2)
with col1:
    selected_source = st.selectbox(
        f"Select Checkshot File", options=df["source_file"].unique()
    )

    df = df[
        (df["source_file"] == selected_source)
        & (df["md_increasing"] == "true")
        & (df["time_increasing"] == "true")
    ]
    st.write(
        "Obs: Data points with decreasing MD or Time will be removed from the dataset."
    )
    df.sort_values(by=["md"], ascending=[True], inplace=True)


with col2:
    selected_source_well_log = st.selectbox(
        f"Select Sonic log File Source", options=["LFP", "FMB"]
    )

    if "connection_ELS" not in st.session_state:
        st.session_state.connection_ELS = False

    @st.cache_data
    def api_data(uwi, selected_source_well_log):
        response, msg = get_api(
            uwi=uwi, selected_source_well_log=selected_source_well_log
        )
        return response, msg

    response, msg = api_data(uwi, selected_source_well_log)

    if selected_source_well_log == "LFP":
        try:

            @st.cache_data
            def load_els_log(uwi):
                column_names = [
                    "MD",
                    "TVDMSL",
                    "LFP_VP_V",
                    "LFP_VP_LOG",
                    "LFP_VP_G",
                    "LFP_VP_O",
                    "LFP_VP_B",
                ]
                df_sonic_els = pd.DataFrame(response[0]["data"], columns=column_names)
                api_connection = "yes"
                return df_sonic_els

            df_sonic_els = load_els_log(uwi)
            options_log = ["LFP_VP_V", "LFP_VP_B", "LFP_VP_G", "LFP_VP_O", "LFP_VP_LOG"]
            selected_log_curve = st.selectbox(
                f"Select Sonic Log Curve", options=options_log
            )
            df_sonic = load_els_data(df_sonic_els, selected_log_curve)
            df_sonic = df_sonic.sort_values(by=["md"])

        except Exception as e:
            df_sonic = pd.DataFrame()
            api_connection = "no"
    elif selected_source_well_log == "FMB":
        try:

            @st.cache_data
            def load_fmb_log(uwi):
                column_names = ["MD", "TVDMSL", "DT"]
                df_sonic_fmb = pd.DataFrame(response[0]["data"], columns=column_names)
                return df_sonic_fmb

            df_sonic_fmb = load_fmb_log(uwi)
            options_log = ["DT"]
            selected_log_curve = st.selectbox(
                f"Select Sonic Log Curve", options=options_log
            )
            df_sonic = load_els_data_fmb(df_sonic_fmb, selected_log_curve)
            # st.write("No data availabe for FMB. App still testing")
            # df_sonic = df_sonic_fmb

        except Exception as e:
            df_sonic = pd.DataFrame()
            api_connection = "no"


col1, col2 = st.columns(2)
with col1:
    st.write(
        df[
            [
                "md",
                "tvd",
                "tvd_ss",
                "tvd_unit",
                "md_unit",
                "depth_source",
                "depth_reference_elevation",
                "time",
                "time_unit",
                "average_velocity",
                "average_velocity_qc",
                "interval_velocity",
                "qc_description",
            ]
        ]
    )
with col2:
    st.write(df_sonic)

td = df[
    [
        "md",
        "tvd",
        "tvd_ss",
        "depth_reference_elevation",
        "time",
        "qc_description",
        "average_velocity",
        "interval_velocity",
        "md_increasing",
        "tvd_ss_increasing",
        "time_increasing",
        "average_velocity_qc",
        "trajectory_checked",
        "depth_source",
    ]
]


col1, col2 = st.columns(2)
with col1:
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        fig1_1 = go.Figure()
        fig1_1.add_trace(
            go.Scatter(
                y=td["tvd_ss"],
                x=td["time"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Checkshot data",
            )
        )

        fig1_1.update_layout(
            title=f"Checkshot data : Well {uwi}",
            xaxis_title="TWT (ms)",
            yaxis_title="TVDSS (m)",
            autosize=False,
            width=800,
            height=800,
            yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        )
        st.plotly_chart(fig1_1)

    with col1_2:
        fig1_2 = go.Figure()
        qc = [
            "md_increasing",
            "tvd_ss_increasing",
            "time_increasing",
            "average_velocity_qc",
            "trajectory_checked",
        ]
        for col in qc:
            colors = [
                "green"
                if (b == "true" or b == "passed")
                else "gray"
                if (b == "not checked")
                else "red"
                for b in td[col]
            ]
            fig1_2.add_trace(
                go.Scatter(
                    y=td["tvd_ss"],
                    x=[col for i in range(len(td["tvd_ss"]))],
                    mode="markers",
                    marker=dict(color=colors, size=10),
                    name=col,
                )
            )

        # fig1_2 = px.scatter(td, x="md_increasing", y="tvd_ss")
        fig1_2.update_layout(
            xaxis=dict(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=["MD", "TVDSS", "Time", "Average vel", "Trajectory"],
            ),
            autosize=False,
            width=10,
            height=800,
            yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
            showlegend=False,
        )
        fig1_2.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig1_2)

st.write("## Onecheckshot DSA QC")
columns_dsa = [
    "unique_wellbore_identifier",
    "source_file",
    "md_data_missing",
    "tvd_ss_data_missing",
    "twt_data_missing",
    "twt_higher_than_max",
    "not_enough_stations",
    "incorrect_datum",
    "station_density_below_cutoff",
    "uniqueness_below_cutoff",
    "tvd_ss_not_increasing",
    "twt_not_increasing",
    "high_average_velocities",
    "low_average_velocities",
    "mean_time_drift",
    "sonic_log_source",
    "description",
    "insert_date",
]
styled_df = return_style_table(df_dsa[columns_dsa])
st.dataframe(styled_df, width=None)
# st.dataframe(display_table(df_dsa[columns_dsa]))


with col2:
    try:
        fig2 = px.line(
            df_sonic, x="interval_velocity_sonic", y="tvd_ss"
        )  # Replace "x_column" and "y_column" with the appropriate column names from df_2
        fig2.update_traces(connectgaps=False)
        fig2.update_layout(
            title=f"Sonic Log data : Well {uwi}",
            xaxis_title="Vp (m/s)",
            yaxis_title="TVDSS (m)",
            autosize=False,
            width=500,
            height=800,
            yaxis_range=[max(td["tvd_ss"]), min(td["tvd_ss"])],
        )

        st.plotly_chart(fig2)
    except:
        pass

    df_checkshot_plot2 = td.copy(deep=True)


col1, col2 = st.columns(2)
with col1:
    col1_1, col2_1 = st.columns(2)
    with col1_1:
        decimate_step = int(
            st.text_input(f"Enter Decimation Step for Checkshot Data:", 0)
        )
        if decimate_step == 0:
            pass
        else:
            td_seabed_sealevel = td[
                (
                    td["depth_source"].str.contains("seabed")
                    | (td["depth_source"].str.contains("sealevel"))
                )
            ]
            td_to_decimate = td[
                ~(
                    td["depth_source"].str.contains("seabed")
                    | (td["depth_source"].str.contains("sealevel"))
                )
            ]
            td_decimate = decimate_dataframe(td_to_decimate, decimate_step)
            td = pd.concat([td_seabed_sealevel, td_decimate])
    with col2_1:
        if "checkshot_values_to_exclude" not in st.session_state:
            st.session_state.checkshot_values_to_exclude = []
        try:
            delete_checkshot_point = float(
                st.text_input(f"Enter Checkshot depth point to exclude:", -999.25)
            )
            st.write(
                "TVDSS point selected to be excluded:",
                delete_checkshot_point,
                ". Click on the bottom below to append it to the exclusion list.",
            )

        except:
            st.write("Only numbers are allowed")

        if st.button("Append"):
            if delete_checkshot_point != -999.25:
                try:
                    if (
                        delete_checkshot_point
                        not in st.session_state.checkshot_values_to_exclude
                    ):
                        st.session_state.checkshot_values_to_exclude.append(
                            delete_checkshot_point
                        )
                except:
                    st.write("Please enter an accurate TVDSS point to be excluded")
        if st.button("Clear values to exclude"):
            st.session_state.checkshot_values_to_exclude = []
        # st.write(st.session_state.checkshot_values_to_exclude)
        st.write(
            "Depth points to exclude:",
            str([x for x in st.session_state.checkshot_values_to_exclude]),
        )
        if st.button("Exclude Values"):
            if len(st.session_state.checkshot_values_to_exclude) != 0:
                td = td[
                    ~td["tvd_ss"].isin((st.session_state.checkshot_values_to_exclude))
                ]
                st.session_state["Checkshot"] = td


with col2:
    try:
        depth_sonic = float(
            st.text_input(
                f"Enter depth (m) to start sonic from:", df_sonic["tvd_ss"].iloc[0]
            )
        )
        df_sonic = df_sonic[df_sonic["tvd_ss"] >= depth_sonic]
    except:
        pass

col3, col4 = st.columns(2)
with col3:
    container1 = st.container()
    with container1:
        # Create your plot here
        st.write("## Time Domain")
        if df_dsa["evaluated_against_sonic"].iloc[0]:
            st.write(
                f"Time drift between Checkshot and Sonic log ({df_dsa['sonic_log_source'].iloc[0]}): {df_dsa['mean_time_drift'].iloc[0]} ms"
            )
        else:
            pass
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                y=td["tvd_ss"],
                x=td["time"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Checkshot data",
            )
        )
        try:
            # TWT for sonic data
            td_tointerpolate = td[["tvd_ss", "time"]].dropna()
            interp_func = interp1d(
                td_tointerpolate["tvd_ss"].astype(float),
                td_tointerpolate["time"],
                kind="linear",
            )
            first_depth_sonic = float(np.array(df_sonic["tvd_ss"])[0])
            first_time_sonic = float(interp_func(first_depth_sonic))
            dz = np.diff(np.array(df_sonic["tvd_ss"].astype(float))) * 1000
            dz = np.insert(dz, 0, 0)
            dt = 2 * dz / np.array(df_sonic["interval_velocity_sonic"])
            t = np.cumsum(dt) + first_time_sonic
            fig1.add_trace(
                go.Line(
                    x=t,
                    y=df_sonic["tvd_ss"].astype(float),
                    name=f"Sonic Log {selected_log_curve}",
                    marker_color="blue",
                )
            )
        except:
            st.write("No Sonic log available for this well on ELS API.")

        fig1.update_layout(
            title=f"#Time Domain",
            xaxis_title="TWT (ms)",
            yaxis_title="TVDSS (m)",
            autosize=False,
            width=900,
            height=1800,
            yaxis_range=[
                max(df_checkshot_plot2["tvd_ss"]),
                min(df_checkshot_plot2["tvd_ss"]),
            ],
        )


df_sonic_plot2 = df_sonic.copy(deep=True)


if isinstance(td, pd.DataFrame):
    # get data
    td_z = td["tvd_ss"].values
    td_t = td["time"].values
    td_t = td_t * 0.001
    td_vp = td["interval_velocity"].values


else:
    print("ERROR: empty time depth - skipping well")
    runWell = 0  # skip this well

with col4:
    container2 = st.container()
    with container2:
        st.write("## Velocity Domain")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Line(
                x=td_vp, y=td_z, line_color="red", name="Vp Checkshot", line_shape="hv"
            )
        )
        # fig2.add_trace(go.Scatter(x=[0, 3], y=[float(seabed), float(seabed)], mode='lines',line=dict(color='black', dash='dash')))

        try:
            df_sonic_plot2 = df_sonic_plot2.dropna(subset=["interval_velocity_sonic"])
            fig2.add_trace(
                go.Line(
                    x=df_sonic_plot2["interval_velocity_sonic"],
                    y=df_sonic_plot2["tvd_ss"],
                    name="Sonic Log",
                    marker_color="blue",
                )
            )
            fig2.add_hline(
                y=float(seabed), line_width=8, line=dict(color="black", dash="dash")
            )
            # Filling the area between the upper and lower bounds
        except:
            pass

        fig2.update_traces(connectgaps=True)
        fig2.update_layout(
            title=f"Velocity Domain",
            xaxis_title="Vp (m/s)",
            yaxis_title="TVDSS (m)",
            autosize=False,
            width=900,
            height=1800,
            yaxis_range=[
                max(df_checkshot_plot2["tvd_ss"]),
                min(df_checkshot_plot2["tvd_ss"]),
            ],
            showlegend=True,
        )

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.add_trace(fig1["data"][0], row=1, col=1)
fig.add_trace(fig2["data"][0], row=1, col=2)

try:
    fig.add_trace(fig1["data"][1], row=1, col=1)
    fig.add_trace(fig2["data"][1], row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=[
                fig2["data"][0]["x"][0],
                fig2["data"][0]["x"][-1],
            ],  # Extend the line across the x-axis
            y=[
                float(seabed),
                float(seabed),
            ],  # Set the y-coordinate for the horizontal line
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Seabed",  # Customize the line style
        ),
        row=1,
        col=2,
    )
except:
    pass
fig.update_xaxes(title_text="Two-way Travel Time (TWT) (ms)", row=1, col=1)
fig.update_xaxes(title_text="Interval Velocity (m/s)", row=1, col=2)
fig.update_layout(
    height=800,
    width=1300,
    yaxis_range=[max(df_checkshot_plot2["tvd_ss"]), min(df_checkshot_plot2["tvd_ss"])],
)
st.plotly_chart(fig, use_container_width=True)

if st.button("Save Checkshot and Sonic data:"):
    try:
        if st.session_state.checkshot_values_to_exclude:
            pass  # Checkshot was already defined

        else:
            st.session_state["Checkshot"] = td
        st.session_state["seabed"] = seabed
        st.write(f"Checkshot file saved for well {uwi} saved")
    except:
        st.write(f"No Checkshot file for well {uwi}")
    try:
        st.session_state["Sonic_log"] = df_sonic
        if not df_sonic.empty:
            st.write(f"Sonic log saved for well {uwi}")
        else:
            pass
            # st.write(f'No sonic log file for well {uwi}')
    except:
        print("d")
    st.session_state["uwi"] = uwi
else:
    st.write("Files not yet saved...")
