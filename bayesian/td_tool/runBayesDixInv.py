# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:33:52 2020

@author: disch
"""

import time
import numpy as np
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.getcwd())
from bayesian.td_tool.td_lib import extendLogsToZero
import matplotlib.pyplot as plt
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython import embed
from bayesian.td_tool.bayes_csc import Run_Bayesian, getDefaultPar, bayes_well_plot
import csv
import sys
import json
import streamlit as st
import time


class Bayesian_Inference:
    def __init__(self):
        pass

    def assert_data(self, df_checkshot, df_sonic):
        if df_checkshot.empty and df_checkshot is pd.DataFrame():
            raise ValueError("Checkshot data is empty.")
        if df_sonic.empty and len(df_sonic) >= 2:
            raise ValueError("Sonic data is empty.")

    def run(
        self,
        df_checkshot,
        df_sonic,
        std_sonic,
        std_checkshot,
        apply_covariance,
        corr_order,
        inversion_start_depth,
        decimation_step,
        uwi,
    ):
        ww = uwi
        ww = ww.replace(" ", "_").replace("/", "_")
        #####################
        # START: PARAMETERS #
        #####################
        runWell = 1  # flag to skip well
        self.assert_data(df_checkshot, df_sonic)
        # Turn on/off processes
        extend2zero = (
            1  # extend well log to z = 0 using seabed and analytical expression
        )

        # parameters
        water_velocity = 1478

        ###################
        # END: PARAMETERS #
        ###################

        df_sonic = df_sonic.rename(columns={"interval_velocity_sonic": "vp"})

        td = df_checkshot[
            ["tvd_ss", "time", "average_velocity", "interval_velocity", "depth_source"]
        ].dropna(subset="time")
        first_row_seabed = df_checkshot[
            df_checkshot["depth_source"].str.contains("seabed")
        ].iloc[0]

        df_sonic = df_sonic[["md", "tvd_ss", "vp"]].dropna(subset="vp")

        print("----")
        print("WELL: " + ww)
        td_z = td["tvd_ss"].values
        td_t = td["time"].values
        td_t = td_t * 0.001  # convert to seconds

        well_z_md = df_sonic["md"].values.astype(float)
        well_z = df_sonic["tvd_ss"].values.astype(float)
        well_vp = df_sonic["vp"].values.astype(float)

        df_well = pd.DataFrame({"md": well_z_md, "TVDMSL": well_z, "VP_IN": well_vp})

        water_depth = first_row_seabed["tvd_ss"]

        water_depth = float(water_depth)
        water_twt = first_row_seabed["time"]
        water_twt = float(water_twt)

        if extend2zero:
            print("EXTEND logs to z = 0")
            print("   #log: " + str(len(well_z)))
            well_z, well_vp = extendLogsToZero(
                well_z, well_vp, water_depth, water_velocity
            )
            print("   #log: " + str(len(well_z)))
            print("...done")

            # merge output with existing dataframe
            df_ext = pd.DataFrame({"TVDMSL": well_z, "VP_EXT": well_vp})
            df_well = pd.merge(df_well, df_ext, how="outer", on=["TVDMSL"])
            df_well = df_well.sort_values("TVDMSL").reset_index(drop=True)
        try:
            if runWell:
                # set parameters for bayesian check-shot correction
                par = getDefaultPar()
                par["apply_corr"] = 0
                par["istep_bayes"] = decimation_step
                par["std_vp_mode"] = 2
                par["std_vp_const"] = std_sonic
                par["std_t_const"] = std_checkshot
                if apply_covariance == "Apply":
                    par["apply_corr"] = 1
                elif apply_covariance == "Do not apply":
                    par["apply_corr"] = 0
                else:
                    par["apply_corr"] = 0
                    # these three were not activated

                par["corr_order"] = corr_order
                par["corr_range"] = 100

                par["zstart"] = inversion_start_depth
                try:
                    st.write(
                        f"Calculating Bayesian time-depth correction for {uwi}... This might take some time."
                    )

                    class_bayes = Run_Bayesian()

                    well_vp, well_z, C, std_total_depth = class_bayes.runCsc(
                        well_z, well_vp, td_z, td_t, par
                    )

                    print("...done")

                    df_bayes = pd.DataFrame({"TVDMSL": well_z, "VP_BAYES": well_vp})

                    df_well = pd.merge(df_well, df_bayes, how="outer", on=["TVDMSL"])
                    df_well = df_well.sort_values("TVDMSL").reset_index(drop=True)

                    print("Bayesian Inversion could be applied")

                except Exception as e:
                    print(e.args)
                    print("ERROR: Bayesian step failed")

        except:
            print(f"Bayesian inference could not be applied for well")
            pass
        return df_well, td_z, td_t, ww, water_depth, water_velocity, C, std_total_depth
