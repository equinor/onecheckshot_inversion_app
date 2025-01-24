import os
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
#from kustoconnectionstringbuilder import KustoConnectionStringBuilder
from azure.kusto.data.helpers import dataframe_from_result_table
import time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# import datashader as ds, colorcet


def log_in():
    print('CURRENT DIRECTORY',os.getcwd())
    os.environ['REQUESTS_CA_BUNDLE'] = 'ca-bundle.crt'
    AAD_TENANT_ID = "3aa4a235-b6e2-48d5-9195-7fcf05b459b0"

    CLIENT_ID = "a721bf21-d9a7-4f6a-9a42-a6c65dec77c3"
    KUSTO_CLUSTER = "https://dec-sdf-dh-prod-ne.northeurope.kusto.windows.net"
    KUSTO_DATABASE = "ELS"
    KCSB = KustoConnectionStringBuilder.with_interactive_login(KUSTO_CLUSTER)
    #KCSB = KustoConnectionStringBuilder.with_aad_managed_service_identity_authentication(KUSTO_CLUSTER, client_id=AAD_TENANT_ID)
    #KCSB = KustoConnectionStringBuilder.with_az_cli_authentication(KUSTO_CLUSTER)
    #KCSB.authority_id = AAD_TENANT_ID
    KUSTO_CLIENT = KustoClient(KCSB)

    print('CONNECTION ELS')
    return KUSTO_CLIENT    

class Connection_ELS_LOG:
    def __init__(self):
        print('CURRENT DIRECTORY',os.getcwd())
        #os.environ['REQUESTS_CA_BUNDLE'] = 'ca-bundle.crt'
        #self.AAD_TENANT_ID = "3aa4a235-b6e2-48d5-9195-7fcf05b459b0"
        #self.KUSTO_CLUSTER = "https://dec-sdf-dh-prod-ne.northeurope.kusto.windows.net"
        #self.KUSTO_DATABASE = "ELS"
        #self.KCSB = KustoConnectionStringBuilder.with_interactive_login(self.KUSTO_CLUSTER)
        #self.KCSB.authority_id = self.AAD_TENANT_ID
        #self.KUSTO_CLIENT = KustoClient(self.KCSB)
        print('CONNECTION ELS')
    def kusto_query_LFP(self, KUSTO_CLIENT_TEST, uwi):

        KUSTO_QUERY = f"""
            els_logdata_lfp
            | where unique_wellbore_identifier == '{uwi}'
            | where isnotnull(MD) and isnotnull(TVDMSL)
            | distinct unique_wellbore_identifier, MD, TVDMSL, LFP_VP_B, LFP_VP_LOG, LFP_VP_G, LFP_VP_O, LFP_VP_V
        """
        #RESPONSE = self.KUSTO_CLIENT.execute(self.KUSTO_DATABASE, KUSTO_QUERY)
        RESPONSE = KUSTO_CLIENT_TEST.execute("ELS", KUSTO_QUERY)
        df = dataframe_from_result_table(RESPONSE.primary_results[0])
        return df
    
    def kusto_query_FMB(self, KUSTO_CLIENT_TEST, uwi):
        KUSTO_QUERY = f"""
            ls_logdata_fmb_1011
            | where unique_wellbore_identifier == '{uwi}'
            | where isnotnull(MD) and isnotnull(TVDMSL)
            | distinct unique_wellbore_identifier, MD, TVDMSL, DT
        """
        #RESPONSE = self.KUSTO_CLIENT.execute(self.KUSTO_DATABASE, KUSTO_QUERY)
        RESPONSE = KUSTO_CLIENT_TEST.execute("ELS", KUSTO_QUERY)        
        df = dataframe_from_result_table(RESPONSE.primary_results[0])
        return df

def load_els_data(df_sonic, selected_log_curve):

    try:
        df_well_log = df_sonic.rename(columns={'TVDMSL':'tvd_ss', 'MD':'md'})
        df_well_log = df_well_log[['unique_wellbore_identifier','md', 'tvd_ss', f"{selected_log_curve}"]]
        df_well_log = df_well_log.dropna()        
        df_well_log['interval_velocity_sonic'] = df_well_log[f"{selected_log_curve}"]
        return df_well_log

    except Exception as e:
        df_filtered = pd.DataFrame()
        df_filtered['source'] = np.nan
        return df_filtered
    
def load_els_data_fmb(df_sonic, selected_log_curve):

    try:
        df_well_log = df_sonic.rename(columns={'TVDMSL':'tvd_ss', 'MD':'md'})
        df_well_log = df_well_log[['unique_wellbore_identifier','md', 'tvd_ss', f"{selected_log_curve}"]]
        df_well_log = df_well_log.dropna()        
        df_well_log['interval_velocity_sonic'] = [0.3048/(float(sonic)*0.000001) if sonic != 0 else 0 for sonic in df_well_log[f'{selected_log_curve}']]
        return df_well_log

    except Exception as e:
        df_filtered = pd.DataFrame()
        df_filtered['source'] = np.nan
        return df_filtered 


