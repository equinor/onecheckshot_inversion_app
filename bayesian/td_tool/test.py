import os
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.helpers import dataframe_from_result_table
import time
import numpy as np
#import matplotlib.pyplot as plt
# import datashader as ds, colorcet

os.environ['REQUESTS_CA_BUNDLE'] = 'ca-bundle.crt'
AAD_TENANT_ID = "3aa4a235-b6e2-48d5-9195-7fcf05b459b0"
KUSTO_CLUSTER = "https://dec-sdf-dh-prod-ne.northeurope.kusto.windows.net"
KUSTO_DATABASE = "ELS"
KCSB = KustoConnectionStringBuilder.with_interactive_login(KUSTO_CLUSTER)
KCSB.authority_id = AAD_TENANT_ID
KUSTO_CLIENT = KustoClient(KCSB)

KUSTO_QUERY = """
els_logdata_lfp
| distinct unique_wellbore_identifier
"""
RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)

# Process the query response
df = dataframe_from_result_table(RESPONSE.primary_results[0])
print(df)
