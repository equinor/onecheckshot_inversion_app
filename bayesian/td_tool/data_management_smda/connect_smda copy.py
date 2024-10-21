import yaml
import pandas as pd
import psycopg2
import os
import numpy as np
from IPython import embed
from pandasgui import show
class Connection_Database:
  def __init__(self,host,dbname,user,password,sslmode):
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
    self.conn = psycopg2.connect(conn_string)
    self.conn.set_client_encoding('UTF8')
    print("Connection established")

  def connect_database(self,database):
    self.query = "SELECT * from {}".format(database)
    self.df = pd.read_sql_query(self.query, self.conn)
    return self.df

  def get_wells(self,database):
    self.query = "SELECT DISTINCT unique_wellbore_identifier FROM {}".format(database)
    self.df = pd.read_sql_query(self.query, self.conn)
    cursor = self.conn.cursor()
    cursor.execute(self.query)
    list_wells = [row[0] for row in cursor.fetchall()]
    return list_wells

  def close_connection(self):
    self.conn.close()

def connect_database():
  config_file = os.path.join(os.getcwd(),"bayesian","td_tool","data_management_smda","smda_password","config.yaml")
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  host = config['host']
  dbname = config['dbname']
  user = config['user']
  password = config['password']
  sslmode = config['sslmode']
  return host, dbname, user, password, sslmode

host, dbname, user, password, sslmode = connect_database()

from IPython import embed
def generate_df(host, dbname, user, password, sslmode, database):
  connect = Connection_Database(host,dbname,user,password,sslmode)
  df = connect.connect_database(database= database)
  connect.close_connection()
  return df 

#database_welllog = "smda.smda_staging.els_wellbore_curve_data"
#df_welllog = generate_df(host, dbname, user, password, sslmode, database=database_welllog)

database_checkshot = "smda.smda_workspace.wellbore_checkshot_data_qc"
df_checkshot = generate_df(host, dbname, user, password, sslmode, database=database_checkshot)

"""
def check_velocity_qc(velocity):
    if pd.isnull(velocity):
        return "not checked"
    elif 1400 <= velocity <= 6500:
        return "passed"
    else:
        return "not passed"
"""
#df_checkshot['average_velocity_qc'] = df_checkshot['average_velocity'].apply(check_velocity_qc)
#

#df_checkshot = df_checkshot[df_checkshot["sample_number_qc"]!='not passed']
df_checkshot = df_checkshot.fillna('NA')
df_checkshot[["tvd_ss", "time", "interval_velocity", "average_velocity", "uniqueness", "depth_reference_elevation", "mae_soniclog_checkshot", "calc_tvd_ss", "calc_tvd"]] = df_checkshot[["tvd_ss", "time", "interval_velocity", "average_velocity", "uniqueness", "depth_reference_elevation", "mae_soniclog_checkshot", "calc_tvd_ss", "calc_tvd"]].apply(pd.to_numeric, errors='coerce')
#df_checkshot = df_checkshot[df_checkshot["average_velocity_qc"]!='not passed']
df_checkshot["sub_tvdss"] = df_checkshot["tvd_ss"] - df_checkshot["calc_tvd_ss"]
#df_checkshot = df_checkshot[df_checkshot["trajectory_checked"]!='not passed']
#df_checkshot = df_checkshot[df_checkshot["comparison_sonic_log_qc"]!='not passed']
df_checkshot = df_checkshot[df_checkshot["sub_tvdss"]<500]

#get percentage of groups
#df_checkshot.groupby('trajectory_checked')['unique_wellbore_identifier'].nunique()/df_checkshot.groupby('trajectory_checked')['unique_wellbore_identifier'].nunique().sum() 
df_checkshot["tvd_ss"] = df_checkshot["tvd_ss"]*(-1)

not_passed_source_files = df_checkshot[df_checkshot['trajectory_checked'] == 'not passed']['source_file'].unique()
df_checkshot.loc[df_checkshot['source_file'].isin(not_passed_source_files), 'trajectory_checked'] = 'not passed'
#df_checkshot['uniqueness_string'] = df_checkshot['uniqueness'].apply(lambda x: 'higher than 0.5' if x > 0.5 else 'lower than 0.5')
embed()
# Write the DataFrame to the database
#database_wellbore_checkshot = "smda.smda_staging.els_wellbore_curve_data"
#write = db_connection.write_dataframe_to_database(df=df_filtered, table_name = database_wellbore_checkshot)

# Close the connection




#df.loc[df['LFP_DT'], 'LFP_DT'] = df['data']
#df.loc[df['TVDMSL'], 'TVDMSL'] = df['data']



