import yaml
import pandas as pd
import psycopg2
import os
import numpy as np
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

  def write_dataframe_to_database(self, df, table_name):
      
      self.cur.execute(f"""TRUNCATE {table_name}""")
      lists = list(df.itertuples(index=False, name=None))
      self.cur.executemany(f"""INSERT INTO {table_name} {str(tuple(df.columns)).replace("'","")} VALUES (%s, %s, %s, %s, %s)""", lists)
      self.conn.commit()
      self.cur.close()
      print("Values commited in Database")

  def close_connection(self):
    self.conn.close()

def connect_database():
  config_file = os.path.join(os.getcwd(),"data_management_smda","smda_password","config.yaml")
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
def generate_df(host, dbname, user, password, sslmode):
  database_wellbore_checkshot = "smda.smda_staging.els_wellbore_curve"
  connect = Connection_Database(host,dbname,user,password,sslmode)
  df = connect.connect_database(database= database_wellbore_checkshot)
  connect.close_connection()
  return df 

df = generate_df(host, dbname, user, password, sslmode)
df['data'] = df['data'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
df['data'] = df['data'].apply(lambda x: x.split(','))
df_encoded = pd.get_dummies(df['curve_identifier'])
df = pd.concat([df, df_encoded], axis=1)
df_pivoted = df.pivot(index='unique_wellbore_identifier', columns=['LFP_DT', 'TVDMSL'], values='data')
df_pivoted.columns = df_pivoted.columns.droplevel(0)
df_pivoted = df_pivoted.reset_index().rename_axis(None, axis=1)
df_pivoted = df_pivoted.rename(columns={df_pivoted.columns[1]: "LFP_DT"})
df_pivoted = df_pivoted.rename(columns={df_pivoted.columns[2]: "TVDMSL"})
#
df_pivoted["equal_lenght"] = df_pivoted.LFP_DT.str.len()==df_pivoted.TVDMSL.str.len()
df_pivoted = df_pivoted[df_pivoted["equal_lenght"]]
df_exploded = df_pivoted.explode(['LFP_DT','TVDMSL']).reset_index(drop=True)
df_exploded.replace("null", np.nan, inplace=True)
df_filtered = df_exploded.dropna(subset=['LFP_DT'])
df_filtered = df_filtered.drop('equal_lenght', axis=1)
criteria = df_filtered['unique_wellbore_identifier'].str.startswith('NO 711')
df_filtered = df_filtered[criteria]

from sqlalchemy import create_engine
class Connection_Database_write:
    def __init__(self, host, dbname, user, password, sslmode):
        conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
        
        
        self.engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}/{dbname}')
        self.conn = self.engine.raw_connection()
        self.cur = self.conn.cursor()
        print("Connection established")

    def connect_database(self, database, columns):
        self.query = f"SELECT {columns} from {database}"
        self.df = pd.read_sql_query(self.query, self.conn)
        return self.df

    def write_dataframe_to_database(self, df, table_name):
        
        self.cur.execute(f"""TRUNCATE {table_name}""")
        lists = list(df.itertuples(index=False, name=None))
        chunk_size = 1000
        for i in range(0, len(lists), chunk_size):
          chunk = lists[i:i+chunk_size]
          self.cur.executemany(f"""INSERT INTO {table_name} {str(tuple(df.columns)).replace("'","")} VALUES (%s, %s, %s)""", chunk)
          print(i)
        self.conn.commit()
        print("Values commited in Database")
        
    def close_connection(self):
        self.cur.close()
        self.conn.close()
        print("Connection closed")

db_connection = Connection_Database_write(host,dbname,user,password,sslmode)


# Write the DataFrame to the database
#database_wellbore_checkshot = "smda.smda_staging.els_wellbore_curve_data"
#write = db_connection.write_dataframe_to_database(df=df_filtered, table_name = database_wellbore_checkshot)

# Close the connection
db_connection.close_connection()




#df.loc[df['LFP_DT'], 'LFP_DT'] = df['data']
#df.loc[df['TVDMSL'], 'TVDMSL'] = df['data']



