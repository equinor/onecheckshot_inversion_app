import yaml
import pandas as pd
import psycopg2
import os

class Connection_Database:
  def __init__(self,host,dbname,user,password,sslmode):
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
    self.conn = psycopg2.connect(conn_string)
    self.conn.set_client_encoding('UTF8')
    print("Connection established")

  def connect_database(self,database, columns, uwi):
    well = 'NO 1/9-2'
    self.query = "SELECT {} from {} WHERE unique_wellbore_identifier = '{}'".format(columns, database, uwi)
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

def chksht_welldb_smda(uwi):
  config_file = os.path.join(os.getcwd(),"smda_password","config.yaml")
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  host = config['host']
  dbname = config['dbname']
  user = config['user']
  password = config['password']
  sslmode = config['sslmode']
    
  database_wellbore_checkshot = "smda.smda_workspace.wellbore_checkshot_data"
  connect = Connection_Database(host,dbname,user,password,sslmode)
  #columns_wellbore_checkshot = "id, unique_wellbore_identifier, source_file, tvd_ss, time, time_unit, tvd, tvd_unit, md, md_unit"
  columns_wellbore_checkshot = "*"
  df = connect.connect_database(database=database_wellbore_checkshot,columns=columns_wellbore_checkshot, uwi=uwi)
  connect.close_connection()
  return df

def get_wells():
  config_file = os.path.join(os.getcwd(),"smda_password","config.yaml")
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  host = config['host']
  dbname = config['dbname']
  user = config['user']
  password = config['password']
  sslmode = config['sslmode']
  database_wellbore_checkshot = "smda.smda_workspace.wellbore_checkshot_data"
  connect = Connection_Database(host,dbname,user,password,sslmode)
  list_wells = connect.get_wells(database=database_wellbore_checkshot)
  connect.close_connection()
  return list_wells