import sys
import yaml
import pandas as pd
import psycopg2
import os
import numpy as np
from IPython import embed

class Connection_Database:
  def __init__(self,host,dbname,user,password,sslmode):
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
    self.conn = psycopg2.connect(conn_string)
    self.conn.set_client_encoding('UTF8')
    print("Connection established")

  def connect_database(self,columns, database, uwi):
    self.query = "SELECT {} from {} where unique_wellbore_identifier = '{}'".format(columns, database, uwi)
       
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

def get_connect_database():
  config_file = os.path.join(os.getcwd(),"smda_password","config.yaml")
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  host = config['host']
  dbname = config['dbname']
  user = config['user']
  password = config['password']
  sslmode = config['sslmode']
  return host, dbname, user, password, sslmode

host, dbname, user, password, sslmode = get_connect_database()

def generate_df(host, dbname, user, password, sslmode, columns, database, uwi):
  connect = Connection_Database(host,dbname,user,password,sslmode)
  df = connect.connect_database(columns=columns, database= database, uwi = uwi)
  connect.close_connection()
  return df 

#database_welllog = "smda.smda_staging.els_wellbore_curve_data"
#df_welllog = generate_df(host, dbname, user, password, sslmode, database=database_welllog)
