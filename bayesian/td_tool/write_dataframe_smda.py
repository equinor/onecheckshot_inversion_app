import pickle
import os
from IPython import embed
import pandas as pd
import psycopg2
import pandas as pd
import sys
import yaml
import numpy as np
import json
from sqlalchemy import create_engine
import mysql.connector
sys.path.append(os.getcwd())
config_file = os.path.join(os.getcwd(),"smda_password","config.yaml")

checkshot_qc_filepath = os.path.join(os.getcwd(),"data","df_checkshot_qc.csv")
checkshot_qc = pd.read_csv(checkshot_qc_filepath)
checkshot_qc = checkshot_qc[["tvd_ss","twt picked", "md", "uwi", "source_tag"]]
checkshot_qc.rename(columns={'twt picked': 'twt'}, inplace=True)

def get_wells():
  with open(config_file, "r") as file:
    config = yaml.safe_load(file)
  return config

class Connection_Database:
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
        self.cur.executemany(f"""INSERT INTO {table_name} {str(tuple(checkshot_qc.columns)).replace("'","")} VALUES (%s, %s, %s, %s, %s)""", lists)
        self.conn.commit() 
        print("Values commited in Database")
        
    def close_connection(self):
        self.cur.close()
        self.conn.close()
        print("Connection closed")

# Define parameters for database connection
config = get_wells()
host = config['host']
dbname = config['dbname']
user = config['user']
password = config['password']
sslmode = config['sslmode']

#Connect Database
db_connection = Connection_Database(host,dbname,user,password,sslmode)


# Write the DataFrame to the database
database_wellbore_checkshot = "smda.smda_workspace.table"
write = db_connection.write_dataframe_to_database(df=checkshot_qc, table_name = database_wellbore_checkshot)

# Close the connection
db_connection.close_connection()