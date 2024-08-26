import pickle
import os
from IPython import embed
import pandas as pd
import psycopg2
import pandas as pd

root_folder = 'demo_example/'
pkl_output_folder = root_folder + 'output/bayes_csc/pkl/'
files_in_folder = os.listdir(pkl_output_folder)

files = list()
for output_pkl_file in files_in_folder:
    with open(pkl_output_folder + output_pkl_file, 'rb') as file:
        data = pickle.load(file)
        df_well = data['df_well']
        df_well['well'] = data['well_name']
        df_well['water_depth'] = data['water_depth']
        files.append(df_well)

output_bayesian = pd.concat(files, ignore_index=True)



class Connection_Database:
    def __init__(self, host, dbname, user, password, sslmode):
        conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

        self.conn = psycopg2.connect(conn_string)

        self.conn.set_client_encoding('UTF8')
        print("Connection established")

    def connect_database(self, database, columns):
        self.query = "SELECT {} from {}".format(columns, database)
        self.df = pd.read_sql_query(self.query, self.conn)
        return self.df

    def write_dataframe_to_database(self, df, table_name):
        cur = self.conn.cursor()
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS smda_workspace.{table_name} (
            md double precision,
            tvd double precision,
            tvd_ss double precision,
            time double precision not null
            )"""
        cur.execute(create_table_sql)
 
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        self.conn.commit()
        
        print("Data written to database")

    def close_connection(self):
        self.conn.close()

# Create an instance of the class

host = "s173-smda-psql-dev.postgres.database.azure.com"
dbname = "smda"
user = "db_admin"
password = "dw5xllbpc3p6v6yd02u6e1eqmzqstnsan"
sslmode = "require"
db_connection = Connection_Database(host,dbname,user,password,sslmode)


# Create a DataFrame
data = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}
df = pd.DataFrame(data)

# Write the DataFrame to the database
database_wellbore_checkshot = "table"

write = db_connection.write_dataframe_to_database(df=df, table_name = database_wellbore_checkshot)



# Close the connection
db_connection.close_connection()

"""

columns_wellbore_checkshot = "id, unique_wellbore_identifier, tvd_ss, time, time_unit, tvd, tvd_unit, md, md_unit"
df = connect.connect_database(database=database_wellbore_checkshot,columns=columns_wellbore_checkshot)
database_smda = "smda.smda_master.v_wellbore_time_depth_data"
columns_smda = "*"
df_smda = connect.connect_database(database=database_smda,columns=columns_smda)
connect.close_connection()

df = convert_and_clean_units(df)
"""