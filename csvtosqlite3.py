import sqlite3
import pandas as pd

df = pd.read_csv("weather.csv")
df.columns = df.columns.str.strip()

connection = sqlite3.connect("weather.db")  # database name
df.to_sql("weather", connection, if_exists="replace")  # table name

connection.close()
