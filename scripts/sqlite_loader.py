import sqlite3
import pandas as pd
import os

def initialize_connection(db_path="data/kpop_wins.db"):
    """
    Connect to SQLite database and return connection and cursor.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def load_csv_to_sqlite(csv_path, table_name, conn, if_exists='replace'):
    """
    Load a single CSV file into a SQLite table.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    return df

def combine_all_shows(conn, output_table="all_awards"):
    """
    Combine all individual show tables into one table using UNION ALL.
    Assumes all individual tables have the same structure.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cursor.fetchall() if row[0] != output_table]

    if not table_names:
        raise ValueError("No tables found to combine.")

    def quote_identifier(s):
        return '"' + s.replace('"', '""') + '"'

    union_parts = []
    for table in table_names:
        quoted_table = quote_identifier(table)
        union_parts.append(
            f"SELECT *, '{table}' AS source_table FROM {quoted_table}"
        )

    union_query = " UNION ALL ".join(union_parts)

    # Execute as a single clean query
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(output_table)}")
    cursor.execute(f"CREATE TABLE {quote_identifier(output_table)} AS {union_query}")
    conn.commit()