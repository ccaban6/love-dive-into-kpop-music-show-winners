import sqlite3
import pandas as pd
import re

RAW_DB = "../data/sql/raw_ingest.db"
CLEAN_DB = "../data/sql/clean.db"

def extract_show_name(table_name: str):
    # Extract the show name from a table name like '2024_show_music_core'
    parts = table_name.split('_', 1)
    return parts[1] if len(parts) == 2 else table_name

def build_clean_database():
    # Connect to raw and clean DBs
    raw_conn = sqlite3.connect(RAW_DB)
    clean_conn = sqlite3.connect(CLEAN_DB)

    # Fetch all table names in raw DB
    cursor = raw_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [row[0] for row in cursor.fetchall()]

    # Dictionary to group tables by show name
    show_tables = {}
    for table in all_tables:
        match = re.match(r'\d{4}_(.+)', table)
        if match:
            show = match.group(1)
            show_tables.setdefault(show, []).append(table)

    # Combine tables by show name
    all_dataframes = []
    for show, tables in show_tables.items():
        show_dfs = []
        for table in tables:
            df = pd.read_sql_query(f"SELECT show, date, placement, artist, song, total FROM '{table}'", raw_conn)
            df['source_table'] = table
            # df['show_name'] = show
            show_dfs.append(df)
    
        # Align columns before concat
        combined_df = pd.concat(show_dfs, ignore_index=True)
        all_dataframes.append(combined_df)

    # Final clean dataset
    clean_df = pd.concat(all_dataframes, ignore_index=True)
    clean_df = preprocess_database(clean_df) 
    clean_df = apply_manual_updates_by_position(clean_df, "../data/manual_changes/manual_awards.xlsx")
    clean_df.to_sql("all_awards", clean_conn, if_exists="replace", index=False)

    metadata_df = add_artist_metadata('../data/manual_changes/kpop_metadata.xlsx')
    metadata_df.to_sql("artist_metadata", clean_conn, if_exists='replace', index=False)

    raw_conn.close()
    clean_conn.close()
    print("Clean database created at:", CLEAN_DB)

def add_artist_metadata(sheet_path: str):
    df = pd.read_excel(sheet_path, sheet_name=None)
    metadata_df = df['artist_metadata_minimal']
    return metadata_df

def apply_manual_updates_by_position(db_df, sheet_path: str):
    df = pd.read_excel(sheet_path, sheet_name=None)
    df_old = df['old_rows']
    df_new = df['updated_rows']
    # Convert date columns to consistent format
    df_old['date'] = pd.to_datetime(df_old['date'])
    df_new['date'] = pd.to_datetime(df_new['date'])

    # Iterate through each old row and apply the corresponding update
    for i in range(len(df_old)):
        old_row = df_old.iloc[i]
        new_row = df_new.iloc[i]

        # Create boolean mask to find the matching row (all columns must match)
        match = (db_df == old_row).all(axis=1)

        # If a match is found, replace that row
        if match.any():
            match_ind = match[match == True].index[0]
            db_df.iloc[match_ind, :] = new_row
        else:
            # If no exact match, append as new (fallback)
            db_df = pd.concat([db_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Could not find exact match. Appended the {i}th row of the updated rows")

    # Concatenate the new (missing) rows
    db_df = pd.concat([db_df, df['new_rows']], ignore_index=True)
    # Remove duplicates after update
    db_df.drop_duplicates(inplace=True)

    print("Manual updates applied based on row order.")
    return db_df

def preprocess_database(df):
    # Fill missing or 'NA' song titles
    df['song'] = df['song'].fillna('NA')
    df.loc[df['song'].str.strip().str.upper() == 'NA', 'song'] = 'NA'

    # Replace fancy apostrophes ’ with straight apostrophes '
    df['song'] = df['song'].str.replace("’", "'", regex=False)
    df['artist'] = df['artist'].str.replace("’", "'", regex=False)

    # Strip whitespace and normalize casing for Artist and Song
    df['artist'] = df['artist'].str.strip()
    df['song'] = df['song'].str.strip()

    # Remove periods from Placement (e.g., "1." -> "1")
    df['placement'] = df['placement'].astype(str).str.replace('.0', '', regex=False)
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

    # Remove commas from Total (e.g., "1,234" -> "1234") and convert to numeric
    df['total'] = df['total'].astype(str).str.replace(',', '', regex=False)
    df['total'] = pd.to_numeric(df['total'], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

if __name__ == "__main__":
    build_clean_database()
