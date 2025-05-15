import pandas as pd
import numpy as np
import mysql.connector
import re
from mysql.connector import Error

# ------------------ CONFIGURATION ------------------

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Padmavathi@09',
    'database': 'tourism_experience_db'
}

FILES = {
    "continent": r"C:/Users/madhugula padmavathi/Downloads/continent_cleaned.csv",
    "region": r"C:/Users/madhugula padmavathi/Downloads/region_cleaned.csv",
    "country": r"C:/Users/madhugula padmavathi/Downloads/country_cleaned.csv",
    "city": r"C:/Users/madhugula padmavathi/Downloads/city_cleaned.csv",
    "type": r"C:/Users/madhugula padmavathi/Downloads/type_cleaned.csv",
    "visitmode": r"C:/Users/madhugula padmavathi/Downloads/visitmode_cleaned.csv",
    "item": r"C:/Users/madhugula padmavathi/Downloads/item_cleaned.csv",
    "user": r"C:/Users/madhugula padmavathi/Downloads/user_cleaned.csv",
    "transaction": r"C:/Users/madhugula padmavathi/Downloads/transaction_cleaned.csv"
}

# ------------------ HELPER FUNCTIONS ------------------

def create_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"❌ Error connecting to database: {e}")
    return None

def clean_text(val, max_len=255):
    if pd.isna(val):
        return None
    val = str(val).strip()
    val = re.sub(r"[^\w\s.,&()'-]", '', val)  # Allow basic punctuation
    return val[:max_len]

def clean_dataframe(df):
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df.replace({np.nan: None})
    return df

# ------------------ DATA INSERTION ------------------

def insert_data(df, table, cursor):
    for _, row in df.iterrows():
        try:
            if table == "continent":
                cursor.execute("INSERT INTO continent (continentid, continent) VALUES (%s, %s)", 
                    (int(row['continentid']), clean_text(row['continent'], 100)))
            elif table == "region":
                cursor.execute("INSERT INTO region (regionid, region, continentid) VALUES (%s, %s, %s)", 
                    (int(row['regionid']), clean_text(row['region'], 100), int(row['continentid'])))
            elif table == "country":
                cursor.execute("INSERT INTO country (countryid, country, regionid) VALUES (%s, %s, %s)", 
                    (int(row['countryid']), clean_text(row['country'], 100), int(row['regionid'])))
            elif table == "city":
                cursor.execute("INSERT INTO city (cityid, cityname, countryid) VALUES (%s, %s, %s)", 
                    (int(row['cityid']), clean_text(row['cityname'], 100), int(row['countryid'])))
            elif table == "type":
                cursor.execute("INSERT INTO type (attractiontypeid, attractiontype) VALUES (%s, %s)", 
                    (int(row['attractiontypeid']), clean_text(row['attractiontype'], 100)))
            elif table == "visitmode":
                cursor.execute("INSERT INTO visitmode (visitmodeid, visitmode) VALUES (%s, %s)", 
                    (int(row['visitmodeid']), clean_text(row['visitmode'], 100)))
            elif table == "item":
                cursor.execute("""
                    INSERT INTO item (attractionid, attractioncityid, attractiontypeid, attraction, attractionaddress)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    int(row['attractionid']),
                    int(row['attractioncityid']),
                    int(row['attractiontypeid']),
                    clean_text(row['attraction'], 255),
                    clean_text(row['attractionaddress'], 255)
                ))
            elif table == "user":
                cursor.execute("""
                    INSERT INTO user (userid, continentid, regionid, countryid, cityid)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    int(row['userid']),
                    int(row['continentid']),
                    int(row['regionid']),
                    int(row['countryid']),
                    int(row['cityid'])
                ))
            elif table == "transaction":
                cursor.execute("""
                    INSERT INTO transaction (transactionid, userid, visityear, visitmonth, visitmode, attractionid, rating)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    int(row['transactionid']),
                    int(row['userid']),
                    int(row['visityear']),
                    int(row['visitmonth']),
                    clean_text(row['visitmode'], 100),
                    int(row['attractionid']),
                    int(row['rating'])
                ))
        except Exception as e:
            print(f"❌ Error inserting into `{table}`: {e}")

# ------------------ MAIN EXECUTION ------------------

def load_and_insert_files():
    conn = create_connection()
    if not conn:
        print("❌ Database connection failed. Exiting.")
        return

    cursor = conn.cursor()

    for table, path in FILES.items():
        try:
            df = pd.read_csv(path)
            df = clean_dataframe(df)
            insert_data(df, table, cursor)
            conn.commit()
            print(f"✅ Inserted into `{table}` from: {path}")
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed to process `{table}`: {e}")

    cursor.close()
    conn.close()
    print("✅ All data insertion completed.")

# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    load_and_insert_files()
