import pandas as pd
import mysql.connector
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set file path
file_path = "C:/Users/madhugula padmavathi/Downloads/combined_tourism_data.csv"

# Load CSV
df = pd.read_csv(file_path)


# all columns
df = df[["UserId", "ContinentId", "RegionId", "CountryId", "CityId", "Continent", "Region", "Country", "AttractionId", "CityName", "AttractionCityId", "AttractionTypeId", "Attraction", "AttractionAddress", "AttractionType", "VisitModeId", "VisitMode", "TransactionId", "VisitYear", "VisitMonth", "Rating"]]


# 🔹 Step 1: Connect to MySQL and Load Data
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="tourism_experience_db"
    )
    
    tables = ["Transaction", "User", "City", "Attraction", "VisitMode", "Country", "Continent", "AttractionType"]
    dfs = {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in tables}
    
    conn.close()
    return dfs

# 🔹 Step 2: Encode Categorical Variables
def encode_categorical(dataframes):
    label_encoders = {}
    categorical_columns = ['VisitMode', 'Continent', 'Country', 'AttractionTypeId']
    
    for col in categorical_columns:
        for table, df in dataframes.items():
            if col in df.columns:
                le = LabelEncoder()
                dataframes[table][col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
                label_encoders[col] = le
    
    return dataframes, label_encoders

# 🔹 Step 3: Aggregate User-Level Features
def aggregate_user_features(df_transaction):
    user_features = df_transaction.groupby(["UserId", "VisitMode"])["Rating"].mean().reset_index()
    user_features.rename(columns={"Rating": "AvgRatingPerVisitMode"}, inplace=True)
    return user_features

# 🔹 Step 4: Merge Relevant Tables
def merge_data(dataframes, user_features):
    merged_df = dataframes["Transaction"].merge(dataframes["User"], on="UserId", how="left")
    merged_df = merged_df.merge(dataframes["City"], on="CityId", how="left")
    merged_df = merged_df.merge(dataframes["Attraction"], on="AttractionId", how="left")
    merged_df = merged_df.merge(user_features, on=["UserId", "VisitMode"], how="left")

    # Drop unnecessary columns
    merged_df.drop(columns=["UserId", "TransactionId"], inplace=True, errors="ignore")
    
    return merged_df

# 🔹 Step 5: Normalize Numerical Features
def normalize_features(df):
    scaler = StandardScaler()
    numeric_columns = ["Rating", "AvgRatingPerVisitMode"]

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# 🔹 Step 6: Save Cleaned Data Back to MySQL
def save_to_mysql(df):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="tourism_experience_db"
    )
    
    cursor = conn.cursor()
    
    # Create a new table for cleaned data
    cursor.execute("DROP TABLE IF EXISTS Tourism_Data")
    cursor.execute("""
        CREATE TABLE Tourism_Data (
            VisitYear INT,
            VisitMonth INT,
            VisitMode INT,
            AttractionId INT,
            Rating FLOAT,
            AvgRatingPerVisitMode FLOAT
        )
    """)
    
    # Insert cleaned data
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO Cleaned_Transaction (VisitYear, VisitMonth, VisitMode, AttractionId, Rating, AvgRatingPerVisitMode)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, (row['VisitYear'], row['VisitMonth'], row['VisitMode'], row['AttractionId'], row['Rating'], row['AvgRatingPerVisitMode']))

    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Cleaned data saved to MySQL successfully!")

# 🔹 Main Execution
if __name__ == "__main__":
    print("📥 Loading data from MySQL...")
    dataframes = load_data()
    
    print("🔄 Encoding categorical variables...")
    dataframes, encoders = encode_categorical(dataframes)
    
    print("📊 Aggregating user features...")
    user_features = aggregate_user_features(dataframes["Transaction"])
    
    print("🔗 Merging data...")
    cleaned_df = merge_data(dataframes, user_features)
    
    print("📏 Normalizing features...")
    cleaned_df = normalize_features(cleaned_df)
    
    print("💾 Saving cleaned data to MySQL...")
    save_to_mysql(cleaned_df)

    print("🎉 Data Preprocessing Completed Successfully!")