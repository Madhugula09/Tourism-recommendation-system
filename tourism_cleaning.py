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
    
   
