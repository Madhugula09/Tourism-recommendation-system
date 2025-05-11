import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Database connection
def load_data(table_name):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="tourism_experience_db"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()
    conn.close()
    return pd.DataFrame(data)

def plot_distribution(data, column, title):
    plt.figure(figsize=(8, 5))
    data[column].value_counts().plot(kind='bar')
    plt.title(title)
    st.pyplot(plt)

# Streamlit App
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("🏝️ Tourism Experience Analytics")

# Sidebar Navigation


