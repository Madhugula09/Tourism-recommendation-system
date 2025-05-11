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
        password="12345",
        database="project4"
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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Models"])

if page == "Home":
    st.header("📌 Project Overview")
    st.write("This app provides insights into tourism data, predicts visit mode, and recommends attractions.")
    
    st.subheader("📊 Available Data Tables")
    tables = ["continent", "region", "country", "city", "user", "attraction", "transaction"]
    for table in tables:
        st.write(f"**{table.capitalize()} Table Sample**")
        data = load_data(table).head()
        st.dataframe(data, height=200, use_container_width=True)

elif page == "EDA":
    st.header("🔍 Exploratory Data Analysis")
    transaction_data = load_data("transaction")
    user_data = load_data("user")
    attraction_data = load_data("attraction")
    
    st.subheader("1. User Distribution by Continent")
    plot_distribution(user_data, "ContinentId", "User Distribution Across Continents")
    
    st.subheader("2. Popular Attraction Types")
    plot_distribution(attraction_data, "AttractionTypeId", "Most Popular Attraction Types")
    
    st.subheader("3. Visit Mode Analysis")
    plot_distribution(transaction_data, "VisitMode", "Visit Mode Preferences")

elif page == "Models":
    st.header("🤖 Model Overview")
    st.write("This section showcases model predictions for Visit Mode and Attractions Recommendations.")
    
    # Load data
    transaction_data = load_data("transaction")
    user_data = load_data("user")
    attraction_data = load_data("attraction")
    
    # Prepare Data for Classification (Visit Mode Prediction)
    st.subheader("🎯 Visit Mode Prediction")
    features = ["UserId", "VisitYear", "VisitMonth", "AttractionId", "Rating"]
    X = transaction_data[features]
    y = transaction_data["VisitMode"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    
    # Prepare Data for Regression (Rating Prediction)
    st.subheader("📈 Rating Prediction")
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_reg_pred = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_reg_pred)
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    
    # Simple Recommendation (Top Rated Attractions)
    st.subheader("🏆 Recommended Attractions")
    top_attractions = attraction_data.groupby("AttractionId")["Attraction"].count().nlargest(5)
    st.dataframe(top_attractions, height=200, use_container_width=True)
