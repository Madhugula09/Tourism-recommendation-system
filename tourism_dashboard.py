import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors

# ---------- Database Connection ----------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="tourism_experience_db"
    )

@st.cache_data
def load_data():
    conn = get_connection()
    query = """
    SELECT 
        t.transactionid,
        t.userid,
        t.visityear,
        t.visitmonth,
        t.visitmode,
        t.rating,
        u.continentid, u.regionid, u.countryid, u.cityid,
        i.attractiontypeid,
        i.attraction
    FROM transaction t
    JOIN user u ON t.userid = u.userid
    JOIN item i ON t.attractionid = i.attractionid
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------- Data Preprocessing ----------
def preprocess_data(df):
    df = df.dropna()
    for col in ['visitmode', 'attraction']:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# ---------- Streamlit UI ----------
st.title("🏞️ Tourism Experience Dashboard")

df = load_data()
st.subheader("Raw Data")
st.dataframe(df.head())

# Cleaned Data
df_clean = preprocess_data(df)
st.subheader("Cleaned Data")
st.dataframe(df_clean.head())

# ---------- EDA ----------
st.header("📊 Exploratory Data Analysis")

# Visit Mode Distribution
st.subheader("Visit Mode Distribution")
st.bar_chart(df['visitmode'].value_counts())

# Region-wise Popular Attractions
st.subheader("Top Attractions by Region")
top_attractions = df.groupby(['regionid', 'attraction']).size().reset_index(name='count')
top_5 = top_attractions.sort_values(['regionid', 'count'], ascending=[True, False]).groupby('regionid').head(5)
st.dataframe(top_5)

# City-wise Visit Mode
st.subheader("Visit Mode Trends by City")
visit_mode_by_city = df.groupby(['cityid', 'visitmode']).size().unstack().fillna(0)
st.bar_chart(visit_mode_by_city)

# Correlation
st.subheader("Feature Correlation Heatmap")
corr = df_clean.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------- Exploratory Data Analysis ----------
st.header("📊 Exploratory Data Analysis")

# Overview metrics
st.subheader("🔢 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", df.shape[0])
col2.metric("Unique Users", df['transactionid'].nunique())
col3.metric("Unique Attractions", df['attraction'].nunique())
col4.metric("Average Rating", f"{df['rating'].mean():.2f}")

# Rating Distribution
st.subheader("⭐ Rating Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['rating'], bins=5, kde=True, ax=ax1)
st.pyplot(fig1)

# Visit Mode Analysis
st.subheader("🚶‍♂️ Visit Mode Distribution")
fig2, ax2 = plt.subplots()
df['visitmode'].value_counts().plot(kind='bar', color='skyblue', ax=ax2)
ax2.set_xlabel("Visit Mode")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Attraction Popularity
st.subheader("🏖️ Most Visited Attractions")
top_attractions = df['attraction'].value_counts().head(10)
st.bar_chart(top_attractions)

# Region-wise Visits
st.subheader("🗺️ Visits by Region")
region_counts = df['regionid'].value_counts().head(10)
fig3, ax3 = plt.subplots()
region_counts.plot(kind='bar', color='green', ax=ax3)
ax3.set_xlabel("Region ID")
ax3.set_ylabel("Number of Visits")
st.pyplot(fig3)

# City-wise Analysis
st.subheader("🏙️ City-wise Visit Count")
city_counts = df['cityid'].value_counts().head(10)
fig4, ax4 = plt.subplots()
city_counts.plot(kind='bar', color='orange', ax=ax4)
ax4.set_xlabel("City ID")
ax4.set_ylabel("Number of Visits")
st.pyplot(fig4)

# Yearly Trends
st.subheader("📆 Yearly Visit Trends")
yearly = df.groupby('visityear').size()
st.line_chart(yearly)

# Monthly Trends
st.subheader("🗓️ Monthly Visit Trends")
monthly = df.groupby('visitmonth').size().sort_index()
st.line_chart(monthly)

# ---------- Regression Model ----------
st.header("📈 Regression: Predict Rating")
features_reg = st.multiselect("Select features for regression:", df_clean.columns.drop(['transactionid', 'rating']), default=['visityear', 'visitmonth', 'attractiontypeid'])

if st.button("Train Regression Model"):
    X = df_clean[features_reg]
    y = df_clean['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

# ---------- Classification Model ----------
st.header("📌 Classification: High/Low Rating")
thresh = st.slider("Rating Threshold (>= High):", 1, 5, 3)
df_clean['rating_class'] = (df_clean['rating'] >= thresh).astype(int)
features_cls = st.multiselect("Select features for classification:", df_clean.columns.drop(['transactionid', 'rating', 'rating_class']), default=['visitmonth', 'attractiontypeid'])

if st.button("Train Classification Model"):
    X = df_clean[features_cls]
    y = df_clean['rating_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

# ---------- Visit Mode Prediction ----------
st.header("🤖 Predict Visit Mode")
features_vm = ['visityear', 'visitmonth', 'continentid', 'regionid', 'countryid', 'cityid', 'attractiontypeid']
X = df_clean[features_vm]
y = df['visitmode']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.subheader("Visit Mode Prediction Results")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Predict for new user input
st.subheader("🔍 Predict Visit Mode from User Input")
user_input = {}
for col in features_vm:
    user_input[col] = st.number_input(f"{col}", min_value=0, value=int(df[col].median()))
input_df = pd.DataFrame([user_input])
predicted_mode = le.inverse_transform(clf.predict(input_df))[0]
st.success(f"Predicted Visit Mode: {predicted_mode}")

# ---------- Personalized Recommendations ----------
st.header("🎯 Personalized Attraction Recommendations")
user_region = st.selectbox("Select Region ID", df['regionid'].unique())
user_attraction_type = st.selectbox("Select Attraction Type ID", df['attractiontypeid'].unique())
recommended = df[
    (df['regionid'] == user_region) &
    (df['attractiontypeid'] == user_attraction_type)
].groupby('attraction').size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
st.subheader("Top Recommended Attractions")
st.dataframe(recommended)

# ---------- Similar Attractions ----------
st.header("📌 Similar Attractions using KNN")
attraction_input = st.selectbox("Select attraction to find similar ones:", df['attraction'].unique())
if st.button("Find Similar Attractions"):
    pivot = df.pivot_table(index='attraction', columns='userid', values='rating').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(pivot)
    dist, idx = model_knn.kneighbors(pivot.loc[attraction_input].values.reshape(1, -1), n_neighbors=6)
    st.subheader("Similar Attractions:")
    for i in range(1, len(idx[0])):
        st.write(f"{i}. {pivot.index[idx[0][i]]}")


# ---------- Future Rating Prediction ----------
st.header("🔮 Future Rating Prediction Based on User Inputs")

# Input fields for user to predict future rating
with st.form("predict_rating_form"):
    st.subheader("Enter details to predict rating:")
    visityear = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2024)
    visitmonth = st.selectbox("Visit Month", list(range(1, 13)))
    visitmode = st.selectbox("Visit Mode", df['visitmode'].unique())
    attraction = st.selectbox("Attraction", df['attraction'].unique())
    attractiontypeid = st.selectbox("Attraction Type ID", sorted(df['attractiontypeid'].unique()))

    submitted = st.form_submit_button("Predict Rating")

    if submitted:
        # Encode categorical features
        visitmode_encoded = LabelEncoder().fit(df['visitmode']).transform([visitmode])[0]
        attraction_encoded = LabelEncoder().fit(df['attraction']).transform([attraction])[0]

        # Prepare input for prediction
        input_data = pd.DataFrame([{
            'visityear': visityear,
            'visitmonth': visitmonth,
            'visitmode': visitmode_encoded,
            'attraction': attraction_encoded,
            'attractiontypeid': attractiontypeid
        }])

        # Train model on entire cleaned dataset
        X = df_clean[['visityear', 'visitmonth', 'visitmode', 'attraction', 'attractiontypeid']]
        y = df_clean['rating']
        model = RandomForestRegressor()
        model.fit(X, y)

        predicted_rating = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Rating: {predicted_rating:.2f}")
