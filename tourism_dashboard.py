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
    df = df.dropna().copy()
    for col in ['visitmode', 'attraction']:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# ---------- Streamlit UI Setup ----------
st.set_page_config(page_title="Tourism Experience Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Regression", "Classification", "Recommendation", "Insights"])

# Load and preprocess data
df = load_data()
df_clean = preprocess_data(df)

# ---------- HOME PAGE ----------
if page == "Home":
    st.title("🏞️ Tourism Experience Dashboard")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())
    st.subheader("Cleaned Data Sample")
    st.dataframe(df_clean.head())
    st.markdown("This project provides classification, prediction, and recommendation insights into tourist behavior using historical data.")

# ---------- EDA ----------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    with st.expander("Filter Data (optional)"):
        years = sorted(df['visityear'].dropna().unique())
        selected_years = st.multiselect("Visit Year(s)", years, default=years)
        regions = sorted(df['regionid'].dropna().unique())
        selected_regions = st.multiselect("Region ID(s)", regions, default=regions)
        cities = sorted(df['cityid'].dropna().unique())
        selected_cities = st.multiselect("City ID(s)", cities, default=cities)
        attraction_types = sorted(df['attractiontypeid'].dropna().unique())
        selected_attraction_types = st.multiselect("Attraction Type ID(s)", attraction_types, default=attraction_types)

    filtered_df = df[
        (df['visityear'].isin(selected_years)) &
        (df['regionid'].isin(selected_regions)) &
        (df['cityid'].isin(selected_cities)) &
        (df['attractiontypeid'].isin(selected_attraction_types))
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", filtered_df.shape[0])
        col2.metric("Unique Users", filtered_df['userid'].nunique())
        col3.metric("Unique Attractions", filtered_df['attraction'].nunique())
        col4.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")

        st.subheader("Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['rating'], bins=5, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Visit Mode Distribution")
        fig, ax = plt.subplots()
        filtered_df['visitmode'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        st.pyplot(fig)

        st.subheader("Top Attractions")
        top_attractions = filtered_df['attraction'].value_counts().head(10)
        st.bar_chart(top_attractions)

        st.subheader("Visits by Region")
        st.bar_chart(filtered_df['regionid'].value_counts().head(10))

        st.subheader("City-wise Visit Count")
        st.bar_chart(filtered_df['cityid'].value_counts().head(10))

        st.subheader("Yearly Visit Trends")
        yearly = filtered_df.groupby('visityear').size()
        st.line_chart(yearly)

        st.subheader("Monthly Visit Trends")
        monthly = filtered_df.groupby('visitmonth').size().sort_index()
        st.line_chart(monthly)

        st.subheader("Feature Correlation Heatmap")
        corr_df = preprocess_data(filtered_df)
        fig, ax = plt.subplots()
        sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


# ---------- REGRESSION ----------

elif page == "Regression":
    st.title("📈 Predict Ratings with Regression")
    features_reg = st.multiselect("Select features:", df_clean.columns.drop(['transactionid', 'rating']), default=['visityear', 'visitmonth', 'attractiontypeid'])

    if st.button("Train Regression Model"):
        X = df_clean[features_reg]
        y = df_clean['rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("Model trained successfully.")
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
        st.write("R²:", r2_score(y_test, y_pred))


elif page == "Classification":
    st.title("📌 Classify High/Low Ratings")
    threshold = st.slider("Rating Threshold (>= High):", 1, 5, 3)
    df_clean['rating_class'] = (df_clean['rating'] >= threshold).astype(int)
    features_cls = st.multiselect("Select features:", df_clean.columns.drop(['transactionid', 'rating', 'rating_class']), default=['visitmonth', 'attractiontypeid'])

    if st.button("Train Classification Model"):
        X = df_clean[features_cls]
        y = df_clean['rating_class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("Classification model trained.")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

# ---------- RECOMMENDATION ----------
elif page == "Recommendation":
    st.title("🎯 Attraction Recommendations")

    st.subheader("🔹 Based on Region and Attraction Type")
    region_id = st.selectbox("Select Region", sorted(df['regionid'].unique()))
    type_id = st.selectbox("Select Attraction Type", sorted(df['attractiontypeid'].unique()))

    recs = df[(df['regionid'] == region_id) & (df['attractiontypeid'] == type_id)]
    top = recs['attraction'].value_counts().head(5).reset_index()
    top.columns = ['Attraction', 'Count']
    st.dataframe(top)

    st.subheader("🔹 Similar Attractions (KNN)")
    selected_attraction = st.selectbox("Choose an Attraction", df['attraction'].unique())
    pivot = df.pivot_table(index='attraction', columns='userid', values='rating').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(pivot)
    dist, idx = model_knn.kneighbors(pivot.loc[selected_attraction].values.reshape(1, -1), n_neighbors=6)
    st.write("Similar Attractions:")
    for i in range(1, len(idx[0])):
        st.write(f"{i}. {pivot.index[idx[0][i]]}")

    st.subheader("🔮 Future Rating Prediction Based on User Inputs")

    with st.form("predict_rating_form"):
        st.markdown("Enter values to predict rating for a future visit:")
        future_visityear = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2025)
        future_visitmonth = st.selectbox("Visit Month", list(range(1, 13)))
        future_visitmode = st.selectbox("Visit Mode", df['visitmode'].unique())
        future_attraction = st.selectbox("Attraction", df['attraction'].unique())
        future_typeid = st.selectbox("Attraction Type ID", sorted(df['attractiontypeid'].unique()))
        submitted = st.form_submit_button("Predict Future Rating")

        if submitted:
            try:
                visitmode_encoded = LabelEncoder().fit(df['visitmode']).transform([future_visitmode])[0]
                attraction_encoded = LabelEncoder().fit(df['attraction']).transform([future_attraction])[0]
                input_df = pd.DataFrame([{
                    'visityear': future_visityear,
                    'visitmonth': future_visitmonth,
                    'visitmode': visitmode_encoded,
                    'attraction': attraction_encoded,
                    'attractiontypeid': future_typeid
                }])
                X = df_clean[['visityear', 'visitmonth', 'visitmode', 'attraction', 'attractiontypeid']]
                y = df_clean['rating']
                future_model = RandomForestRegressor()
                future_model.fit(X, y)
                prediction = future_model.predict(input_df)[0]
                st.success(f"🎯 Predicted Rating: {prediction:.2f}")
            except ValueError:
                st.error("❌ Error: Please select values that were seen in the original dataset.")


# ---------- INSIGHTS PAGE ----------
elif page == "Insights":
    st.title("📌 Insights and Recommendations")

    st.subheader("❌ Common Patterns in Low Ratings")
    low_rating_df = df[df['rating'] <= 2]

    col1, col2 = st.columns(2)

    with col1:
        st.write("Low Ratings by Visit Month")
        st.bar_chart(low_rating_df['visitmonth'].value_counts())

    with col2:
        st.write("Low Ratings by Attraction Type")
        st.bar_chart(low_rating_df['attractiontypeid'].value_counts())

    st.write("Low Ratings by Visit Mode")
    st.bar_chart(low_rating_df['visitmode'].value_counts())

    st.write("Top 5 Attractions with Most Low Ratings")
    low_attractions = low_rating_df['attraction'].value_counts().head(5)
    st.dataframe(low_attractions)

    st.info("""
    🔍 **Suggestions to Improve Ratings:**
    - Focus on improving visitor experience during summer months (May–June)
    - Enhance interactivity and engagement in attraction type ID 3 and 5
    - Offer group packages or guided options to solo travelers
    - Investigate and address common complaints in Region ID 2 and 4
    """)
