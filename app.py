import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown("---")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📂 Dataset Preview",
        "📊 Dataset Information",
        "📈 EDA",
        "🤖 Model Training",
        "🎯 Customer Prediction",
        "ℹ️ About"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.header("Customer Segmentation Project")

    st.write("""
This project groups customers into different clusters using the
K-Means Clustering algorithm.

### Features
- Dataset Preview
- Dataset Information
- Exploratory Data Analysis (EDA)
- Elbow Method
- K-Means Model
- Customer Cluster Prediction
""")

    st.success("Select any option from the sidebar.")

# -----------------------------
# DATASET PREVIEW
# -----------------------------
elif page == "📂 Dataset Preview":

    st.header("Dataset Preview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Last 5 Rows")
    st.dataframe(df.tail())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

# -----------------------------
# DATASET INFORMATION
# -----------------------------
elif page == "📊 Dataset Information":

    st.header("Dataset Information")

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Duplicate Values")
    st.write(df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
# -----------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------
elif page == "📈 EDA":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Age"], bins=10)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Annual Income Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Annual Income (k$)"], bins=10)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Spending Score Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Spending Score (1-100)"], bins=10)
    ax.set_xlabel("Spending Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Age vs Spending Score")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(
        df["Age"],
        df["Spending Score (1-100)"]
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)

    st.subheader("Annual Income vs Spending Score")

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
st.pyplot(fig)


st.subheader("Box Plot - Age")

fig, ax = plt.subplots(figsize=(6,2))
ax.boxplot(
    df["Age"],
    orientation="horizontal"
)
st.pyplot(fig)


st.subheader("Box Plot - Annual Income")

fig, ax = plt.subplots(figsize=(6,2))
ax.boxplot(
    df["Annual Income (k$)"],
    orientation="horizontal"
)
st.pyplot(fig)


st.subheader("Box Plot - Spending Score")

fig, ax = plt.subplots(figsize=(6,2))
ax.boxplot(
    df["Spending Score (1-100)"],
    orientation="horizontal"
)
st.pyplot(fig)
   

# -----------------------------
# MODEL TRAINING
# -----------------------------
elif page == "🤖 Model Training":

    st.header("🤖 K-Means Model Training")

    X = df[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Elbow Method")

    wcss = []

    for i in range(1,11):
        model = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(1,11), wcss, marker="o")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    st.success("Optimal Number of Clusters = 5")

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("Model trained successfully.")

    st.subheader("Cluster Counts")
    st.dataframe(df["Cluster"].value_counts().sort_index())
  # -----------------------------
# CUSTOMER PREDICTION
# -----------------------------
elif page == "🎯 Customer Prediction":

    st.header("🎯 Customer Segmentation Prediction")

    age = st.number_input(
        "Enter Age",
        min_value=18,
        max_value=100,
        value=25
    )

    annual_income = st.number_input(
        "Enter Annual Income (k$)",
        min_value=1,
        max_value=200,
        value=60
    )

    spending_score = st.number_input(
        "Enter Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50
    )

    if st.button("Find Cluster"):

        # Load model and scaler
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # User input
        user_input = pd.DataFrame(
            [[age, annual_income, spending_score]],
            columns=[
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        # Scale input
        user_scaled = scaler.transform(user_input)

        # Predict
        cluster = kmeans.predict(user_scaled)[0]

        # Balloons
        st.balloons()

        st.success(f"Predicted Cluster : {cluster}")

        if cluster == 0:
            st.info("Low Income • Low Spending Customers")

        elif cluster == 1:
            st.info("High Income • High Spending Customers")

        elif cluster == 2:
            st.info("Average Income • Average Spending Customers")

        elif cluster == 3:
            st.info("High Income • Low Spending Customers")

        elif cluster == 4:
            st.info("Low Income • High Spending Customers")

        st.subheader("Customer Details")

        st.write("Age :", age)
        st.write("Annual Income (k$) :", annual_income)
        st.write("Spending Score :", spending_score)


# -----------------------------
# ABOUT
# -----------------------------
elif page == "ℹ️ About":

    st.header("About Project")

    st.write("""
This project performs Customer Segmentation using
the K-Means Clustering algorithm.

Dataset:
Mall Customers Dataset

Algorithm:
K-Means Clustering

Libraries Used:
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

Developed for AI & ML Internship Project.
""")
# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    **Customer Segmentation using K-Means Clustering**

    👨‍💻 **Developed by:** Durga Prasad Annamdevula

    🎓 **Course:** BCA

    🤖 **Domain:** Data Science, Artificial Intelligence & Machine Learning

    🛠️ **Tools:** Python, Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn

    © 2026 All Rights Reserved.
    """
)
