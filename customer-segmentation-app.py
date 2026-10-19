import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation & Cluster Analysis")
st.markdown("Segment customers into distinct behavioral groups using **K-Means Clustering**.")

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("customer_churn.csv")
    except FileNotFoundError:
        # Fallback synthetic dataset matching your notebook structure
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "CustomerID": [f"C{str(i).zfill(5)}" for i in range(1, n + 1)],
            "Tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.randint(20, 200, size=n),
            "TotalCharges": np.random.randint(150, 8000, size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaymentMethod": np.random.choice(["Credit Card", "Electronic Check", "Bank Transfer"], size=n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "Churn": np.random.choice([0, 1], size=n, p=[0.89, 0.11])
        })

uploaded_file = st.sidebar.file_uploader("Upload customer_churn.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Model Settings")
k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3)

# -----------------------------------------------------------------------------
# 4. Data Preprocessing (matching the notebook)
# -----------------------------------------------------------------------------
# Drop CustomerID and Churn from the clustering inputs
drop_cols = [c for c in ["CustomerID", "Churn"] if c in df.columns]
df_clustering = df.drop(columns=drop_cols)

cat_cols = df_clustering.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = df_clustering.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

X_preprocessed = preprocessor.fit_transform(df_clustering)

# -----------------------------------------------------------------------------
# 5. Fit K-Means
# -----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=k_clusters, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_preprocessed)

df_with_clusters = df.copy()
df_with_clusters["Cluster"] = cluster_labels
sil_score = silhouette_score(X_preprocessed, cluster_labels)

# -----------------------------------------------------------------------------
# 6. Dashboard Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📊 Diagnostics", "🗺️ 2D PCA Map", "🔎 Predict Profile"])

with tab1:
    st.subheader("Segmented Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Clusters (k)", k_clusters)
    c3.metric("Silhouette Score", f"{sil_score:.3f}")

    st.dataframe(df_with_clusters.head(10), use_container_width=True)

    st.subheader("Cluster Distribution")
    counts = df_with_clusters["Cluster"].value_counts().sort_index()
    st.bar_chart(counts)

with tab2:
    st.subheader("Elbow & Silhouette Evaluation")
    wcss = []
    scores = []
    test_range = range(2, 9)

    for k in test_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=5)
        km.fit(X_preprocessed)
        wcss.append(km.inertia_)
        scores.append(silhouette_score(X_preprocessed, km.labels_))

    col_a, col_b = st.columns(2)
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(list(test_range), wcss, marker="o", linestyle="--", color="#1f77b4")
        ax1.set_title("Elbow Method (Inertia)")
        ax1.set_xlabel("k")
        ax1.set_ylabel("WCSS")
        ax1.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(list(test_range), scores, marker="o", linestyle="--", color="#ff7f0e")
        ax2.set_title("Silhouette Score vs k")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Score")
        ax2.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig2)

with tab3:
    st.subheader("2D Projection via PCA")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_preprocessed)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = cluster_labels.astype(str)

    fig_pca, ax_pca = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        alpha=0.7,
        ax=ax_pca
    )
    ax_pca.set_title(f"Customer Clusters (k={k_clusters})")
    ax_pca.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig_pca)

    st.subheader("Cluster Averages (Numerical)")
    st.dataframe(df_with_clusters.groupby("Cluster")[num_cols].mean().style.format("{:.2f}"))

with tab4:
    st.subheader("Classify a New Customer Profile")

    with st.form("predict_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            in_tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            in_senior = st.selectbox("Senior Citizen", options=[0, 1], index=0)
        with f2:
            in_monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
            in_contract = st.selectbox("Contract", options=df["Contract"].unique())
        with f3:
            in_total = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=780.0)
            in_payment = st.selectbox("Payment Method", options=df["PaymentMethod"].unique())
            in_paperless = st.selectbox("Paperless Billing", options=df["PaperlessBilling"].unique())

        submit = st.form_submit_button("Find Segment")

    if submit:
        new_customer = pd.DataFrame([{
            "Tenure": in_tenure,
            "MonthlyCharges": in_monthly,
            "TotalCharges": in_total,
            "SeniorCitizen": in_senior,
            "Contract": in_contract,
            "PaymentMethod": in_payment,
            "PaperlessBilling": in_paperless
        }])

        new_scaled = preprocessor.transform(new_customer)
        predicted_cluster = kmeans.predict(new_scaled)[0]

        st.success(f"This customer belongs to **Cluster {predicted_cluster}**.")