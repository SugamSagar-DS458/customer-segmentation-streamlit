# 👥 Customer Segmentation & Cluster Analysis App

An interactive machine learning web application built with **Streamlit** and **Scikit-Learn** that performs customer segmentation on telecom customer data. The application applies **K-Means Clustering**, evaluates clustering performance using the **Elbow Method** and **Silhouette Analysis**, visualizes customer clusters in 2D via **Principal Component Analysis (PCA)**, and classifies new customer profiles in real time.

---

## 📌 Features

- **Data Ingestion & Preprocessing**: Handles missing columns, scales continuous numerical features (`StandardScaler`), and encodes categorical variables (`OneHotEncoder`) using scikit-learn's `ColumnTransformer`.
- **Dynamic K-Means Clustering**: Choose any value of $k$ from 2 to 8 dynamically via the sidebar slider.
- **Diagnostic Curves**:
  - **Elbow Curve (Inertia/WCSS)**: Find the optimal number of clusters.
  - **Silhouette Score Analysis**: Measure how well-separated the clusters are.
- **2D PCA Visualization**: Projects multi-dimensional preprocessed customer features onto a 2D scatter plot for intuitive cluster inspection.
- **Cluster Profiling**: Displays mean values across key numerical metrics (Tenure, Monthly Charges, Total Charges) per cluster.
- **Live Customer Profiling Form**: Input a new customer's attributes to predict and assign their corresponding cluster segment instantly.

---

## 🏗️ Project Architecture

```text
customer-segmentation-app/
├── app.py                  # Main Streamlit application
├── customer_churn.csv      # Dataset (or user-uploaded CSV)
├── requirements.txt        # Python library dependencies
├── .gitignore              # Files/folders excluded from git
└── README.md               # Project documentation
```

---

## 📊 Dataset Schema

The model utilizes demographic, financial, and contract features from the customer churn dataset:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Tenure` | Numeric | Number of months the customer has stayed with the company |
| `MonthlyCharges`| Numeric | Amount charged to the customer monthly |
| `TotalCharges` | Numeric | Total amount charged over the customer's tenure |
| `SeniorCitizen` | Categorical | Whether the customer is a senior citizen (`0`, `1`) |
| `Contract` | Categorical | Contract term (`Month-to-month`, `One year`, `Two year`) |
| `PaymentMethod` | Categorical | Payment method (`Credit Card`, `Electronic Check`, `Bank Transfer`) |
| `PaperlessBilling` | Categorical | Whether paperless billing is enabled (`Yes`, `No`) |

*Note: Identifiers like `CustomerID` and labels like `Churn` are dropped prior to clustering to ensure pure behavioral segmentation.*

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SugamSagar-DS458/customer-segmentation-streamlit.git
cd customer-segmentation-streamlit
```

### 2. Set Up a Virtual Environment
- **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run customer-segmentation-app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🚀 Deployment to Streamlit Community Cloud

1. Push your repository to your **GitHub** account.
2. Visit [share.streamlit.io](https://customer-segementation-and-cluster-analysis-sugamsagar-ds458.streamlit.app/) and log in with GitHub.
3. Click **New app**.
4. Select your repository, set the branch to `main`, and enter `app.py` as the **Main file path**.
5. Click **Deploy**!

---

## 🛠️ Built With

- **[Python](https://www.python.org/)** (v3.9+)
- **[Streamlit](https://streamlit.io/)** - Web framework
- **[scikit-learn](https://scikit-learn.org/)** - Preprocessing, K-Means, PCA, & Metrics
- **[pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** - Data wrangling
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)** - Visualizations

---

## 📄 License

Feel free to modify and use it for your learning and portfolios.
