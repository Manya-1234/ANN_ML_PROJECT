import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

st.title("🚀 Interactive ML Pipeline Dashboard")
st.markdown("---")

# ---------------- SIDEBAR ----------------
problem_type = st.sidebar.selectbox("1. Select Problem Type", ["Classification", "Regression"])
uploaded_file = st.sidebar.file_uploader("2. Upload Dataset (CSV)", type="csv")

# ---------------- INIT SESSION STATE ----------------
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.clean_df = df.copy()

df = st.session_state.clean_df

if df is None:
    st.info("Upload a CSV file to start")
    st.stop()

# ---------------- TABS ----------------
tabs = st.tabs([
    "📊 Data & EDA",
    "🛠️ Cleaning",
    "🎯 Feature Selection",
    "🤖 Model Training",
    "📈 Performance"
])

# =====================================================
# TAB 1: EDA
# =====================================================
with tabs[0]:
    st.subheader("Exploratory Data Analysis")

    target_feature = st.selectbox(
        "Select Target Variable",
        df.columns,
        key="target_selector"
    )

    st.session_state.target_feature = target_feature

    all_features = [col for col in df.columns if col != target_feature]

    selected_features = st.multiselect(
        "Select Features",
        all_features,
        default=all_features[:4]
    )

    if selected_features:
        selected_df = df[selected_features]

        numeric_df = selected_df.select_dtypes(include=np.number)
        categorical_df = selected_df.select_dtypes(exclude=np.number)

        if len(categorical_df.columns) > 0:
            st.info(f"Ignoring categorical columns: {list(categorical_df.columns)}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📊 Data Summary")
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe())
            else:
                st.warning("No numeric columns")

        with col2:
            st.write("### 🔗 Correlation")
            if numeric_df.shape[1] > 1:
                fig_corr = px.imshow(
                    numeric_df.corr(),
                    text_auto=True,
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns")

        st.markdown("---")
        st.write("### 📉 PCA Visualization")

        if numeric_df.shape[1] >= 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(numeric_df.fillna(0))

            pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
            pca_df[target_feature] = df[target_feature].astype(str)

            fig_pca = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color=target_feature,
                title="PCA - 2D Projection"
            )
            st.plotly_chart(fig_pca, use_container_width=True)

# =====================================================
# TAB 2: CLEANING
# =====================================================
with tabs[1]:
    st.subheader("🛠️ Data Cleaning")

    clean_df = st.session_state.clean_df
    st.write("### Current Shape:", clean_df.shape)

    st.markdown("### Missing Values Handling")
    method = st.selectbox("Method", ["Mean", "Median", "Mode"])

    if st.button("Apply Missing Handling"):
        for col in clean_df.columns:
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                if method == "Mean":
                    clean_df[col].fillna(clean_df[col].mean(), inplace=True)
                elif method == "Median":
                    clean_df[col].fillna(clean_df[col].median(), inplace=True)
                else:
                    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
            else:
                clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)

        st.session_state.clean_df = clean_df
        st.success("Missing values handled")

    st.markdown("### Outlier Removal")
    outlier_method = st.selectbox("Outlier Method", ["IQR", "Isolation Forest"])

    if st.button("Remove Outliers"):
        numeric_cols = clean_df.select_dtypes(include=np.number).columns

        if outlier_method == "IQR":
            for col in numeric_cols:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]

        else:
            iso = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso.fit_predict(clean_df[numeric_cols]) == -1
            clean_df = clean_df[~outliers]

        st.session_state.clean_df = clean_df
        st.success(f"Outliers removed. New shape: {clean_df.shape}")

    st.markdown("### Remove Duplicates")

    if st.button("Remove Duplicates"):
        before = clean_df.shape[0]
        clean_df = clean_df.drop_duplicates()
        after = clean_df.shape[0]

        st.session_state.clean_df = clean_df
        st.success(f"Removed {before - after} duplicates")

    st.dataframe(st.session_state.clean_df.head())

# =====================================================
# TAB 3: FEATURE SELECTION
# =====================================================
with tabs[2]:
    st.subheader("Feature Selection")

    df = st.session_state.clean_df
    target_feature = st.session_state.target_feature

    X = df.drop(columns=[target_feature])
    y = df[target_feature]

    X_numeric = X.select_dtypes(include=np.number)

    method = st.selectbox("Method", ["Variance", "Correlation", "Mutual Info"])

    if method == "Variance":
        selector = VarianceThreshold(0.1)
        selector.fit(X_numeric)
        selected = X_numeric.columns[selector.get_support()]

    elif method == "Correlation":
        corr = X_numeric.corr().abs().mean().sort_values(ascending=False)
        selected = corr.index[:5]

    else:
        if problem_type == "Classification":
            mi = mutual_info_classif(X_numeric, y)
        else:
            mi = mutual_info_regression(X_numeric, y)

        selected = X_numeric.columns[np.argsort(mi)[-5:]]

    st.write("Recommended Features:", list(selected))

    st.session_state.final_features = st.multiselect(
        "Final Features",
        X.columns,
        default=list(selected)
    )

# =====================================================
# TAB 4: MODEL TRAINING
# =====================================================
with tabs[3]:
    st.subheader("Model Training")

    test_size = st.slider("Test Size %", 10, 50, 20)
    k = st.slider("K-Fold", 2, 10, 5)

    model_choice = st.selectbox("Model", [
        "Logistic Regression",
        "Linear Regression",
        "SVM",
        "Random Forest"
    ])

    if st.button("Train Model"):

        df = st.session_state.clean_df
        target_feature = st.session_state.target_feature
        final_features = st.session_state.final_features

        # =========================
        # 1. INPUT (X) + OUTPUT (y)
        # =========================
        X = df[final_features].copy()
        y = df[target_feature].copy()

        # =========================
        # 2. ENCODE TARGET (IMPORTANT FOR low/medium/high)
        # =========================
        if problem_type == "Classification":
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

        # =========================
        # 3. ENCODE FEATURES
        # =========================
        X = pd.get_dummies(X, drop_first=True)

        # =========================
        # 4. FEATURE SCALING
        # =========================
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # =========================
        # 5. TRAIN-TEST SPLIT
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size / 100,
            random_state=42
        )

        # =========================
        # 6. MODEL SELECTION
        # =========================
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
        else:
            model = SVC() if problem_type == "Classification" else SVR()

        # =========================
        # 7. CROSS VALIDATION
        # =========================
        cv = cross_validate(model, X_train, y_train, cv=k, return_train_score=True)

        # =========================
        # 8. TRAIN MODEL
        # =========================
        model.fit(X_train, y_train)

        # =========================
        # 9. STORE RESULTS
        # =========================
        st.session_state.update({
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "cv": cv
        })

        st.success("Model trained successfully on selected features + encoded target")

# =====================================================
# TAB 5: PERFORMANCE
# =====================================================
with tabs[4]:
    st.subheader("Performance")

    if "model" in st.session_state:

        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        y_pred = model.predict(X_test)

        if problem_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

        train_score = np.mean(st.session_state.cv["train_score"])
        test_score = np.mean(st.session_state.cv["test_score"])

        st.write("Train Score:", round(train_score, 2))
        st.write("Validation Score:", round(test_score, 2))

        # if train_score > test_score + 0.15:
            # st.error("Overfitting detected")
        # elif train_score < 0.5:
            # st.warning("Underfitting detected")

        
    else:
        st.info("Train model first")