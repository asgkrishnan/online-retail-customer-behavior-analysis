import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to system path
sys.path.append(os.path.abspath("src"))

from preprocessing import load_dataset, get_preprocessing_insights, preprocess_retail_data
from statistics_eda import (
    calculate_summary_statistics,
    plot_top_selling_products,
    plot_purchase_amount_distribution,
    plot_country_distribution,
    plot_monthly_revenue_trend,
    get_top_products
)
from segmentation_model import (
    compute_rfm_metrics,
    train_rfm_kmeans_segmentation,
    train_customer_value_classifier,
    predict_single_customer,
    plot_rfm_clusters
)
from generate_sample_data import generate_online_retail_dataset

# Streamlit Page Config
st.set_page_config(
    page_title="Online Retail Customer Behavior Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .status-banner {
        padding: 12px 20px;
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        border-radius: 8px;
        font-weight: 600;
        color: #065F46;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🛍️ Customer Purchase Behavior Analysis for Online Retail</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Course:</b> CGB1204 - Data Science Active Learning - 1 | <b>Student:</b> Gopalakrishnan A S (Reg: 927625BIT0311) | II Year IT A</div>', unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart-loaded.png", width=80)
st.sidebar.title("📌 Dataset & Settings")

data_source = st.sidebar.radio("Data Source:", ["Upload Custom CSV/Excel File", "Use Pre-loaded Sample Dataset"])

uploaded_file = None
file_loaded_name = "online_retail_sample.csv"
is_custom_uploaded = False

if data_source == "Upload Custom CSV/Excel File":
    uploaded_file = st.sidebar.file_uploader("📂 Choose your CSV or Excel file:", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            raw_df = load_dataset(uploaded_file)
            file_loaded_name = uploaded_file.name
            is_custom_uploaded = True
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            sample_path = "data/online_retail_sample.csv"
            if not os.path.exists(sample_path):
                generate_online_retail_dataset(sample_path)
            raw_df = load_dataset(sample_path)
    else:
        st.sidebar.info("💡 Drag & drop or browse your CSV file above. Showing sample data until file is selected.")
        sample_path = "data/online_retail_sample.csv"
        if not os.path.exists(sample_path):
            generate_online_retail_dataset(sample_path)
        raw_df = load_dataset(sample_path)
else:
    sample_path = "data/online_retail_sample.csv"
    if not os.path.exists(sample_path):
        generate_online_retail_dataset(sample_path)
    raw_df = load_dataset(sample_path)

# Display Status Banner
if is_custom_uploaded:
    st.markdown(f'<div class="status-banner">✅ Custom Dataset Loaded Successfully: <b>"{file_loaded_name}"</b> ({len(raw_df):,} rows, {len(raw_df.columns)} columns)</div>', unsafe_allow_html=True)
else:
    st.info(f"ℹ️ Currently analyzing dataset: **'{file_loaded_name}'** ({len(raw_df):,} rows). You can upload your own CSV file anytime using the sidebar on the left!")

# Preprocessing Data
clean_df = preprocess_retail_data(raw_df)

# Navigation Menu
menu_choice = st.sidebar.selectbox(
    "Select Module:",
    [
        "1. Overview & Data Preprocessing",
        "2. Statistics & Visual EDA",
        "3. Customer Segmentation (RFM)",
        "4. ML Customer Tier Prediction",
        "5. Business Recommendations"
    ]
)

# ----------------------------------------------------
# TAB 1: OVERVIEW & DATA PREPROCESSING
# ----------------------------------------------------
if menu_choice == "1. Overview & Data Preprocessing":
    st.header("🧹 Data Preprocessing & Cleaning")
    
    st.markdown("""
    **PPT Requirements Addressed:**
    - **Missing Values:** Handled missing `CustomerID` and `Description`.
    - **Duplicate Values:** Dropped duplicate transaction records.
    - **Outliers & Cancellations:** Filtered out negative quantities (`Quantity <= 0`) and cancelled orders (`InvoiceNo` starting with 'C').
    - **Data Transformation:** Calculated `TotalAmount = Quantity * UnitPrice`.
    """)
    
    insights = get_preprocessing_insights(raw_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Raw Records", f"{insights['total_records']:,}")
    col2.metric("Missing Customer IDs", f"{insights['missing_customer_id']:,}")
    col3.metric("Missing Descriptions", f"{insights['missing_description']:,}")
    col4.metric("Duplicates Dropped", f"{insights['duplicate_records']:,}")
    col5.metric("Cleaned Records", f"{len(clean_df):,}")
    
    st.divider()
    st.subheader("📋 Raw Uploaded Data (First 10 Rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    st.subheader("✨ Cleaned & Transformed Data (with TotalAmount)")
    st.dataframe(clean_df.head(10), use_container_width=True)

# ----------------------------------------------------
# TAB 2: STATISTICS & VISUAL EDA
# ----------------------------------------------------
elif menu_choice == "2. Statistics & Visual EDA":
    st.header("📊 Exploratory Data Analysis & Summary Statistics")
    
    st.subheader("1. Summary Statistics (Mean, Median, Mode, Variance, Std Dev)")
    stats_df = calculate_summary_statistics(clean_df)
    st.table(stats_df)
    
    st.divider()
    st.subheader("2. Visual Analysis Charts")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_top_selling_products(clean_df, top_n=10))
    with col2:
        st.pyplot(plot_purchase_amount_distribution(clean_df))
        
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_country_distribution(clean_df))
    with col4:
        st.pyplot(plot_monthly_revenue_trend(clean_df))

# ----------------------------------------------------
# TAB 3: CUSTOMER SEGMENTATION (RFM)
# ----------------------------------------------------
elif menu_choice == "3. Customer Segmentation (RFM)":
    st.header("🎯 Customer Segmentation via RFM & K-Means Clustering")
    
    if 'CustomerID' not in clean_df.columns or clean_df['CustomerID'].nunique() == 0:
        st.error("⚠️ The dataset does not contain valid CustomerID values required for RFM segmentation.")
    else:
        st.markdown("""
        **RFM Analysis Metrics:**
        - **Recency (R):** Days since customer's last purchase date.
        - **Frequency (F):** Total number of distinct orders placed.
        - **Monetary (M):** Total monetary spending ($).
        """)
        
        rfm_df = compute_rfm_metrics(clean_df)
        rfm_clustered, kmeans_model, scaler = train_rfm_kmeans_segmentation(rfm_df, n_clusters=3)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Customer Segment Summary")
            summary = rfm_clustered.groupby('CustomerTier').agg(
                CustomerCount=('CustomerID', 'count'),
                AvgRecency=('Recency', 'mean'),
                AvgFrequency=('Frequency', 'mean'),
                AvgMonetary=('Monetary', 'mean')
            ).reset_index()
            st.dataframe(summary.style.format({
                'AvgRecency': '{:.1f} days',
                'AvgFrequency': '{:.1f} orders',
                'AvgMonetary': '${:,.2f}'
            }), use_container_width=True)
            
        with col2:
            st.pyplot(plot_rfm_clusters(rfm_clustered))

# ----------------------------------------------------
# TAB 4: ML PREDICTION
# ----------------------------------------------------
elif menu_choice == "4. ML Customer Tier Prediction":
    st.header("🤖 Machine Learning Model: Customer Tier Prediction")
    
    if 'CustomerID' not in clean_df.columns or clean_df['CustomerID'].nunique() == 0:
        st.error("⚠️ Valid CustomerID values are needed to train the Machine Learning Classifier.")
    else:
        rfm_df = compute_rfm_metrics(clean_df)
        rfm_clustered, _, _ = train_rfm_kmeans_segmentation(rfm_df, n_clusters=3)
        model_res = train_customer_value_classifier(rfm_clustered)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classifier Model Accuracy", f"{model_res['accuracy'] * 100:.2f}%")
            st.markdown("**Feature Importance Breakdown:**")
            st.bar_chart(model_res['feature_importance'])
            
        with col2:
            st.markdown("**Confusion Matrix Heatmap:**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(model_res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['High', 'Medium', 'Low'], yticklabels=['High', 'Medium', 'Low'], ax=ax)
            ax.set_ylabel('Actual Tier')
            ax.set_xlabel('Predicted Tier')
            st.pyplot(fig)
            
        st.divider()
        st.subheader("🔮 Interactive Customer Tier Predictor Form")
        st.markdown("Enter transaction behavior parameters for a customer to predict their segment tier:")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            rec_input = st.number_input("Recency (Days since last purchase)", min_value=1, max_value=365, value=15)
        with c2:
            freq_input = st.number_input("Frequency (Total orders count)", min_value=1, max_value=200, value=12)
        with c3:
            mon_input = st.number_input("Monetary Value ($ Total amount spent)", min_value=1.0, max_value=100000.0, value=850.0)
            
        if st.button("Predict Customer Tier", type="primary"):
            tier_pred, probabilities = predict_single_customer(model_res['model'], rec_input, freq_input, mon_input)
            st.success(f"🎉 **Predicted Customer Segment:** {tier_pred}")
            st.json({k: f"{v*100:.1f}%" for k, v in probabilities.items()})

# ----------------------------------------------------
# TAB 5: BUSINESS RECOMMENDATIONS
# ----------------------------------------------------
elif menu_choice == "5. Business Recommendations":
    st.header("💡 Business Decision Support & Real-World Impact")
    
    st.markdown("""
    Based on customer purchasing behavior analysis and ML predictions, online retail businesses can execute the following targeted strategies:
    
    ---
    ### 👑 1. High-Value Customer Retention
    - **Strategy:** Provide VIP loyalty rewards, exclusive early access to new products, and dedicated customer support.
    - **Goal:** Protect high-revenue contributors and reduce churn.
    
    ### 🏷️ 2. Targeted Marketing & Personalization
    - **Strategy:** Send automated personalized discounts to **Medium-Value** customers who haven't purchased in > 30 days to re-engage them.
    - **Goal:** Increase purchase frequency and move customers into the High-Value tier.
    
    ### 📦 3. Optimized Inventory Stock Management
    - **Strategy:** Maintain high buffer stock for **Top 10 Best-Selling Items** (e.g., hanging t-light holders, cake stands) to prevent stockouts.
    - **Goal:** Minimize missed revenue and maximize fulfillment customer satisfaction.
    """)
