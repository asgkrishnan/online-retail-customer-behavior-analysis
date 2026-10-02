# CGB1204 – Data Science Active Learning - 1
## Customer Purchase Behavior Analysis for Online Retail

**Student Details:**
- **Name:** Gopalakrishnan A S
- **Register Number:** 927625BIT0311
- **Year / Semester:** II Year / III Semester
- **Section:** IT A
- **Academic Year:** 2026 – 2027 (Odd Semester)

---

## 📌 Project Overview
This project presents an end-to-end Data Science pipeline to analyze customer purchasing behavior in an online retail store. Using transaction-level data, the project handles data cleaning, calculates core summary statistics, performs exploratory visual analysis, segments customers using RFM (Recency, Frequency, Monetary) analysis with K-Means clustering, and trains a Random Forest Machine Learning classifier to predict customer spending tiers for real-world business decision-making.

---

## 🏗️ Project Architecture

```
DS project/
├── data/
│   └── online_retail_sample.csv   # Pre-loaded sample dataset matching PPT schema
├── notebooks/
│   └── 01_Customer_Behavior_Analysis.ipynb  # Step-by-step Jupyter Notebook
├── src/
│   ├── generate_sample_data.py    # Synthetic dataset generator matching PPT
│   ├── preprocessing.py           # Preprocessing (missing values, duplicates, cancellations)
│   ├── statistics_eda.py          # Summary stats (Mean, Median, Mode, Variance, Std Dev) & plots
│   └── segmentation_model.py      # RFM Analysis, K-Means & RF Classifier
├── app.py                         # Interactive Streamlit Web Dashboard UI
├── requirements.txt               # Required Python libraries
└── README.md                      # Documentation
```

---

## 🚀 How to Run the Project

### Option A: Launch Interactive Streamlit Dashboard (Recommended)
Run the following command in PowerShell / Terminal:
```powershell
python -m streamlit run app.py
```
This opens an interactive dashboard where you can:
1. Upload your custom dataset CSV/Excel file or use the built-in dataset.
2. View Data Cleaning metrics & before/after dataset tables.
3. Inspect summary statistics (**Mean, Median, Mode, Variance, Standard Deviation**).
4. Explore interactive charts (Top Selling Products, Spending Distribution, Country Distribution, Monthly Trends).
5. Interact with **RFM Customer Segmentation** and test live predictions using the **ML Customer Tier Predictor**.

### Option B: Run Jupyter Notebook
Open Jupyter Lab or Notebook:
```powershell
jupyter notebook notebooks/01_Customer_Behavior_Analysis.ipynb
```

---

## 📥 How to Use Your Own Dataset File
If you have a custom dataset CSV or Excel file:
1. Simply place your file inside the `data/` directory (e.g. `data/my_retail_data.csv`).
2. Alternatively, launch the Streamlit app (`python -m streamlit run app.py`) and use the **Upload Custom File (CSV/Excel)** button in the left sidebar!

---

## 📊 Key Features Implemented (Matching PPT Requirements)
1. **Data Preprocessing:** Handled missing `CustomerID` & `Description`, dropped duplicate rows, filtered out cancelled orders (`Quantity <= 0`), computed `TotalAmount = Quantity * UnitPrice`.
2. **Statistical Analysis:** Calculated Mean, Median, Mode, Variance, and Standard Deviation for `Quantity`, `UnitPrice`, and `TotalAmount`.
3. **Visualizations:** Bar charts for top sellers, histograms for order spending, pie charts for customer distribution across countries.
4. **RFM Segmentation & Machine Learning:** Segmented customers into **High-Value**, **Medium-Value**, and **Low-Value** tiers using K-Means Clustering and Random Forest Classifier.
5. **Business Decision Support:** Actionable insights for VIP customer retention, targeted promotional discounts, and inventory stock management.
