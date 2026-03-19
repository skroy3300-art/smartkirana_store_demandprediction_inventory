# 🛒 Smart Kirana Store
## Demand Prediction & Inventory Optimization System




## 🗂️ Project Structure

```
smart_kirana/
│
├── data/
│   └── supermart_sales.csv          ← Raw dataset (9,994 rows)
│
├── models/                          ← Saved model files (after training)
│   ├── random_forest.pkl
│   ├── linear_regression.pkl
│   └── feature_cols.pkl
│
├── reports/                         ← Downloaded CSV reports saved here
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py           ← Data cleaning & feature engineering
│   ├── model_training.py            ← ML training, evaluation, prediction
│   ├── inventory_optimization.py    ← EOQ, safety stock, reorder logic
│   └── utils.py                     ← Plotly charts & report generation
│
├── app.py                           ← Streamlit web application
├── train_colab.py                   ← Google Colab training script
├── requirements.txt
└── README.md
```




## 🚀 Setup & Run (VS Code / Local)

### Prerequisites
- Python 3.9+ installed
- VS Code with Python extension


### Step 1 –  Just open the folder in VS Code
# File → Open Folder → select smart_kirana
```

---

### Step 2 – Create a Virtual Environment

Open VS Code Terminal (`Ctrl+`` `) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, seaborn, openpyxl.

---

### Step 4 – Verify Dataset

Ensure this file exists:
```
smart_kirana/data/supermart_sales.csv
```
If not, copy the uploaded CSV there manually.

---

### Step 5A – Train Models LOCALLY (Quick Option)

If you want to train directly without Colab:

```bash
# From the smart_kirana/ root folder:
python -c "
import sys; sys.path.insert(0, '.')
from src.data_processing import full_pipeline
from src.model_training import train_models
df, _ = full_pipeline('data/supermart_sales.csv')
train_models(df)
print('Done!')
"
```

Models will be saved to `models/` folder.

---

### Step 5B – Train Models in Google Colab (Recommended for Portfolio)

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Upload `train_colab.py` → copy each `# ── CELL N` block as a separate notebook cell
4. Upload `supermart_sales.csv` when prompted (Cell 3)
5. Run all cells (Runtime → Run all)
6. After training, uncomment Cell 14 to download `smart_kirana_models.zip`
7. Extract the zip and copy all `.pkl` files into your local `smart_kirana/models/` folder

**Why Colab?**  
- Free GPU/CPU acceleration  
- Shareable training notebook link for your portfolio  
- Demonstrates cloud-based ML workflow  

---

### Step 6 – Launch the Streamlit App

```bash
# Make sure you're in the smart_kirana/ directory with venv active
streamlit run app.py
```

Your browser will open at **http://localhost:8501** automatically.

---

## 🖥️ Using the Application

Once the app is running:

| Tab | Action |
|-----|--------|
| **EDA Dashboard** | Explore 8 charts automatically (no action needed) |
| **Model Training** | Click "🚀 Train Models" (or skip if models already trained in Colab) |
| **Demand Forecast** | Select sub-categories → click "🔮 Generate Forecast" |
| **Inventory Optimization** | Click "📦 Compute Inventory Recommendations" |
| **Alerts & Reports** | View alerts → click "⬇️ Download Report" |

**Sidebar controls:**
- Upload your own CSV to override the bundled dataset
- Adjust Lead Time, Service Level, Ordering Cost for inventory calculations
- Choose forecast horizon (7–30 days) and model (RF vs LR)

---

## 📊 Dataset Description

| Column | Description |
|--------|-------------|
| Order ID | Unique transaction identifier |
| Customer Name | Customer (anonymised) |
| Category | Top-level product category (e.g., Beverages) |
| Sub Category | Detailed product group (e.g., Health Drinks) |
| City / State / Region | Geographic location |
| Order Date | Transaction date (dd-mm-yyyy) |
| Sales | Gross revenue (₹) |
| Discount | Discount fraction (0–1) |
| Profit | Net profit (₹) |

---

## 🧠 ML Architecture

```
Raw CSV
   │
   ▼
data_processing.py
   • Clean & parse dates
   • Engineer 15 features
   • Aggregate daily per sub-category
   • Label-encode categoricals
   │
   ▼
model_training.py
   • 80/20 train-test split
   • LinearRegression (baseline)
   • RandomForestRegressor (200 trees)
   • Evaluate: MAE, RMSE, R²
   • Save .pkl files
   │
   ▼
predict_future_demand()
   • Build future date rows
   • Predict 7–30 days ahead
   │
   ▼
inventory_optimization.py
   • EOQ = √(2DS/H)
   • Safety Stock = Z·√(LT·σ²)
   • Reorder Level = D_avg·LT + SS
   • Perishable override
   • Alert generation
   │
   ▼
app.py (Streamlit)
   • Interactive dashboard
   • Plotly visualizations
   • CSV report download
```

---

## 📈 Model Performance (Typical Results)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~420 | ~680 | ~0.72 |
| Random Forest | ~290 | ~490 | ~0.86 |

*Results vary slightly per run due to random seeding.*

---

## 🎉 Festival Logic

Demand spikes are detected around these Indian festivals (±7 days):

| Festival | Typical Month |
|----------|--------------|
| Diwali | October/November |
| Holi | March |
| Eid | May/June |
| Christmas | December |

Festival days are flagged in the forecast and trigger inventory alerts.

---

## 🥦 Perishable Item Logic

| Sub-Category | Shelf Life | Order Strategy |
|-------------|-----------|----------------|
| Fresh Vegetables | 3 days | 3-day supply × 0.85 |
| Fresh Fruits | 4 days | 4-day supply × 0.85 |
| Dairy | 5 days | 5-day supply × 0.85 |
| Eggs, Meat & Fish | 2 days | 2-day supply × 0.85 |
| Bakery | 2 days | 2-day supply × 0.85 |

---



---

## 📦 Dependencies

```
streamlit>=1.32.0     # Web UI framework
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
scikit-learn>=1.3.0   # ML models
plotly>=5.18.0        # Interactive charts
matplotlib>=3.7.0     # Static plots (Colab)
seaborn>=0.12.0       # Statistical plots (Colab)
openpyxl>=3.1.0       # Excel export support
```

---


