# ============================================================
# Smart Kirana Store – Model Training Notebook (Google Colab)
# Run this notebook in Google Colab to train and export models
# then download the /models folder and paste into your project.
# ============================================================

# ── CELL 1: Install dependencies ──────────────────────────────
# !pip install scikit-learn pandas numpy matplotlib seaborn -q

# ── CELL 2: Mount Google Drive (to save models) ───────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ── CELL 3: Upload dataset ────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()   # Upload supermart_sales.csv

# ── CELL 4: Imports ───────────────────────────────────────────
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
print("✅ Libraries loaded")

# ── CELL 5: Load & clean data ─────────────────────────────────
DATASET_PATH = "supermart_sales.csv"   # change if different path

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
df.columns = [c.strip().lower().replace(" ", "_").replace("&","and") for c in df.columns]
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
df.dropna(subset=["order_date","sales","category","sub_category"], inplace=True)
df["discount"] = df["discount"].fillna(0)
df["profit"]   = df["profit"].fillna(0)
df["sales"]    = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

print(f"Dataset shape: {df.shape}")
df.head()

# ── CELL 6: Feature Engineering ───────────────────────────────
FESTIVAL_DATES = {
    "2016-10-30","2017-10-19","2018-11-07","2019-10-27",
    "2020-11-14","2021-11-04","2022-10-24","2023-11-12","2024-11-01",
}
FESTIVAL_WINDOW = 7

def get_season(m):
    if m in (12,1,2): return "Winter"
    elif m in (3,4,5): return "Summer"
    elif m in (6,7,8,9): return "Monsoon"
    else: return "Autumn"

def is_festival(date):
    for fd in FESTIVAL_DATES:
        if abs((date - pd.to_datetime(fd)).days) <= FESTIVAL_WINDOW:
            return 1
    return 0

PERISHABLE = {"Fresh Vegetables","Fresh Fruits","Dairy","Eggs, Meat & Fish",
              "Bakery","Organic Fruits","Organic Vegetables"}

df["day"]          = df["order_date"].dt.day
df["month"]        = df["order_date"].dt.month
df["year"]         = df["order_date"].dt.year
df["day_of_week"]  = df["order_date"].dt.dayofweek
df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
df["quarter"]      = df["order_date"].dt.quarter
df["is_weekend"]   = df["day_of_week"].isin([5,6]).astype(int)
df["is_month_start"] = (df["day"] <= 5).astype(int)
df["is_month_end"]   = (df["day"] >= 25).astype(int)
df["season"]       = df["month"].apply(get_season)
df["is_festival"]  = df["order_date"].apply(is_festival)
df["is_perishable"] = df["sub_category"].isin(PERISHABLE).astype(int)
df["net_sales"]    = df["sales"] * (1 - df["discount"])

print("✅ Features engineered")
df[["day","month","season","is_festival","is_perishable"]].head()

# ── CELL 7: Aggregate to daily level ──────────────────────────
agg = (
    df.groupby(["order_date","sub_category","category",
                "season","is_festival","is_perishable",
                "day","month","year","day_of_week",
                "week_of_year","quarter","is_weekend",
                "is_month_start","is_month_end"])
    .agg(
        total_sales=("sales","sum"),
        order_count=("order_id","count"),
        avg_discount=("discount","mean"),
    )
    .reset_index()
    .sort_values("order_date")
)

print(f"Aggregated shape: {agg.shape}")
agg.head()

# ── CELL 8: Encode categoricals ───────────────────────────────
le = LabelEncoder()
season_map = {"Winter":0,"Summer":1,"Monsoon":2,"Autumn":3}
agg["season_enc"]       = agg["season"].map(season_map).fillna(0).astype(int)
agg["sub_category_enc"] = le.fit_transform(agg["sub_category"].astype(str))
agg["category_enc"]     = le.fit_transform(agg["category"].astype(str))

FEATURE_COLS = [
    "day","month","year","day_of_week","week_of_year","quarter",
    "is_weekend","is_month_start","is_month_end",
    "is_festival","is_perishable",
    "avg_discount","order_count",
    "sub_category_enc","category_enc","season_enc",
]

X = agg[FEATURE_COLS].fillna(0)
y = agg["total_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── CELL 9: Train Linear Regression ───────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_preds = np.clip(lr.predict(X_test), 0, None)
lr_mae   = mean_absolute_error(y_test, lr_preds)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2    = lr.score(X_test, y_test)

print(f"Linear Regression → MAE: ₹{lr_mae:.2f}  RMSE: ₹{lr_rmse:.2f}  R²: {lr_r2:.4f}")

# ── CELL 10: Train Random Forest ──────────────────────────────
rf = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_split=5,
    random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_preds = np.clip(rf.predict(X_test), 0, None)
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2    = rf.score(X_test, y_test)

print(f"Random Forest     → MAE: ₹{rf_mae:.2f}  RMSE: ₹{rf_rmse:.2f}  R²: {rf_r2:.4f}")

# ── CELL 11: Feature Importance ──────────────────────────────
imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": rf.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=imp_df, x="importance", y="feature", palette="Oranges_r")
plt.title("Random Forest – Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
plt.show()
print(imp_df.head(10).to_string(index=False))

# ── CELL 12: Actual vs Predicted plot ─────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_preds, alpha=0.3, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_preds, alpha=0.3, color="darkorange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Random Forest")
plt.tight_layout()
plt.savefig("models/actual_vs_predicted.png", dpi=150)
plt.show()

# ── CELL 13: Save models ──────────────────────────────────────
with open("models/linear_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("✅ Models saved to models/ folder")
print("   → linear_regression.pkl")
print("   → random_forest.pkl")
print("   → feature_cols.pkl")

# ── CELL 14: Download models (Colab only) ────────────────────
# import shutil
# shutil.make_archive("smart_kirana_models", "zip", "models")
# files.download("smart_kirana_models.zip")
# print("📦 Download started – extract into your project's models/ folder")

# ── CELL 15: Model comparison summary ────────────────────────
print("\n" + "="*50)
print("   MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>10} {'R²':>8}")
print("-"*50)
print(f"{'Linear Regression':<22} {lr_mae:>8.2f} {lr_rmse:>10.2f} {lr_r2:>8.4f}")
print(f"{'Random Forest':<22} {rf_mae:>8.2f} {rf_rmse:>10.2f} {rf_r2:>8.4f}")
print("="*50)
winner = "Random Forest" if rf_rmse < lr_rmse else "Linear Regression"
print(f"\n🏆 Best Model: {winner}")
