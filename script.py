# ============================================
# U.S. GOLD RESERVES ML PIPELINE
# Supervised + Unsupervised Learning
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ============================================
# 1️⃣ DATA LOADING
# ============================================

def load_data(path):
    df = pd.read_csv(path)
    df["Record Date"] = pd.to_datetime(df["Record Date"])
    df = df.sort_values("Record Date")
    return df


# ============================================
# 2️⃣ FEATURE ENGINEERING
# ============================================

def build_monthly_features(df):
    monthly = (
        df.groupby("Record Date")
        .agg({
            "Fine Troy Ounces": "sum",
            "Book Value": "sum"
        })
        .reset_index()
    )

    # Time-based features
    monthly["Ounces_Change"] = monthly["Fine Troy Ounces"].diff()
    monthly["BookValue_Change"] = monthly["Book Value"].diff()

    monthly["Ounces_Rolling_3M"] = monthly["Fine Troy Ounces"].rolling(3).mean()
    monthly["BookValue_Rolling_3M"] = monthly["Book Value"].rolling(3).mean()

    monthly["Month"] = monthly["Record Date"].dt.month
    monthly["Year"] = monthly["Record Date"].dt.year

    return monthly.dropna()


# ============================================
# 3️⃣ EXPLORATORY DATA ANALYSIS
# ============================================

def plot_book_value(monthly):
    plt.figure()
    plt.plot(monthly["Record Date"], monthly["Book Value"])
    plt.title("Total U.S. Gold Book Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Book Value (USD)")
    plt.tight_layout()
    plt.show()


# ============================================
# 4️⃣ UNSUPERVISED LEARNING (REGIME DETECTION)
# ============================================

def run_clustering(monthly, n_clusters=3):
    features = [
        "Fine Troy Ounces",
        "Book Value",
        "Ounces_Change",
        "BookValue_Change"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(monthly[features])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    monthly["Reserve_Regime"] = model.fit_predict(X_scaled)

    return monthly, model


def plot_regimes(monthly):
    plt.figure()
    for regime in sorted(monthly["Reserve_Regime"].unique()):
        subset = monthly[monthly["Reserve_Regime"] == regime]
        plt.scatter(
            subset["Record Date"],
            subset["Book Value"],
            label=f"Regime {regime}",
            s=15
        )

    plt.legend()
    plt.title("Gold Reserve Regimes (Unsupervised Learning)")
    plt.xlabel("Date")
    plt.ylabel("Book Value")
    plt.tight_layout()
    plt.show()


# ============================================
# 5️⃣ SUPERVISED LEARNING (TIME-SERIES SAFE)
# ============================================

def train_supervised_models(monthly):
    X = monthly[
        ["Fine Troy Ounces", "Ounces_Change", "Ounces_Rolling_3M", "Month"]
    ]
    y = monthly["Book Value"]

    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        "Linear Regression": [],
        "Random Forest": []
    }

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds_lr = lr.predict(X_test)

        results["Linear Regression"].append([
            mean_absolute_error(y_test, preds_lr),
            np.sqrt(mean_squared_error(y_test, preds_lr)),
            r2_score(y_test, preds_lr)
        ])

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)

        results["Random Forest"].append([
            mean_absolute_error(y_test, preds_rf),
            np.sqrt(mean_squared_error(y_test, preds_rf)),
            
            r2_score(y_test, preds_rf)
        ])

    # Average metrics
    summary = {
        model: np.mean(scores, axis=0)
        for model, scores in results.items()
    }

    return summary


# ============================================
# 6️⃣ MAIN PIPELINE
# ==========================================

def main():
    DATA_PATH = "TreasGold_20120131_20251231.csv"

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building features...")
    monthly = build_monthly_features(df)

    print("Running EDA...")
    plot_book_value(monthly)

    print("Running unsupervised clustering...")
    monthly, cluster_model = run_clustering(monthly)
    plot_regimes(monthly)

    print("Training supervised models...")
    results = train_supervised_models(monthly)

    print("\nModel Performance (MAE, RMSE, R²):")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
