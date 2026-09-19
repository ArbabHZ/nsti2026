import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

print("Loading feature data...")
train = pd.read_csv("intl_train_features.csv", parse_dates=["Date"])

# ── DEFINE FEATURES ───────────────────────────────────
FEATURES = [
    # Calendar
    "day_of_week", "day_of_month", "month",
    "day_of_year", "week_of_year", "year", "is_weekend",
    # Hotel signals
    "New Arrivals", "same_day_filled", "same_day_missing",
    # Lag features
    "guests_lag_1", "guests_lag_7", "guests_lag_28", "guests_lag_365",
    "arrivals_lag_1", "arrivals_lag_7",
    "rolling_7", "rolling_28",
    # Flight signals
    "total_p2p", "total_seats", "load_factor", "total_transfer",
    # Country
    "nationality_code", "has_direct_flight",
]

TARGET = "Guests"

# ── TIME-BASED TRAIN / VALIDATION SPLIT ───────────────
# Train on 2022-2024, validate on 2025
# This is the ONLY correct way to split time series data
cutoff = pd.Timestamp("2025-01-01")

df_train = train[train["Date"] <  cutoff].copy()
df_val   = train[train["Date"] >= cutoff].copy()

# Drop rows where key lag features are NaN
df_train = df_train.dropna(subset=["guests_lag_1", "guests_lag_7"])
df_val   = df_val.dropna(subset=["guests_lag_1", "guests_lag_7"])

# Fill remaining NaN values in all feature columns
# (lag_28, lag_365 may still have NaN for early rows)
df_train[FEATURES] = df_train[FEATURES].fillna(df_train[FEATURES].median())
df_val[FEATURES]   = df_val[FEATURES].fillna(df_train[FEATURES].median())

print(f"Training rows:   {len(df_train):,}  ({df_train['Date'].min().date()} → {df_train['Date'].max().date()})")
print(f"Validation rows: {len(df_val):,}  ({df_val['Date'].min().date()} → {df_val['Date'].max().date()})")

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val   = df_val[FEATURES]
y_val   = df_val[TARGET]

# ── BASELINE MODEL A: NAIVE ───────────────────────────
# Predict today = yesterday
naive_preds = df_val["guests_lag_1"].fillna(df_val["guests_lag_7"])
naive_wmape = (
    (y_val - naive_preds).abs().sum() / y_val.sum() * 100
)
print(f"\nModel A — Naive baseline WMAPE:    {naive_wmape:.2f}%")

# ── BASELINE MODEL B: SEASONAL ────────────────────────
# Predict today = same day last year
seasonal_preds = df_val["guests_lag_365"].fillna(naive_preds)
seasonal_wmape = (
    (y_val - seasonal_preds).abs().sum() / y_val.sum() * 100
)
print(f"Model B — Seasonal baseline WMAPE: {seasonal_wmape:.2f}%")

# ── MODEL C: ML MODEL ─────────────────────────────────
print("\nTraining ML model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)

ml_preds = model.predict(X_val)
ml_preds = np.maximum(ml_preds, 0)  # no negative guests

ml_wmape = (
    (y_val - ml_preds).abs().sum() / y_val.sum() * 100
)
print(f"Model C — ML model WMAPE:          {ml_wmape:.2f}%")

# ── WMAPE BY NATIONALITY ──────────────────────────────
print("\n" + "=" * 50)
print("WMAPE BY TOP 10 NATIONALITIES")
print("=" * 50)
df_val = df_val.copy()
df_val["ml_pred"] = ml_preds

nationality_wmape = []
for nat in df_val["Nationality"].unique():
    mask   = df_val["Nationality"] == nat
    actual = y_val[mask]
    pred   = df_val.loc[mask, "ml_pred"]
    if actual.sum() > 0:
        w = (actual - pred).abs().sum() / actual.sum() * 100
        nationality_wmape.append((nat, w, actual.sum()))

nationality_wmape = sorted(nationality_wmape, key=lambda x: -x[2])
for nat, w, vol in nationality_wmape[:10]:
    bar = "█" * min(int(w / 2), 30)
    print(f"  {nat:<35} {w:>6.1f}%  {bar}")

# ── FEATURE IMPORTANCE ────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (sensitivity analysis)")
print("=" * 50)
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

for feat, imp in importance.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ── MONTHLY ACCURACY ──────────────────────────────────
print("\n" + "=" * 50)
print("MONTHLY WMAPE (validation period 2025)")
print("=" * 50)
df_val["month_name"] = df_val["Date"].dt.strftime("%b %Y")
for month in sorted(df_val["Date"].dt.to_period("M").unique()):
    mask   = df_val["Date"].dt.to_period("M") == month
    actual = y_val[mask]
    pred   = df_val.loc[mask, "ml_pred"]
    if actual.sum() > 0:
        w = (actual - pred).abs().sum() / actual.sum() * 100
        print(f"  {str(month):<12} {w:>6.1f}%")

# ── SAVE MODEL AND PREDICTIONS ────────────────────────
import joblib
joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")

# Save validation predictions for charting later
df_val["actual"]    = y_val.values
df_val["predicted"] = ml_preds
df_val[["Date", "Nationality", "actual", "predicted"]].to_csv(
    "validation_predictions.csv", index=False
)
print("Validation predictions saved to validation_predictions.csv")

# ── FINAL SUMMARY ─────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY FOR YOUR SLIDES")
print("=" * 50)
print(f"  Naive baseline WMAPE:    {naive_wmape:.1f}%")
print(f"  Seasonal baseline WMAPE: {seasonal_wmape:.1f}%")
print(f"  ML model WMAPE:          {ml_wmape:.1f}%")
print(f"  Top feature:             {importance.index[0]}")
print(f"  2nd feature:             {importance.index[1]}")
print(f"  3rd feature:             {importance.index[2]}")
print()
print("These numbers go directly on Slide 6 and Slide 7.")
print("\nStep 3 complete. Run 04_simulator.py next.")