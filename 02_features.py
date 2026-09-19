import pandas as pd
import numpy as np

print("Building master dataset with features...")

# ── LOAD DATA ─────────────────────────────────────────
intl_train = pd.read_excel("data international_train.xlsx")
intl_test  = pd.read_excel("data international_test.xlsx")
flights    = pd.read_excel("flight_data.xlsx")

# ── FIX DATES ─────────────────────────────────────────
intl_train["Date"] = pd.to_datetime(intl_train["Date"])
intl_test["Date"]  = pd.to_datetime(intl_test["Date"])
flights["Date"]    = pd.to_datetime(flights["Date"])

# ── HANDLE 2022 MONTHLY FLIGHT DATA ───────────────────
# 2022 flight data is monthly, 2023+ is daily
# We flag this so the model knows
flights["is_monthly"] = flights["Date"].dt.year == 2022
print(f"2022 monthly flight records: {flights[flights['is_monthly']].shape[0]:,}")
print(f"2023+ daily flight records:  {flights[~flights['is_monthly']].shape[0]:,}")

# ── AGGREGATE FLIGHTS TO DAILY TOTALS ─────────────────
# Sum all flights per day (regardless of country/airline)
daily_flights = (
    flights.groupby("Date")
    .agg(
        total_p2p      = ("Total P2P",    "sum"),
        total_pax      = ("Total PAX",    "sum"),
        total_seats    = ("Total Seats",  "sum"),
        total_transfer = ("Total Transfer","sum"),
        total_transit  = ("Total Transit","sum"),
    )
    .reset_index()
)

# Calculate load factor correctly: PAX/Seats (not average of LF column)
daily_flights["load_factor"] = (
    daily_flights["total_pax"] / daily_flights["total_seats"]
)

print(f"\nDaily flight summary rows: {len(daily_flights):,}")

# ── COUNTRY-SPECIFIC P2P ──────────────────────────────
# For each departure country, get daily P2P
country_flights = (
    flights.groupby(["Date", "Departure Country Name"])
    .agg(country_p2p = ("Total P2P", "sum"))
    .reset_index()
    .rename(columns={"Departure Country Name": "flight_country"})
)

# ── BUILD NATIONALITY → FLIGHT COUNTRY MAPPING ────────
# Not perfect (departure country ≠ nationality) but useful where they match
# We use uppercase to match hotel nationality format
country_map = {
    "India":                "INDIA",
    "United Kingdom":       "UNITED KINGDOM",
    "Saudi Arabia":         "SAUDI ARABIA",
    "Egypt":                "EGYPT",
    "Turkey":               "TURKEY",
    "Qatar":                "QATAR",
    "Jordan":               "JORDAN",
    "Oman":                 "OMAN",
    "Bahrain":              "BAHRAIN",
    "Kuwait":               "KUWAIT",
    "Germany":              "GERMANY",
    "France":               "FRANCE",
    "China":                "CHINA",
    "United States":        "UNITED STATES OF AMERICA",
    "Russia":               "RUSSIAN FEDERATION",
    "Kazakhstan":           "KAZAKHSTAN",
    "Armenia":              "ARMENIA",
    "Poland":               "POLAND",
    "Philippines":          "PHILIPPINES",
    "Lebanon":              "LEBANON",
    "Pakistan":             "PAKISTAN",
}

# ── ADD CALENDAR FEATURES ─────────────────────────────
def add_calendar_features(df):
    df = df.copy()
    df["day_of_week"]  = df["Date"].dt.dayofweek      # 0=Monday
    df["day_of_month"] = df["Date"].dt.day
    df["month"]        = df["Date"].dt.month
    df["day_of_year"]  = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["Date"].dt.year
    df["is_weekend"]   = df["day_of_week"].isin([4, 5]).astype(int)
    return df

intl_train = add_calendar_features(intl_train)
intl_test  = add_calendar_features(intl_test)

# ── HANDLE MISSING VALUES ─────────────────────────────
# Same-Day Guests: 59.75% missing — create indicator + fill with 0
intl_train["same_day_missing"] = intl_train["Same-Day Guests"].isna().astype(int)
intl_train["same_day_filled"]  = intl_train["Same-Day Guests"].fillna(0)
intl_test["same_day_missing"]  = intl_test["Same-Day Guests"].isna().astype(int)
intl_test["same_day_filled"]   = intl_test["Same-Day Guests"].fillna(0)

# New Arrivals: tiny amount missing — forward fill
intl_train["New Arrivals"] = intl_train["New Arrivals"].ffill()
intl_test["New Arrivals"]  = intl_test["New Arrivals"].ffill()

print(f"\nSame-Day missing in train: {intl_train['same_day_missing'].mean():.1%}")
print(f"Same-Day missing in test:  {intl_test['same_day_missing'].mean():.1%}")

# ── JOIN FLIGHT DATA TO HOTEL DATA ────────────────────
intl_train = intl_train.merge(daily_flights, on="Date", how="left")
intl_test  = intl_test.merge(daily_flights,  on="Date", how="left")

print(f"\nAfter joining flights to hotel train: {len(intl_train):,} rows")
print(f"After joining flights to hotel test:  {len(intl_test):,} rows")

# ── ADD DIRECT FLIGHT MATCH FLAG ──────────────────────
# Flag whether this nationality has a matching departure country
intl_train["has_direct_flight"] = (
    intl_train["Nationality"].map(
        {v: 1 for v in country_map.values()}
    ).fillna(0).astype(int)
)
intl_test["has_direct_flight"] = (
    intl_test["Nationality"].map(
        {v: 1 for v in country_map.values()}
    ).fillna(0).astype(int)
)

print(f"\nNationalities with direct flight match: "
      f"{intl_train['has_direct_flight'].mean():.1%} of rows")

# ── ADD LAG FEATURES (training only — carefully) ──────
# Sort so lags are computed in time order, per nationality
intl_train = intl_train.sort_values(["Nationality", "Date"])

intl_train["guests_lag_1"]   = intl_train.groupby("Nationality")["Guests"].shift(1)
intl_train["guests_lag_7"]   = intl_train.groupby("Nationality")["Guests"].shift(7)
intl_train["guests_lag_28"]  = intl_train.groupby("Nationality")["Guests"].shift(28)
intl_train["guests_lag_365"] = intl_train.groupby("Nationality")["Guests"].shift(365)

intl_train["arrivals_lag_1"] = intl_train.groupby("Nationality")["New Arrivals"].shift(1)
intl_train["arrivals_lag_7"] = intl_train.groupby("Nationality")["New Arrivals"].shift(7)

intl_train["rolling_7"]  = (
    intl_train.groupby("Nationality")["Guests"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)
intl_train["rolling_28"] = (
    intl_train.groupby("Nationality")["Guests"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

print(f"\nLag features created.")
print(f"Rows lost to lag warmup: "
      f"{intl_train['guests_lag_1'].isna().sum():,} "
      f"(expected — first few rows per nationality)")

# ── ENCODE NATIONALITY ────────────────────────────────
intl_train["nationality_code"] = pd.Categorical(
    intl_train["Nationality"]
).codes
intl_test["nationality_code"] = pd.Categorical(
    intl_test["Nationality"]
).codes

# ── DROP COLUMNS WE DON'T NEED ────────────────────────
drop_cols = ["Residence (groups)", "Arrival City", "Destination"]
intl_train = intl_train.drop(
    columns=[c for c in drop_cols if c in intl_train.columns]
)
intl_test = intl_test.drop(
    columns=[c for c in drop_cols if c in intl_test.columns]
)

# ── SAVE CLEAN DATASETS ───────────────────────────────
intl_train.to_csv("intl_train_features.csv", index=False)
intl_test.to_csv("intl_test_features.csv",   index=False)

print("\nSaved:")
print("  intl_train_features.csv")
print("  intl_test_features.csv")

# ── QUICK SUMMARY ─────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE SUMMARY")
print("=" * 50)
print(f"Training rows:  {len(intl_train):,}")
print(f"Test rows:      {len(intl_test):,}")
print(f"Features built: {len(intl_train.columns)} columns")
print("\nColumns in training data:")
for col in intl_train.columns:
    print(f"  {col}")

print("\nStep 2 complete. Run 03_model.py next.")