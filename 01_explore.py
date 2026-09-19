import pandas as pd

print("Loading all datasets...")

# Load all 5 files
intl_train = pd.read_excel("data international_train.xlsx")
intl_test  = pd.read_excel("data international_test.xlsx")
dom_train  = pd.read_excel("data domestic_train.xlsx")
dom_test   = pd.read_excel("data domestic_test.xlsx")
flights    = pd.read_excel("flight_data.xlsx")

print("All files loaded successfully!\n")

# ── INTERNATIONAL HOTEL DATA ──────────────────────────
print("=" * 50)
print("INTERNATIONAL HOTEL DATA (TRAINING)")
print("=" * 50)
print(f"Rows:        {len(intl_train):,}")
print(f"Date range:  {intl_train['Date'].min()} → {intl_train['Date'].max()}")
print(f"Columns:     {list(intl_train.columns)}")
print(f"Nationalities ({intl_train['Nationality'].nunique()} total):")
print()

# Top 10 nationalities by total guests
top_nations = (
    intl_train.groupby("Nationality")["Guests"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("Top 10 nationalities by total hotel guests:")
for country, guests in top_nations.items():
    print(f"  {country:<30} {guests:>12,.0f} guests")

# ── FLIGHT DATA ───────────────────────────────────────
print()
print("=" * 50)
print("FLIGHT DATA")
print("=" * 50)
print(f"Rows:        {len(flights):,}")
print(f"Date range:  {flights['Date'].min()} → {flights['Date'].max()}")
print(f"Columns:     {list(flights.columns)}")
print()

# Top 10 departure countries by P2P passengers
p2p_col = "Total P2P"
country_col = [c for c in flights.columns if "country" in c.lower() and "name" in c.lower()]
country_col = country_col[0] if country_col else flights.columns[2]

top_routes = (
    flights.groupby(country_col)[p2p_col]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(f"Top 10 departure countries by P2P passengers:")
for country, pax in top_routes.items():
    print(f"  {country:<30} {pax:>12,.0f} P2P passengers")

# ── KEY RATIOS ────────────────────────────────────────
print()
print("=" * 50)
print("KEY NUMBERS FOR YOUR MODEL")
print("=" * 50)

total_guests    = intl_train["Guests"].sum()
total_arrivals  = intl_train["New Arrivals"].sum()
avg_stay        = total_guests / total_arrivals
total_p2p       = flights[p2p_col].sum()
total_pax       = flights["Total PAX"].sum()
total_seats     = flights["Total Seats"].sum()
avg_load_factor = total_pax / total_seats

print(f"Avg length of stay (Guests / New Arrivals): {avg_stay:.2f} nights")
print(f"Avg load factor (PAX / Seats):              {avg_load_factor:.2%}")
print(f"P2P share of total PAX:                     {total_p2p/total_pax:.2%}")
print()

# ── SEASONALITY ───────────────────────────────────────
print("=" * 50)
print("SEASONALITY (avg daily international guests by month)")
print("=" * 50)
intl_train["Date"] = pd.to_datetime(intl_train["Date"])
intl_train["Month"] = intl_train["Date"].dt.month
monthly = (
    intl_train.groupby("Month")["Guests"]
    .mean()
    .round(0)
)
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
for m, avg in monthly.items():
    bar = "█" * int(avg / 500)
    print(f"  {month_names[m-1]}: {avg:>8,.0f}  {bar}")

print()
print("Done! You now understand your data.")