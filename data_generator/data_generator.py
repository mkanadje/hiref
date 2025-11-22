import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# DEFINE HIERARCHIES
# =============================================================================

# Regions and their states (nested structure)
region_states = {
    "Northeast": ["Northfield", "Maplewood", "Riverdale", "Bridgeport", "Harborview"],
    "Southeast": ["Suncoast", "Palmetto", "Magnolia", "Tideland", "Ridgemont"],
    "Midwest": ["Lakewood", "Plainview", "Cedarburg", "Millbrook", "Dairyland"],
    "Southwest": ["Mesquite", "Redrock", "Silverado", "Sandstone", "Prairiewind"],
    "West": ["Coastline", "Evergreen", "Timberland", "Highpeak", "Canyonview"],
}

# Flatten to get all states with their regions
states_to_region = {}
for region, states in region_states.items():
    for state in states:
        states_to_region[state] = region

all_states = list(states_to_region.keys())
all_regions = list(region_states.keys())

# Segments
segments = ["Premium", "Mid-tier", "Economy"]

# Brands with their segments
brands_by_segment = {
    "Premium": [
        "Alpine Springs",
        "Golden Harvest",
        "Luxe Essence",
        "Royal Crest",
        "Velvet Touch",
        "Diamond Select",
        "Platinum Reserve",
        "Crown Jewel",
        "Elite Choice",
        "Noble Heritage",
        "Grand Legacy",
        "Prestige Gold",
    ],
    "Mid-tier": [
        "Fresh Valley",
        "Sunrise Farms",
        "Blue Ribbon",
        "Green Meadows",
        "Silver Lake",
        "Harvest Moon",
        "Morning Dew",
        "Crystal Creek",
        "Golden Fields",
        "Amber Grove",
        "Willow Brook",
        "Autumn Harvest",
    ],
    "Economy": [
        "Value Plus",
        "Smart Choice",
        "Budget Best",
        "Daily Savings",
        "Price Right",
        "Thrifty Pick",
        "Super Saver",
        "Everyday Low",
        "Family Value",
        "Quick Save",
        "Pocket Friendly",
        "Deal Finder",
    ],
}

# Create brand to segment mapping
brand_to_segment = {}
for segment, brands in brands_by_segment.items():
    for brand in brands:
        brand_to_segment[brand] = segment

all_brands = list(brand_to_segment.keys())

# Pack sizes (in oz)
pack_sizes = ["6oz", "8oz", "12oz", "16oz", "20oz", "24oz", "32oz", "48oz", "64oz"]

# =============================================================================
# GENERATE TIME PERIODS
# =============================================================================

# 6 years of monthly data
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 12, 31)

dates = pd.date_range(start=start_date, end=end_date, freq="MS")
num_months = len(dates)

print(
    f"Generating data for {num_months} months ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')})"
)

# =============================================================================
# CREATE SKU COMBINATIONS
# =============================================================================

# Each SKU is a combination of State x Brand x Pack
sku_combinations = list(product(all_states, all_brands, pack_sizes))
num_skus = len(sku_combinations)

print(f"Total SKUs: {num_skus}")
print(f"Total data points: {num_skus * num_months:,}")

# =============================================================================
# GENERATE BASE ATTRIBUTES
# =============================================================================

# State-level attributes (climate zones for temperature patterns)
state_climate = {
    # Northeast - Cold winters, mild summers
    "Northfield": {"base_temp": 50, "temp_amplitude": 25, "base_precip": 4},
    "Maplewood": {"base_temp": 48, "temp_amplitude": 26, "base_precip": 4.2},
    "Riverdale": {"base_temp": 52, "temp_amplitude": 24, "base_precip": 3.8},
    "Bridgeport": {"base_temp": 54, "temp_amplitude": 23, "base_precip": 4},
    "Harborview": {"base_temp": 49, "temp_amplitude": 25, "base_precip": 4.1},
    # Southeast - Warm, humid
    "Suncoast": {"base_temp": 72, "temp_amplitude": 10, "base_precip": 5.5},
    "Palmetto": {"base_temp": 64, "temp_amplitude": 18, "base_precip": 4.8},
    "Magnolia": {"base_temp": 60, "temp_amplitude": 20, "base_precip": 4.5},
    "Tideland": {"base_temp": 56, "temp_amplitude": 22, "base_precip": 4.2},
    "Ridgemont": {"base_temp": 58, "temp_amplitude": 21, "base_precip": 5},
    # Midwest - Cold winters, hot summers
    "Lakewood": {"base_temp": 50, "temp_amplitude": 28, "base_precip": 3.5},
    "Plainview": {"base_temp": 51, "temp_amplitude": 26, "base_precip": 3.8},
    "Cedarburg": {"base_temp": 46, "temp_amplitude": 28, "base_precip": 3.2},
    "Millbrook": {"base_temp": 52, "temp_amplitude": 26, "base_precip": 4},
    "Dairyland": {"base_temp": 44, "temp_amplitude": 30, "base_precip": 3.3},
    # Southwest - Hot, dry
    "Mesquite": {"base_temp": 66, "temp_amplitude": 18, "base_precip": 2.8},
    "Redrock": {"base_temp": 70, "temp_amplitude": 20, "base_precip": 1.2},
    "Silverado": {"base_temp": 62, "temp_amplitude": 22, "base_precip": 1},
    "Sandstone": {"base_temp": 58, "temp_amplitude": 20, "base_precip": 1.5},
    "Prairiewind": {"base_temp": 60, "temp_amplitude": 22, "base_precip": 3.5},
    # West - Mild, varied
    "Coastline": {"base_temp": 62, "temp_amplitude": 12, "base_precip": 2.5},
    "Evergreen": {"base_temp": 50, "temp_amplitude": 15, "base_precip": 4.5},
    "Timberland": {"base_temp": 52, "temp_amplitude": 14, "base_precip": 4.8},
    "Highpeak": {"base_temp": 48, "temp_amplitude": 24, "base_precip": 1.8},
    "Canyonview": {"base_temp": 50, "temp_amplitude": 26, "base_precip": 1.5},
}

# Segment-level base prices (per oz)
segment_base_price = {"Premium": 0.15, "Mid-tier": 0.10, "Economy": 0.06}

# Pack size multipliers for price (larger packs have lower per-oz price)
pack_price_multiplier = {
    "6oz": 1.4,
    "8oz": 1.3,
    "12oz": 1.15,
    "16oz": 1.0,
    "20oz": 0.95,
    "24oz": 0.9,
    "32oz": 0.85,
    "48oz": 0.78,
    "64oz": 0.75,
}

# Pack size in numeric oz for calculations
pack_oz = {
    "6oz": 6,
    "8oz": 8,
    "12oz": 12,
    "16oz": 16,
    "20oz": 20,
    "24oz": 24,
    "32oz": 32,
    "48oz": 48,
    "64oz": 64,
}

# =============================================================================
# GENERATE MACROECONOMIC DATA (same for all SKUs in a month)
# =============================================================================

# GDP Index (base 100, gradual growth with some fluctuations)
gdp_trend = np.linspace(100, 115, num_months)
gdp_noise = np.random.normal(0, 1.5, num_months)
gdp_index = gdp_trend + gdp_noise

# Unemployment rate (starts around 4%, spikes during COVID, recovers)
unemployment_base = np.ones(num_months) * 4
# COVID spike (March 2020 is month 15)
covid_start = 14
covid_peak = 17
for i in range(covid_start, min(covid_start + 12, num_months)):
    if i <= covid_peak:
        unemployment_base[i] = 4 + (i - covid_start) * 3
    else:
        unemployment_base[i] = unemployment_base[covid_peak] - (i - covid_peak) * 0.8
unemployment_base = np.clip(unemployment_base, 3.5, 15)
unemployment_rate = unemployment_base + np.random.normal(0, 0.3, num_months)

# Consumer Confidence Index (base 100, inverse to unemployment)
cci_base = 100 - (unemployment_rate - 4) * 5
cci_noise = np.random.normal(0, 3, num_months)
consumer_confidence = np.clip(cci_base + cci_noise, 50, 120)

# =============================================================================
# GENERATE THE DATASET
# =============================================================================

data_rows = []

for month_idx, date in enumerate(dates):
    month = date.month
    year = date.year

    # Seasonal factor (peaks in summer for beverages)
    seasonal_factor = 1 + 0.3 * np.sin((month - 4) * np.pi / 6)

    # Holiday factors
    holiday_factor = 1.0
    if month == 12:  # December holiday boost
        holiday_factor = 1.25
    elif month == 7:  # July 4th
        holiday_factor = 1.15
    elif month == 11:  # Thanksgiving
        holiday_factor = 1.1

    # Year-over-year trend (2% annual growth)
    trend_factor = 1 + 0.02 * (year - 2019 + (month - 1) / 12)

    # Macroeconomic factors for this month
    gdp = gdp_index[month_idx]
    unemployment = unemployment_rate[month_idx]
    cci = consumer_confidence[month_idx]

    for state, brand, pack in sku_combinations:
        region = states_to_region[state]
        segment = brand_to_segment[brand]

        # -----------------------------------------------------------------
        # WEATHER FEATURES (state-month level)
        # -----------------------------------------------------------------
        climate = state_climate[state]

        # Temperature with seasonal variation
        temp_seasonal = climate["temp_amplitude"] * np.sin((month - 4) * np.pi / 6)
        temperature = climate["base_temp"] + temp_seasonal + np.random.normal(0, 3)

        # Precipitation with some randomness
        precip_seasonal = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)
        precipitation = climate[
            "base_precip"
        ] * precip_seasonal + np.random.exponential(0.5)

        # -----------------------------------------------------------------
        # SKU-LEVEL FEATURES
        # -----------------------------------------------------------------

        # Base price (segment + pack size dependent)
        base_price_per_oz = segment_base_price[segment] * pack_price_multiplier[pack]
        base_price = base_price_per_oz * pack_oz[pack]

        # Add some price variation over time (inflation)
        price_inflation = 1 + 0.03 * (year - 2019)
        price = base_price * price_inflation + np.random.normal(0, 0.05 * base_price)

        # Distribution coverage (0-100%)
        # Premium brands have slightly lower coverage, economy higher
        if segment == "Premium":
            base_distribution = 65
        elif segment == "Mid-tier":
            base_distribution = 80
        else:
            base_distribution = 90

        # Distribution grows over time for newer brands
        distribution = min(
            98, base_distribution + np.random.normal(0, 5) + (year - 2019) * 2
        )

        # -----------------------------------------------------------------
        # PROMOTIONAL FEATURES
        # -----------------------------------------------------------------

        # TV Spend (in thousands) - higher for premium brands
        if segment == "Premium":
            tv_base = 50
        elif segment == "Mid-tier":
            tv_base = 30
        else:
            tv_base = 15

        # Higher spend in summer and holidays
        tv_seasonal = tv_base * (seasonal_factor * 0.5 + holiday_factor * 0.5)
        tv_spend = max(0, tv_seasonal + np.random.exponential(tv_base * 0.3))

        # Digital Spend (in thousands) - growing over time
        digital_growth = 1 + 0.15 * (year - 2019)
        digital_base = tv_base * 0.6 * digital_growth
        digital_spend = max(0, digital_base + np.random.exponential(digital_base * 0.4))

        # Trade Promotion (in thousands) - retailer incentives
        trade_base = 20 if segment == "Economy" else 15
        trade_spend = max(0, trade_base + np.random.exponential(trade_base * 0.5))

        # Discount percentage (economy has higher discounts)
        if segment == "Premium":
            discount_base = 5
        elif segment == "Mid-tier":
            discount_base = 10
        else:
            discount_base = 15

        # Higher discounts during promotions
        if holiday_factor > 1:
            discount_base *= 1.3

        discount_pct = min(40, max(0, discount_base + np.random.normal(0, 3)))

        # =================================================================
        # CALCULATE SALES WITH 70/30 BASELINE/DRIVER SPLIT
        # =================================================================

        # -----------------------------------------------------------------
        # PART 1: BASELINE (70%) - Inherent SKU potential
        # Based on: segment, pack, region, brand, seasonal patterns
        # -----------------------------------------------------------------

        # Base demand by segment
        if segment == "Premium":
            base_demand = 800
        elif segment == "Mid-tier":
            base_demand = 1500
        else:
            base_demand = 2500

        # Pack size effect (larger packs sell fewer units but more volume)
        pack_demand_factor = {
            "6oz": 1.6,
            "8oz": 1.5,
            "12oz": 1.3,
            "16oz": 1.0,
            "20oz": 0.9,
            "24oz": 0.8,
            "32oz": 0.6,
            "48oz": 0.45,
            "64oz": 0.4,
        }

        # Baseline includes: inherent demand + seasonal pattern + trend
        baseline_sales = (
            base_demand
            * pack_demand_factor[pack]
            * seasonal_factor
            * holiday_factor
            * trend_factor
        )

        # Add regional variation to baseline (brand strength varies by region)
        regional_baseline_mult = {
            "Northeast": 1.1,
            "Southeast": 0.95,
            "Midwest": 1.0,
            "Southwest": 0.9,
            "West": 1.05,
        }
        baseline_sales *= regional_baseline_mult[region]

        # -----------------------------------------------------------------
        # PART 2: DRIVER CONTRIBUTION (30%) - Time-varying factors
        # Weather, price, promotions, macro (all additive)
        # SCALED TO MATCH BASELINE MAGNITUDE
        # -----------------------------------------------------------------

        # Weather drivers (additive impact on units sold)
        temp_impact = 50 * (temperature - 60)  # Hot weather boosts sales significantly
        precip_impact = -30 * (precipitation - 3)  # Rain reduces sales

        # Price driver (elasticity-based, additive)
        price_elasticity = (
            -1.5 if segment == "Economy" else -1.2 if segment == "Mid-tier" else -0.8
        )
        price_deviation_pct = (price / base_price) - 1  # % deviation from base price
        price_impact = base_demand * 2.0 * price_deviation_pct * abs(price_elasticity)

        # Promotional drivers (properly scaled to match baseline)
        tv_impact = 8 * tv_spend  # Each $1K TV spend adds 8 units
        digital_impact = (
            10 * digital_spend
        )  # Digital is more efficient - 10 units per $1K
        trade_impact = (
            12 * trade_spend
        )  # Trade promotions most effective - 12 units per $1K
        discount_impact = 80 * discount_pct  # Each 1% discount adds 80 units

        # Distribution driver (deviation from segment baseline)
        if segment == "Premium":
            expected_dist = 65
        elif segment == "Mid-tier":
            expected_dist = 80
        else:
            expected_dist = 90
        distribution_impact = 50 * (distribution - expected_dist) / 10

        # Macroeconomic drivers (additive impacts, scaled up)
        gdp_impact = 100 * (gdp - 100) / 10  # GDP deviation from 100
        unemployment_impact = -150 * (
            unemployment - 4
        )  # High unemployment hurts significantly
        cci_impact = 50 * (cci - 100) / 10  # Consumer confidence

        # Total driver contribution (sum of all drivers)
        driver_contribution = (
            temp_impact
            + precip_impact
            + price_impact
            + tv_impact
            + digital_impact
            + trade_impact
            + discount_impact
            + distribution_impact
            + gdp_impact
            + unemployment_impact
            + cci_impact
        )

        # -----------------------------------------------------------------
        # FINAL SALES: 70% baseline + 30% drivers
        # -----------------------------------------------------------------

        sales = 0.70 * baseline_sales + 0.30 * driver_contribution

        # Add moderate random noise (reduced from 0.15 to 0.08 for clarity)
        noise = np.random.normal(1, 0.08)
        sales *= noise

        # Ensure non-negative
        sales = max(0, sales)

        # -----------------------------------------------------------------
        # APPEND ROW
        # -----------------------------------------------------------------

        data_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "year": year,
                "month": month,
                "region": region,
                "state": state,
                "segment": segment,
                "brand": brand,
                "pack": pack,
                # Weather features
                "temperature": round(temperature, 1),
                "precipitation": round(precipitation, 2),
                # Macroeconomic features
                "gdp_index": round(gdp, 2),
                "unemployment_rate": round(unemployment, 2),
                "consumer_confidence": round(cci, 2),
                # SKU features
                "price": round(price, 2),
                "distribution": round(distribution, 1),
                # Promotional features
                "tv_spend": round(tv_spend, 2),
                "digital_spend": round(digital_spend, 2),
                "trade_spend": round(trade_spend, 2),
                "discount_pct": round(discount_pct, 1),
                # Target
                "sales": round(sales, 0),
            }
        )

# =============================================================================
# CREATE DATAFRAME AND SAVE
# =============================================================================

df = pd.DataFrame(data_rows)

# Display info
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nSample data:")
print(df.head(10))

print(f"\nHierarchy counts:")
print(f"  Regions: {df['region'].nunique()}")
print(f"  States: {df['state'].nunique()}")
print(f"  Segments: {df['segment'].nunique()}")
print(f"  Brands: {df['brand'].nunique()}")
print(f"  Packs: {df['pack'].nunique()}")

print(f"\nSales statistics:")
print(df["sales"].describe())

print(f"\nSales by segment:")
print(df.groupby("segment")["sales"].mean().round(0))

print(f"\nSales by region:")
print(df.groupby("region")["sales"].mean().round(0))

# Save to CSV
output_path = "./sample_data/fmcg_hierarchical_data.csv"
df.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")

# Also save a summary file
summary_path = "./sample_data/fmcg_data_summary.txt"
with open(summary_path, "w") as f:
    f.write("FMCG Hierarchical Dataset Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Time Period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}\n"
    )
    f.write(f"Total Months: {num_months}\n")
    f.write(f"Total SKUs: {num_skus}\n")
    f.write(f"Total Records: {len(df):,}\n\n")

    f.write("Hierarchy Structure:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Regions: {df['region'].nunique()}\n")
    for region in all_regions:
        states = region_states[region]
        f.write(f"  {region}: {', '.join(states)}\n")

    f.write(f"\nSegments: {df['segment'].nunique()}\n")
    for segment in segments:
        brands = brands_by_segment[segment]
        f.write(f"  {segment}: {', '.join(brands)}\n")

    f.write(f"\nPack Sizes: {', '.join(pack_sizes)}\n")

    f.write("\n\nFeatures:\n")
    f.write("-" * 30 + "\n")
    f.write("Weather: temperature, precipitation\n")
    f.write("Macroeconomic: gdp_index, unemployment_rate, consumer_confidence\n")
    f.write("SKU-level: price, distribution\n")
    f.write("Promotional: tv_spend, digital_spend, trade_spend, discount_pct\n")
    f.write("Target: sales\n")

    f.write("\n\nSales Statistics:\n")
    f.write("-" * 30 + "\n")
    f.write(df["sales"].describe().to_string())

    f.write("\n\nSales by Segment:\n")
    f.write(
        df.groupby("segment")["sales"]
        .agg(["mean", "std", "min", "max"])
        .round(0)
        .to_string()
    )

    f.write("\n\nSales by Region:\n")
    f.write(
        df.groupby("region")["sales"]
        .agg(["mean", "std", "min", "max"])
        .round(0)
        .to_string()
    )

print(f"Summary saved to: {summary_path}")
