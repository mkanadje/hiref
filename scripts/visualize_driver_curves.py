"""
Visualize the non-linear driver relationship curves used in data generation.

This script creates plots showing how each driver impacts sales, helping to
understand and validate the curve choices (sigmoid, exponential, logarithmic, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create figure with subplots
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Driver Impact Curves - Non-Linear Relationships', fontsize=16, fontweight='bold')

# ============================================================================
# 1. TEMPERATURE (Sigmoid)
# ============================================================================
ax = axes[0, 0]
temperature = np.linspace(30, 100, 200)
temp_normalized = (temperature - 60) / 25
temp_impact = 400 * (1 / (1 + np.exp(-2 * temp_normalized)) - 0.5)

ax.plot(temperature, temp_impact, linewidth=2.5, color='orangered')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(60, color='gray', linestyle=':', alpha=0.5, label='Baseline (60°F)')
ax.set_xlabel('Temperature (°F)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Temperature → Sales\n(POSITIVE - Sigmoid)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 2. PRECIPITATION (Exponential Decay - Negative)
# ============================================================================
ax = axes[0, 1]
precipitation = np.linspace(0, 10, 200)
precip_normalized = precipitation / 5
precip_impact = -200 * (1 - np.exp(-0.8 * precip_normalized))

ax.plot(precipitation, precip_impact, linewidth=2.5, color='steelblue')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Precipitation (inches)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Precipitation → Sales\n(NEGATIVE - Exponential Decay)', fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================================================
# 3. GDP INDEX (Logarithmic)
# ============================================================================
ax = axes[0, 2]
gdp = np.linspace(95, 120, 200)
gdp_deviation = np.maximum(0.1, gdp - 95)
gdp_impact = 250 * np.log(gdp_deviation / 100 + 1) / np.log(1.15)

ax.plot(gdp, gdp_impact, linewidth=2.5, color='green')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(100, color='gray', linestyle=':', alpha=0.5, label='Baseline (100)')
ax.set_xlabel('GDP Index', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('GDP Index → Sales\n(POSITIVE - Logarithmic)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 4. UNEMPLOYMENT RATE (Exponential - Negative)
# ============================================================================
ax = axes[1, 0]
unemployment = np.linspace(3, 15, 200)
unemployment_normalized = (unemployment - 4) / 10
unemployment_impact = -350 * (np.exp(0.8 * unemployment_normalized) - 1)

ax.plot(unemployment, unemployment_impact, linewidth=2.5, color='darkred')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(4, color='gray', linestyle=':', alpha=0.5, label='Baseline (4%)')
ax.set_xlabel('Unemployment Rate (%)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Unemployment → Sales\n(NEGATIVE - Exponential)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 5. CONSUMER CONFIDENCE (Power Curve)
# ============================================================================
ax = axes[1, 1]
cci = np.linspace(60, 120, 200)
cci_normalized = (cci - 80) / 40
cci_impact = 200 * np.sign(cci_normalized) * (np.abs(cci_normalized) ** 1.3)

ax.plot(cci, cci_impact, linewidth=2.5, color='purple')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(80, color='gray', linestyle=':', alpha=0.5, label='Baseline (80)')
ax.set_xlabel('Consumer Confidence Index', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Consumer Confidence → Sales\n(POSITIVE - Power Curve x^1.3)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 6. PRICE (Power Curve - Negative, Economy Segment)
# ============================================================================
ax = axes[1, 2]
price_deviation_pct = np.linspace(-0.2, 0.3, 200)
base_demand = 2500  # Economy segment
price_elasticity = -1.8  # Economy segment
price_impact = base_demand * 1.5 * np.sign(price_deviation_pct) * (
    np.abs(price_deviation_pct) ** 1.2
) * abs(price_elasticity)

ax.plot(price_deviation_pct * 100, price_impact, linewidth=2.5, color='crimson')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Base Price')
ax.set_xlabel('Price Change (%)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Price → Sales (Economy)\n(NEGATIVE - Power Curve x^1.2)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 7. DISTRIBUTION (Sigmoid)
# ============================================================================
ax = axes[2, 0]
distribution = np.linspace(40, 100, 200)
expected_dist = 80  # Mid-tier
dist_deviation = (distribution - expected_dist) / 20
distribution_impact = 300 * (1 / (1 + np.exp(-2.5 * dist_deviation)) - 0.5)

ax.plot(distribution, distribution_impact, linewidth=2.5, color='teal')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(80, color='gray', linestyle=':', alpha=0.5, label='Expected (80%)')
ax.set_xlabel('Distribution Coverage (%)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Distribution → Sales\n(POSITIVE - Sigmoid)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 8. TV SPEND (Logarithmic)
# ============================================================================
ax = axes[2, 1]
tv_spend = np.linspace(0, 150, 200)
tv_normalized = tv_spend / 50
tv_impact = 350 * np.log(tv_normalized + 1) / np.log(3)

ax.plot(tv_spend, tv_impact, linewidth=2.5, color='navy')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('TV Spend ($1K)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('TV Spend → Sales\n(POSITIVE - Logarithmic)', fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================================================
# 9. DIGITAL SPEND (Square Root)
# ============================================================================
ax = axes[2, 2]
digital_spend = np.linspace(0, 100, 200)
digital_normalized = digital_spend / 30
digital_impact = 400 * np.sqrt(digital_normalized)

ax.plot(digital_spend, digital_impact, linewidth=2.5, color='dodgerblue')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Digital Spend ($1K)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Digital Spend → Sales\n(POSITIVE - Square Root)', fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================================================
# 10. TRADE SPEND (Near-Linear)
# ============================================================================
ax = axes[3, 0]
trade_spend = np.linspace(0, 80, 200)
trade_normalized = trade_spend / 20
trade_impact = 380 * (trade_normalized ** 0.9)

ax.plot(trade_spend, trade_impact, linewidth=2.5, color='darkgreen')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
# Add true linear for comparison
trade_impact_linear = 380 * trade_normalized
ax.plot(trade_spend, trade_impact_linear, linewidth=1.5, linestyle='--',
        color='gray', alpha=0.5, label='Linear (x^1.0)')
ax.set_xlabel('Trade Spend ($1K)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Trade Spend → Sales\n(POSITIVE - Near-Linear x^0.9)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# ============================================================================
# 11. DISCOUNT % (Exponential Saturation)
# ============================================================================
ax = axes[3, 1]
discount_pct = np.linspace(0, 40, 200)
discount_normalized = discount_pct / 20
discount_impact = 450 * (1 - np.exp(-1.2 * discount_normalized))

ax.plot(discount_pct, discount_impact, linewidth=2.5, color='orange')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Discount (%)', fontweight='bold')
ax.set_ylabel('Sales Impact (units)', fontweight='bold')
ax.set_title('Discount % → Sales\n(POSITIVE - Exp Saturation)', fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================================================
# 12. SUMMARY - All Curves Normalized
# ============================================================================
ax = axes[3, 2]

# Normalize all curves to -1 to 1 range for comparison
x = np.linspace(-1, 1, 200)

# Different curve types
sigmoid = 1 / (1 + np.exp(-3 * x))
exponential = np.sign(x) * (np.exp(abs(x)) - 1) / (np.e - 1)
logarithmic = np.sign(x) * np.log(abs(x) + 1) / np.log(2)
power_accel = np.sign(x) * (np.abs(x) ** 1.3)
power_decel = np.sign(x) * (np.abs(x) ** 0.9)
sqrt_curve = np.sign(x) * np.sqrt(np.abs(x))

ax.plot(x, sigmoid, label='Sigmoid', linewidth=2, alpha=0.8)
ax.plot(x, exponential, label='Exponential', linewidth=2, alpha=0.8)
ax.plot(x, logarithmic, label='Logarithmic', linewidth=2, alpha=0.8)
ax.plot(x, power_accel, label='Power (x^1.3)', linewidth=2, alpha=0.8)
ax.plot(x, sqrt_curve, label='Square Root (x^0.5)', linewidth=2, alpha=0.8)
ax.plot(x, power_decel, label='Near-Linear (x^0.9)', linewidth=2, alpha=0.8)
ax.plot(x, x, label='Linear (x^1.0)', linewidth=2, linestyle='--', color='black', alpha=0.5)

ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax.set_xlabel('Normalized Input', fontweight='bold')
ax.set_ylabel('Normalized Impact', fontweight='bold')
ax.set_title('Curve Comparison\n(All Normalized)', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('outputs/driver_curves_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Driver curves visualization saved to: outputs/driver_curves_visualization.png")

# Create a second figure showing just the summary statistics
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Create summary table data
curve_types = [
    'Temperature', 'Precipitation', 'GDP Index', 'Unemployment',
    'Consumer Conf.', 'Price', 'Distribution', 'TV Spend',
    'Digital Spend', 'Trade Spend', 'Discount %'
]
relationships = [
    'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE',
    'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE',
    'POSITIVE', 'POSITIVE', 'POSITIVE'
]
curve_shapes = [
    'Sigmoid', 'Exp Decay', 'Logarithmic', 'Exponential',
    'Power (1.3)', 'Power (1.2)', 'Sigmoid', 'Logarithmic',
    'Sqrt (0.5)', 'Linear (0.9)', 'Exp Saturation'
]
max_impacts = [
    '±200', '-200', '+250', '-350',
    '±200', 'Variable', '±150', '+350',
    '+400', '+380', '+450'
]

# Create text summary
ax2.axis('tight')
ax2.axis('off')

table_data = []
for i in range(len(curve_types)):
    sign = '✅' if relationships[i] == 'POSITIVE' else '❌'
    table_data.append([
        curve_types[i],
        f"{sign} {relationships[i]}",
        curve_shapes[i],
        max_impacts[i]
    ])

table = ax2.table(
    cellText=table_data,
    colLabels=['Driver', 'Relationship', 'Curve Type', 'Impact Range'],
    cellLoc='left',
    loc='center',
    colWidths=[0.25, 0.25, 0.25, 0.25]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style the header
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(curve_types) + 1):
    color = '#E7E6E6' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax2.set_title('Driver Relationship Summary Table',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/driver_summary_table.png', dpi=150, bbox_inches='tight')
print("✅ Driver summary table saved to: outputs/driver_summary_table.png")

plt.show()

print("\n" + "="*60)
print("DRIVER CURVE VISUALIZATION COMPLETE")
print("="*60)
print("\nGenerated 2 visualizations:")
print("  1. outputs/driver_curves_visualization.png - 12 curve plots")
print("  2. outputs/driver_summary_table.png - Summary table")
print("\nAll curves follow realistic business relationships:")
print("  ✅ 8 Positive relationships")
print("  ❌ 3 Negative relationships")
print("  📊 6 different curve types (sigmoid, exponential, log, power, sqrt, linear)")
