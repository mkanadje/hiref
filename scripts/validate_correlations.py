"""
Validate that the generated data has correct correlation signs between drivers and sales.

This script checks:
1. Positive correlations: temperature, gdp_index, consumer_confidence, distribution,
                          tv_spend, digital_spend, trade_spend, discount_pct
2. Negative correlations: precipitation, unemployment_rate, price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the generated data
print("Loading data...")
df = pd.read_csv("sample_data/fmcg_hierarchical_data.csv")
print(f"Loaded {len(df):,} records\n")

# Define expected relationships
expected_relationships = {
    'temperature': 'POSITIVE',
    'precipitation': 'NEGATIVE',
    'gdp_index': 'POSITIVE',
    'unemployment_rate': 'NEGATIVE',
    'consumer_confidence': 'POSITIVE',
    'price': 'NEGATIVE',
    'distribution': 'POSITIVE',
    'tv_spend': 'POSITIVE',
    'digital_spend': 'POSITIVE',
    'trade_spend': 'POSITIVE',
    'discount_pct': 'POSITIVE'
}

# Calculate correlations with sales
print("="*70)
print("CORRELATION ANALYSIS: Driver → Sales")
print("="*70)

feature_cols = list(expected_relationships.keys())
correlations = df[feature_cols + ['sales']].corr()['sales'][feature_cols]

# Check if correlations match expected signs
results = []
all_correct = True

for feature, expected in expected_relationships.items():
    corr = correlations[feature]
    actual_sign = 'POSITIVE' if corr > 0 else 'NEGATIVE'
    is_correct = (actual_sign == expected)

    if is_correct:
        status = '✅ CORRECT'
    else:
        status = '❌ WRONG'
        all_correct = False

    results.append({
        'Feature': feature,
        'Expected': expected,
        'Correlation': corr,
        'Actual': actual_sign,
        'Status': status
    })

    print(f"{feature:25} | Expected: {expected:8} | Corr: {corr:+.4f} | {status}")

print("="*70)

if all_correct:
    print("\n🎉 SUCCESS! All driver correlations have the correct signs!")
else:
    print("\n⚠️  WARNING: Some correlations have incorrect signs. Check data generation logic.")

# Calculate summary statistics
print("\n" + "="*70)
print("CORRELATION MAGNITUDE SUMMARY")
print("="*70)

positive_corrs = [r['Correlation'] for r in results if r['Expected'] == 'POSITIVE']
negative_corrs = [r['Correlation'] for r in results if r['Expected'] == 'NEGATIVE']

print(f"\nPositive Relationships ({len(positive_corrs)} features):")
print(f"  Mean correlation: {np.mean(positive_corrs):+.4f}")
print(f"  Range: {np.min(positive_corrs):+.4f} to {np.max(positive_corrs):+.4f}")

print(f"\nNegative Relationships ({len(negative_corrs)} features):")
print(f"  Mean correlation: {np.mean(negative_corrs):+.4f}")
print(f"  Range: {np.max(negative_corrs):+.4f} to {np.min(negative_corrs):+.4f}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Correlation bar chart
ax = axes[0]
colors = ['green' if r['Expected'] == 'POSITIVE' else 'red' for r in results]
features = [r['Feature'] for r in results]
corr_values = [r['Correlation'] for r in results]

bars = ax.barh(features, corr_values, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Correlation with Sales', fontweight='bold', fontsize=12)
ax.set_title('Driver → Sales Correlations\n(Green=Expected Positive, Red=Expected Negative)',
             fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
            ha='left' if val > 0 else 'right', va='center', fontsize=9)

# Plot 2: Heatmap of all feature correlations
ax = axes[1]
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix\n(Check for multicollinearity)',
             fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('outputs/correlation_validation.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Visualization saved to: outputs/correlation_validation.png")

# Additional validation: Check for non-linearity
print("\n" + "="*70)
print("NON-LINEARITY VALIDATION")
print("="*70)

# Sample data for efficiency
sample_df = df.sample(min(10000, len(df)), random_state=42)

# Check R² of linear vs polynomial fits
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

non_linear_features = []

for feature in feature_cols:
    X = sample_df[[feature]].values
    y = sample_df['sales'].values

    # Linear fit
    lr_linear = LinearRegression()
    lr_linear.fit(X, y)
    r2_linear = r2_score(y, lr_linear.predict(X))

    # Polynomial fit (degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)
    r2_poly = r2_score(y, lr_poly.predict(X_poly))

    # If polynomial is significantly better, relationship is non-linear
    improvement = r2_poly - r2_linear

    if improvement > 0.05:  # 5% improvement threshold
        status = "📈 NON-LINEAR"
        non_linear_features.append(feature)
    else:
        status = "📉 ~Linear"

    print(f"{feature:25} | Linear R²: {r2_linear:.4f} | Poly R²: {r2_poly:.4f} | {status}")

print("\n" + "="*70)
print(f"✅ Found {len(non_linear_features)} features with non-linear relationships")
print(f"   (Expected: most features should show non-linearity)")

# Final summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"✅ Correlation signs: {'ALL CORRECT' if all_correct else 'SOME INCORRECT'}")
print(f"✅ Positive correlations: {len(positive_corrs)} features")
print(f"✅ Negative correlations: {len(negative_corrs)} features")
print(f"✅ Non-linear relationships: {len(non_linear_features)}/{len(feature_cols)} features")
print("="*70)

# Save validation results to file
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/correlation_validation_results.csv', index=False)
print(f"\n💾 Detailed results saved to: outputs/correlation_validation_results.csv")
