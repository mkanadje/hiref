# Driver Relationship Documentation

This document describes the non-linear relationships between drivers and sales in the FMCG hierarchical regression model.

## Overview

All driver impacts use **realistic non-linear distribution curves** based on business logic and economic theory. These ensure that the model captures real-world phenomena like diminishing returns, saturation effects, and accelerating impacts.

---

## Driver Relationships Summary

| Driver | Relationship | Curve Type | Business Rationale |
|--------|--------------|------------|-------------------|
| **Temperature** | ✅ POSITIVE | Sigmoid | Hot weather boosts beverage sales with optimal range |
| **Precipitation** | ❌ NEGATIVE | Exponential Decay | Rain discourages shopping/consumption |
| **GDP Index** | ✅ POSITIVE | Logarithmic | Economic growth helps but with diminishing returns |
| **Unemployment** | ❌ NEGATIVE | Exponential | High unemployment exponentially hurts spending |
| **Consumer Confidence** | ✅ POSITIVE | Power Curve | Confidence boosts spending with acceleration |
| **Price** | ❌ NEGATIVE | Power Curve | Price increases reduce demand (elasticity) |
| **Distribution** | ✅ POSITIVE | Sigmoid | Coverage drives sales with network effects |
| **TV Spend** | ✅ POSITIVE | Logarithmic | Awareness saturates as spend increases |
| **Digital Spend** | ✅ POSITIVE | Square Root | Efficient but with moderate saturation |
| **Trade Spend** | ✅ POSITIVE | Near-Linear | Direct incentives = predictable ROI |
| **Discount %** | ✅ POSITIVE | Exponential Saturation | Strong response at mid-range discounts |

---

## Detailed Mathematical Formulations

### 1. Temperature (POSITIVE - Sigmoid)

```python
temp_normalized = (temperature - 60) / 25
temp_impact = 400 * (1 / (1 + exp(-2 * temp_normalized)) - 0.5)
```

**Why Sigmoid?**
- Beverage sales have an optimal temperature range (60-85°F)
- Too cold: people don't want cold drinks (negative impact)
- Moderately warm: linear increase in demand
- Hot weather: strong positive impact but eventually saturates
- S-shaped curve captures this three-phase behavior

**Impact Range:** -200 to +200 units

---

### 2. Precipitation (NEGATIVE - Exponential Decay)

```python
precip_normalized = precipitation / 5
precip_impact = -200 * (1 - exp(-0.8 * precip_normalized))
```

**Why Exponential Decay?**
- Light rain: modest negative impact (people still shop)
- Moderate rain: stronger deterrent
- Heavy rain: severe impact approaching maximum penalty
- Exponential captures accelerating discouragement

**Impact Range:** 0 to -200 units (always negative)

---

### 3. GDP Index (POSITIVE - Logarithmic)

```python
gdp_deviation = max(0.1, gdp - 95)
gdp_impact = 250 * log(gdp_deviation / 100 + 1) / log(1.15)
```

**Why Logarithmic?**
- Classic economic diminishing returns
- Early GDP growth (95→100): significant boost to discretionary spending
- Later growth (110→115): smaller marginal impact
- Mature economies see saturation in consumer behavior

**Impact Range:** 0 to ~250 units

---

### 4. Unemployment Rate (NEGATIVE - Exponential)

```python
unemployment_normalized = (unemployment - 4) / 10
unemployment_impact = -350 * (exp(0.8 * unemployment_normalized) - 1)
```

**Why Exponential?**
- Unemployment has compounding psychological effects
- 4%→6%: moderate negative impact (some concern)
- 6%→10%: strong negative impact (widespread worry)
- 10%→15%: catastrophic impact (fear-driven cutbacks)
- Each percentage point hurts more than the previous

**Impact Range:** 0 to -350 units (increasingly severe)

---

### 5. Consumer Confidence (POSITIVE - Power Curve)

```python
cci_normalized = (cci - 80) / 40
cci_impact = 200 * sign(cci_normalized) * (|cci_normalized| ** 1.3)
```

**Why Power Curve (exponent > 1)?**
- Confidence has accelerating effects on spending
- Low confidence (60-80): modest negative impact
- Normal confidence (80-100): baseline behavior
- High confidence (100-120): disproportionate boost (people splurge)
- Power exponent 1.3 captures this acceleration

**Impact Range:** -200 to +200 units (accelerating at extremes)

---

### 6. Price (NEGATIVE - Power Curve)

```python
price_deviation_pct = (price / base_price) - 1
price_impact = base_demand * 1.5 * sign(price_deviation_pct) * (|price_deviation_pct| ** 1.2) * |price_elasticity|
```

**Elasticity by Segment:**
- Economy: -1.8 (highly sensitive)
- Mid-tier: -1.3 (moderate sensitivity)
- Premium: -0.9 (less sensitive)

**Why Power Curve?**
- Small price changes (±5%): roughly linear response
- Large price increases (>10%): exponential demand loss (sticker shock)
- Captures consumer psychology around "fair price" thresholds

**Impact Range:** Varies by segment and base demand

---

### 7. Distribution (POSITIVE - Sigmoid)

```python
dist_deviation = (distribution - expected_dist) / 20
distribution_impact = 300 * (1 / (1 + exp(-2.5 * dist_deviation)) - 0.5)
```

**Expected Distribution:**
- Premium: 65%
- Mid-tier: 80%
- Economy: 90%

**Why Sigmoid?**
- Low distribution (0-40%): limited availability severely constrains sales
- Growing distribution (40-80%): rapid sales growth (network effects)
- High distribution (80-100%): diminishing returns (market saturation)
- S-curve captures these three phases

**Impact Range:** -150 to +150 units

---

### 8. TV Spend (POSITIVE - Logarithmic)

```python
tv_normalized = tv_spend / 50
tv_impact = 350 * log(tv_normalized + 1) / log(3)
```

**Why Logarithmic?**
- Classic advertising saturation curve
- First $10K of spend: high ROI (reaching untapped audience)
- Next $20K: moderate ROI (reaching harder-to-reach segments)
- Beyond $50K: low ROI (audience already saturated)
- Logarithm captures diminishing marginal returns

**Impact Range:** 0 to ~350 units

---

### 9. Digital Spend (POSITIVE - Square Root)

```python
digital_normalized = digital_spend / 30
digital_impact = 400 * sqrt(digital_normalized)
```

**Why Square Root?**
- Digital marketing is more efficient than TV (lower saturation)
- Square root (exponent 0.5) provides:
  - Better than linear returns initially
  - Less severe saturation than logarithmic
- Reflects targeted nature of digital vs. broadcast TV

**Impact Range:** 0 to ~400 units

---

### 10. Trade Spend (POSITIVE - Near-Linear)

```python
trade_normalized = trade_spend / 20
trade_impact = 380 * (trade_normalized ** 0.9)
```

**Why Near-Linear (exponent 0.9)?**
- Trade promotions (retailer incentives) have most predictable ROI
- Direct incentives → direct sales lift
- Slight sublinearity (0.9 vs 1.0) accounts for:
  - Some retailer saturation
  - Diminishing shelf space availability
- Most linear of all promotional drivers

**Impact Range:** 0 to ~380 units (nearly proportional)

---

### 11. Discount % (POSITIVE - Exponential Saturation)

```python
discount_normalized = discount_pct / 20
discount_impact = 450 * (1 - exp(-1.2 * discount_normalized))
```

**Why Exponential Saturation?**
- 0-5% discount: modest response (regular shoppers)
- 10-20% discount: strong response (deal-seekers activated)
- 25%+ discount: saturation (everyone who will buy already did)
- Exponential saturation (1 - exp) captures this threshold effect

**Impact Range:** 0 to ~450 units (strongest promotional driver)

---

## Curve Characteristics Summary

### Logarithmic Curves (GDP, TV Spend)
- **Shape:** Steep initially, then flatten
- **Use case:** Diminishing returns, saturation effects
- **Formula:** `log(x + 1)`

### Exponential Curves (Unemployment, Precipitation)
- **Shape:** Accelerating impact as x increases
- **Use case:** Compounding negative effects
- **Formula:** `exp(x) - 1` or `1 - exp(-x)`

### Sigmoid Curves (Temperature, Distribution)
- **Shape:** S-curve with inflection point
- **Use case:** Three-phase behavior (low/medium/high)
- **Formula:** `1 / (1 + exp(-x))`

### Power Curves (Price, Consumer Confidence)
- **Shape:** Accelerating (exponent > 1) or decelerating (exponent < 1)
- **Use case:** Non-linear elasticity, psychological thresholds
- **Formula:** `x^n` where n ≠ 1

### Square Root (Digital Spend)
- **Shape:** Moderate saturation (between linear and log)
- **Use case:** Efficient channels with eventual saturation
- **Formula:** `sqrt(x)` or `x^0.5`

### Near-Linear (Trade Spend)
- **Shape:** Almost proportional with slight curve
- **Use case:** Direct, predictable relationships
- **Formula:** `x^0.9` (close to `x^1.0`)

---

## Overall Impact Scaling

All impacts are calibrated to achieve:
- **70% Baseline:** Inherent SKU potential (segment, pack, region, seasonality, trend)
- **30% Drivers:** Combined effect of all 12 time-varying drivers

Typical driver contribution magnitudes:
- Strongest: Discount % (450), Digital Spend (400), Temperature (400)
- Moderate: Trade Spend (380), TV Spend (350), Unemployment (350)
- Supportive: Distribution (300), GDP (250), Consumer Confidence (200), Precipitation (200)

These ensure realistic sales decomposition where drivers meaningfully contribute to the 30% target without any single driver dominating.

---

## Validation Checks

To verify these relationships, check:
1. **Correlation signs** match expected direction (positive/negative)
2. **Non-linearity** is evident in scatter plots (not straight lines)
3. **Contribution magnitudes** sum to ~30% of baseline
4. **No single driver** dominates (balanced importance)
5. **Realistic ranges** for all features and impacts

---

## Business Implications

This non-linear design enables the model to capture:
- **Threshold effects** (e.g., discounts only work above certain levels)
- **Saturation** (e.g., more TV spend eventually stops helping)
- **Accelerating impacts** (e.g., high unemployment compounds rapidly)
- **Optimal ranges** (e.g., temperature sweet spot for beverages)

These patterns match real-world FMCG dynamics, making the model more realistic and interpretable for business stakeholders.
