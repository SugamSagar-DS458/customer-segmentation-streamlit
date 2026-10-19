# Customer Segment Profiles & Strategic Playbooks

## Overview & Clustering Summary

The customer base ($N = 500$) was segmented using unsupervised machine learning applied to standardized numerical attributes (`Tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) and one-hot-encoded categorical attributes (`Contract`, `PaymentMethod`, `PaperlessBilling`).

Optimal cluster count ($k = 3$) was verified via within-cluster sum of squares (Elbow Method) and silhouette analysis, corroborated by Agglomerative Hierarchical Clustering and 2D Principal Component Analysis (PCA).

| Metric | Overall Dataset | Segment 0 (Loyal / Contract-Bound) | Segment 1 (Core Volume / Month-to-Month) | Segment 2 (High-Risk / Price-Sensitive) |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster Size ($N$)** | 500 (100%) | 114 (22.8%) | 249 (49.8%) | 137 (27.4%) |
| **Agglomerative Distribution** | — | 107 (21.4%) | 250 (50.0%) | 143 (28.6%) |
| **Baseline Churn Rate** | 10.6% | ~5.8% | ~10.4% | ~14.6% |
| **Average Tenure** | 36.5 months | High (>40 mo) | Moderate (25–38 mo) | Short-to-Bimodal (<25 mo) |
| **Average Monthly Charges** | $113.64 | Low-to-Mid ($60–$95) | Moderate ($95–$125) | Elevated ($130–$185+) |
| **Dominant Contract Type** | Mixed | Two-Year / One-Year | Month-to-Month | Month-to-Month / Mixed |
| **Primary Payment Method** | Mixed | Bank Transfer / Credit Card | Electronic Check | Electronic Check |
| **Tuned Model Configuration** | — | $C=10$, Penalty = `l2` | $C=1$, Penalty = `l1` | $C=10$, Penalty = `l1` |
| **Tuned Model Accuracy** | — | **97.14%** | **97.33%** | **88.10%** |
| **Tuned Model Precision** | — | **1.0000** | **0.8750** | **0.6000** |
| **Tuned Model Recall** | — | **0.6667** | **0.8750** | **0.5000** |
| **Tuned Model F1-Score** | — | **0.8000** | **0.8750** | **0.5455** |

---

## Segment 0: Established Loyalty (The Anchors)

### 1. Persona & Characteristics
- **Population:** 114 accounts (22.8% of base)
- **Profile Summary:** Long-tenure, committed customers with high stability. They utilize formal, automated payment systems (Credit Card / Bank Transfer) and predominantly operate under multi-year contracts.
- **Spending Behavior:** Stable, moderate monthly fees with high accumulated lifetime value (`TotalCharges`).
- **Churn Propensity:** Very low baseline churn.

### 2. Tuned Model Feature Importances (Coefficients)

```
Tenure                          [ -4.023971 ] ========================================
Contract: Two-Year              [ -2.470214 ] ========================
SeniorCitizen                   [ +1.824895 ] ++++++++++++++++++
Contract: Month-to-Month        [ +1.672686 ] +++++++++++++++++
PaperlessBilling: Yes           [ -1.561911 ] ===============
PaymentMethod: Electronic Check [ -1.157604 ] ============
Contract: One-Year              [ -1.034681 ] ==========
TotalCharges                    [ -0.918294 ] =========
PaymentMethod: Bank Transfer    [ -0.550189 ] =====
PaperlessBilling: No            [ -0.270298 ] ===
```

### 3. Key Behavioral Dynamics
- **Protective Anchors:** `Tenure` ($-4.02$) and `Contract_Two year` ($-2.47$) provide the strongest friction against cancellation.
- **Vulnerability Triggers:** When accounts in this segment are placed on `Month-to-month` contracts ($+1.67$) or identify as `SeniorCitizen` ($+1.82$), churn risk escalates sharply.
- **Model Efficiency:** Achieving $100\%$ precision means every churn alert for this segment is a genuine risk, allowing high-cost, high-touch interventions without wasting capital.

### 4. Strategic Objectives & Action Plan
- **Primary Goal:** Protect lifetime value and preempt contract expiration drop-offs.
- **Intervention Playbook:**
  1. **Proactive Renewal Windows:** Initiate contract extension offers 60 to 90 days before 1-year and 2-year terms expire.
  2. **Loyalty Continuity Incentives:** Provide hardware upgrades or speed bumps tied to multi-year renewal commitments.
  3. **Payment Integrity:** Ensure credit card expiration reminders are delivered via multi-channel prompts to avoid involuntary billing-failure churn.

---

## Segment 1: High-Volume Core (The Subscription Mass)

### 1. Persona & Characteristics
- **Population:** 249 accounts (49.8% of base — largest cluster)
- **Profile Summary:** The operational backbone of the business. Characterized by average tenure, flexible commitment terms, and widespread adoption of electronic invoicing and payment systems.
- **Spending Behavior:** Centered around benchmark monthly fees ($110–$130/month).
- **Churn Propensity:** Moderate, highly sensitive to contract type and service satisfaction.

### 2. Tuned Model Feature Importances (Coefficients)

```
SeniorCitizen                   [ -4.545144 ] =============================================
Tenure                          [ -3.176005 ] ===============================
Contract: Month-to-Month        [ +1.862631 ] +++++++++++++++++++
PaperlessBilling: No            [ -0.825778 ] ========
MonthlyCharges                  [ +0.459340 ] ++++
Agglomerative_Cluster           [ -0.317794 ] ===
TotalCharges                    [ +0.291458 ] +++
Contract: One-Year              [ -0.212310 ] ==
PaymentMethod: Bank Transfer    [ -0.201933 ] ==
Contract: Two-Year              [  0.000000 ]
```

### 3. Key Behavioral Dynamics
- **Protective Anchors:** `SeniorCitizen` status ($-4.55$) strongly correlates with stability in this group, serving as a reliable retention anchor alongside accumulated `Tenure` ($-3.18$).
- **Vulnerability Triggers:** `Contract_Month-to-month` ($+1.86$) is the dominant driver of customer departures. Higher `MonthlyCharges` ($+0.46$) further amplifies month-to-month exit likelihood.
- **Model Efficiency:** Balanced precision ($87.50\%$) and recall ($87.50\%$) make this model the most consistent for automated, programmatic retention campaigns.

### 4. Strategic Objectives & Action Plan
- **Primary Goal:** Drive contractual lock-in and cultivate senior demographics.
- **Intervention Playbook:**
  1. **Contract Migration Bridge:** Incentivize the shift from Month-to-Month to 12-Month agreements by offering a 1-month discount or bill credit.
  2. **Dedicated Senior Engagement:** Create accessible customer support pathways and specialized senior customer service tiers to solidify the $-4.55$ retention anchor.
  3. **Billing Transparency:** Implement predictive billing summaries to prevent invoice disputes among electronic check payers.

---

## Segment 2: High-Velocity Risk (The Churn Vulnerable)

### 1. Persona & Characteristics
- **Population:** 137 accounts (27.4% of base)
- **Profile Summary:** Shorter-tenured accounts paying higher-than-average monthly fees. This group exhibits elevated sensitivity to service outages, cost spikes, and onboarding friction.
- **Spending Behavior:** Elevated monthly billing rates ($130–$185+), creating strong fee sensitivity.
- **Churn Propensity:** Highest attrition rate across the customer base.

### 2. Tuned Model Feature Importances (Coefficients)

```
Tenure                          [ -7.986573 ] ===============================================================================
SeniorCitizen                   [ +4.831598 ] ++++++++++++++++++++++++++++++++++++++++++++++++
Contract: Two-Year              [ -2.689983 ] ===========================
MonthlyCharges                  [ +2.069529 ] +++++++++++++++++++++
PaymentMethod: Bank Transfer    [ -1.234614 ] ============
PaymentMethod: Electronic Check [ -1.215642 ] ============
Contract: One-Year              [ -1.159334 ] ===========
PaperlessBilling: No            [ -0.328566 ] ===
TotalCharges                    [  0.000000 ]
Contract: Month-to-Month        [  0.000000 ]
```

### 3. Key Behavioral Dynamics
- **Protective Anchors:** `Tenure` ($-7.99$) is decisive: once customers survive the initial months, their risk drops exponentially. `Contract_Two year` ($-2.69$) provides substantial insulation.
- **Vulnerability Triggers:** Unlike Segment 1, `SeniorCitizen` status here is the primary risk multiplier ($+4.83$), compounded by intense sensitivity to `MonthlyCharges` ($+2.07$).
- **Model Efficiency:** F1-score of $54.55\%$ reflects high volatility, requiring broader safety-net interventions rather than narrow threshold-based targeting.

### 4. Strategic Objectives & Action Plan
- **Primary Goal:** Onboarding stabilization, price relief, and first-year milestone defense.
- **Intervention Playbook:**
  1. **The "First 180 Days" Program:** Deploy structured customer success check-ins at 30, 60, and 90 days post-onboarding to bridge the dangerous tenure gap.
  2. **Down-Tiering / Right-Sizing Offers:** When accounts trigger early churn indicators, offer plan re-benchmarking rather than forcing cancellation due to high monthly charges ($+2.07$).
  3. **Senior Account Audit:** Investigate service complexity issues driving the $+4.83$ churn coefficient for seniors in this segment (e.g., confusing bills, digital onboarding hurdles).

---

## Cross-Segment Feature Comparison

| Feature Name | Segment 0 | Segment 1 | Segment 2 | Business Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Tenure** | **-4.02** | **-3.18** | **-7.99** | Universal protective effect; highest leverage in stabilizing Segment 2. |
| **SeniorCitizen** | **+1.82** | **-4.55** | **+4.83** | Opposite polarity across segments; demographic targeting must be segment-aware. |
| **Contract_Two year** | **-2.47** | 0.00 | **-2.69** | Multi-year contracts provide strong protection in Segments 0 & 2. |
| **Contract_Month-to-month** | **+1.67** | **+1.86** | 0.00 | Key risk factor in Segments 0 & 1; targets for contract upgrade campaigns. |
| **MonthlyCharges** | Negligible | +0.46 | **+2.07** | Major exit catalyst in Segment 2; negligible in Segment 0. |
| **PaperlessBilling_Yes/No** | **-1.56** | -0.83 | -0.33 | Standard paper billing correlates with lower churn across all tiers. |

---

## Operational Roadmap for Retention Teams

```
[Day 1–30: Triage]
  ├── Identify all Segment 2 accounts with Tenure < 12 months & MonthlyCharges > $120
  └── Route at-risk accounts to specialized customer retention reps for plan right-sizing

[Day 31–60: Conversion Campaign]
  ├── Target Segment 1 Month-to-Month subscribers with 12-month lock-in incentives
  └── Audit onboarding flows for Senior Citizens in Segments 0 and 2

[Day 61–90: Automated Production Scoring]
  ├── Embed tuned Logistic Regression models into CRM workflows
  ├── Route churn risk scores (> 0.60 threshold) directly to account managers
  └── Monitor precision/recall drift monthly
```