# Marketing Mix Modeling

This project implemented a Marketing Mix Model (MMM) to estimate the contribution of media channels to sales while accounting for adstock and saturation effects. These estimates were then used for budget optimization and forecasting futures sales. 

**Media Channel Variables:** Facebook, Google, Email, YouTube Paid, YouTube Organic, Affiliate 

**Other Variables:** Division, Week

**Target Variable**: Sales

## What I did

**Exploration** - Explored the variables and their distributions

**Cleaning** - Renamed and dropped variables and handled outliers

**Feature Engineering** - Created new date features and encoded categorical variables 

**Parameter Tuning** - Selected the best adstock (β) and saturation (α) parameters for each media channel using grid search with cross-validation

**Model Building** - Built and compared Linear and Ridge Regression models on the transformed data

**Budget Optimization** - Determined the media budget allocation that maximizes predicted sales under defined objectives and constraints

## Insights & Recommendations  

### Media Spending Priority
1. Email
2. Facebook
3. Organic YouTube
4. Google

> Paid YouTube and Affiliate are not recommended since they show no predicted contributions

### Optimized Budget Allocation for $1,000,000
- **Email:** $502,826  
- **Facebook:** $350,779  
- **Organic YouTube:** $146,395
- **Googe:** $0
- **Paid Youtube:** $0
- **Affiliate:** $0 

**Predicted Sales:** $4,246,416
