# Marketing Mix Modeling

This project implements a Marketing Mix Model (MMM) that estimate the contribution of media channels to sales while accounting for adstock and saturation effects. These estimates are then used for budget optimization and forecast sales. 

**Media Channel Variables:** Facebook, Google, Email, YouTube Paid, YouTube Organic, Affiliate 

**Other Variables:** Division, Week

**Target Variable**: Sales

## Workflow Overview

**Data Cleaning**
- Rename and drop unnecessary variables
- Transform categorical variables into dummy variables to account for seasonality and regional effects

**Parameter Tuning**
- Grid search over β (adstock) and α (saturation) values on the training data for each media channel
- Use K-fold cross-validation with a linear regression model to evaluate the R² for each β and α combination
- Select the β and α combination for each media channel with the highest R²

**Model Building**
- Apply the best β and α parameters to transform training and testing datasets
- Build and compare Linear Regression and Ridge Regression model on transformed data

**Budget Optimization**
- Define objective function and constraints for maximizing predicted sales 
- Use the SLSQP optimization method to calculate the optimal media budget allocation under those constraints

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
