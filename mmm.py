import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

"""# Data Loading & Exploration"""

df = pd.read_csv('Sample Media Spend Data.csv')
df.head()

df.info()

df.describe()

df.isnull().sum()

clean_cols = ['Paid_Youtube', 'Organic_Youtube', 'Google', 'Email', 'Facebook', 'Affiliate', 'Overall_Views', 'Sales']

df[clean_cols].hist(
    bins=50,
    figsize=(12, 5)
)
plt.show()

"""Right skewed we may have to use log transformation after transforming for adstock and saturation effects

# Data Cleaning & Feature Engineering
"""

df = df.rename(columns={
    'Paid_Views': 'Paid_Youtube',
    'Organic_Views': 'Organic_Youtube',
    'Google_Impressions': 'Google',
    'Email_Impressions': 'Email',
    'Facebook_Impressions': 'Facebook',
    'Affiliate_Impressions': 'Affiliate'
})
print(df.columns)

df['Calendar_Week'] = pd.to_datetime(df['Calendar_Week'])

df['Year'] = df['Calendar_Week'].dt.year
df['Month'] = df['Calendar_Week'].dt.month
df['Day'] = df['Calendar_Week'].dt.day
df['Week'] = df['Calendar_Week'].dt.isocalendar().week

df = df.drop(columns=['Calendar_Week'])

df = pd.get_dummies(df, columns=['Division','Week'], drop_first=True)

df.info()

corr_matrix = df.corr()
plt.figure(figsize=(50,50))
sns.heatmap(corr_matrix, annot=True)
plt.show()

"""Paid Views and Organic Views are highly correlated, both are part of the Youtube channel

Overall Views is highly correlated with Paid Views and Organic Views
"""

df = df.drop(['Overall_Views', 'Month','Year','Day'], axis=1)

"""Adstock + Saturation Transformation"""

#Calculate adstock and saturation transformation
def adstock_saturation_transform(spent, beta, alpha):
    transform = np.zeros_like(spent, dtype=float)
    for t in range(len(spent)):
        if t == 0:
            transform[t] = spent[t] ** alpha
        else:
            transform[t] = (transform[t-1] * beta + spent[t]) ** alpha
    return transform

media_cols = [
    'Paid_Youtube', 'Organic_Youtube','Google',
    'Email', 'Facebook', 'Affiliate'
]

#Parameter grids for adstock (beta) and saturation (alpha)
beta_grid = np.arange(0.1, 1.0, 0.1)
alpha_grid = np.arange(0.1, 1.0, 0.1)

X = df.drop(columns=['Sales'])
y = df['Sales']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)

#Set up cross validation for selecting optimal adstock and saturation parameters for training data
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_params_per_channel = {}

#Loop through each channel
for col in media_cols:
    best_r2 = 0
    best_combo = None

    #Loop through all possible beta and alpha values in the grid
    for beta in beta_grid:
        for alpha in alpha_grid:
            X_train_transformed = X_train.copy()
            X_train_transformed[col] = adstock_saturation_transform(
                X_train[col].values, beta, alpha
            )

            r2_scores = []
            #Perform cross validation; split data into training and validation sets for the current fold
            for train_idx, val_idx in kf.split(X_train_transformed):
                X_cv_train = X_train_transformed.iloc[train_idx]
                X_cv_val = X_train_transformed.iloc[val_idx]
                y_cv_train = y_train.iloc[train_idx]
                y_cv_val = y_train.iloc[val_idx]

                #Run a linear regression model on the training fold
                model = LinearRegression()
                model.fit(X_cv_train, y_cv_train)
                y_pred = model.predict(X_cv_val)
                #Find r^2 score for this fold and store it
                r2_scores.append(r2_score(y_cv_val, y_pred))

            #Calculate the average r^2 across all folds
            avg_r2 = np.mean(r2_scores)

            #Find the best r^2 for the current channel
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_combo = {'beta': beta, 'alpha': alpha}

    #Store the best beta and alpha for the current channel
    best_params_per_channel[col] = best_combo


print("Best Beta and Alpha per Channel:")
print(best_params_per_channel)

X_train_final = X_train.copy()
X_test_final = X_test.copy()

#Transform the training and testing data with the best beta and alpha paramaters for each channel
for col in media_cols:
    beta_best = best_params_per_channel[col]['beta']
    alpha_best = best_params_per_channel[col]['alpha']
    X_train_final[col] = adstock_saturation_transform(X_train[col].values, beta_best, alpha_best)
    X_test_final[col] = adstock_saturation_transform(X_test[col].values, beta_best, alpha_best)

"""Check if adstock and saturation transformation helped with making it less skewed


"""

clean_cols = ['Paid_Youtube', 'Organic_Youtube', 'Google', 'Email', 'Facebook', 'Affiliate']

X_train_final[clean_cols].hist(
    bins=50,
    figsize=(12, 5)
)
plt.show()

"""It didn't, so we will have to log transform"""

X_train_final[media_cols] = np.log1p(X_train_final[media_cols])
X_test_final[media_cols]  = np.log1p(X_test_final[media_cols])

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

clean_cols = ['Paid_Youtube', 'Organic_Youtube', 'Google', 'Email', 'Facebook', 'Affiliate']

X_train_final[clean_cols].hist(
    bins=50,
    figsize=(12, 5)
)
plt.show()

"""# Linear Regression Model"""

final_model = LinearRegression()
final_model.fit(X_train_final, y_train)
y_pred_train = final_model.predict(X_train_final)
test_r2 = r2_score(y_train, y_pred_train)
print("Train R²:", round(test_r2, 4))

y_pred_test = final_model.predict(X_test_final)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Final Model Performance:")
print("Test R²:", round(test_r2, 4))
print("RMSE:", round(rmse, 4))

"""# Ridge Regression Model"""

scaler = StandardScaler()

X_train_final_scaled = scaler.fit_transform(X_train_final)
X_test_final_scaled = scaler.transform(X_test_final)

ridge_alpha_grid = np.logspace(-3, 3, 10)
ridge = Ridge()
param_grid = {'alpha': ridge_alpha_grid}
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf, scoring='r2')
ridge_cv.fit(X_train_final_scaled, y_train)

print("Best Ridge alpha:", ridge_cv.best_params_)
final_ridge_model = ridge_cv.best_estimator_


y_test_pred = final_ridge_model.predict(X_test_final_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Ridge Test R²:", test_r2)
print("RMSE:", rmse)

print("Coefficients:", final_model.coef_)
print("Intercept:", final_model.intercept_)

print("Coefficients:", final_ridge_model.coef_)
print("Intercept:", final_ridge_model.intercept_)

"""Both models have about the same r^2 and RMSE, signifying that multicollinearity does not have an impact on prediction performance.  However, the coefficients for the ridge model look to be more stable as expected, so we will use it over the linear model.

# Budget Allocation
"""

from scipy.optimize import minimize

total_budget = 1000000
last_period_spend = df.iloc[-1]

#Objective function for maximizing predicted sales
def objective(spends):
    pred_sales = final_ridge_model.intercept_
    for i, ch in enumerate(media_cols):
        beta = best_params_per_channel[ch]['beta']
        alpha = best_params_per_channel[ch]['alpha']
        prev_adstock = last_period_spend[ch]
        adstock = (prev_adstock * beta + spends[i]) ** alpha
        pred_sales += final_ridge_model.coef_[i] * np.log(adstock)
    return -np.exp(pred_sales)

#Budget Constraints
constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
bounds = [(0, total_budget) for _ in media_cols]
x0 = np.full(len(media_cols), total_budget / len(media_cols))

#Calculate optimal spending for each channel
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[constraint])

optimal_spends = result.x
predicted_sales = -result.fun

print("\n=== Optimal Budget Allocation ===")
for ch, val in zip(media_cols, optimal_spends):
    print(f"{ch:<25} ${val:,.0f}")
print(f"\nPredicted Sales: {predicted_sales:,.2f}")

"""
Insights & Recomendations:

Based on the model, the media spending priority should be Email, Facebook, Organic YouTube, and Google
Paid YouTube and Affiliate are not recommended since they show no predicted contributions

With a $1,000,000 budget, the model suggests the following budget allocation:
Email $502,826
Facebook $350,779
Organic Youtube $146,395

The predicted sales for this budget allocation are $4,246,416.23

"""