# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:05:02 2024

@author: Ryan
"""

# Import packages
import tkinter as tk
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import missingno as msno
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from math import sqrt
from sklearn.metrics import r2_score


# Import packages
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import missingno as msno
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, ward, fcluster


# %% Load the data and perform train/test split
# Load data
file_path = r"C:\Users\Ryan\Documents\AlgorithmsI\FinalProject\train_fp.csv"
df1 = pd.read_csv(file_path)

# Perform the initial train/test split
train_df, test_df = train_test_split(df1, test_size=0.2, random_state=23)

# %% Data Exploration - train_df
df2 = df1

# Visualizing the null values in all columns
plt.figure(figsize=(30, 8))
sns.heatmap(df2.isnull(), cmap='flare')

# Visualize dependent variable 'SalePrice'
sns.histplot(df2['SalePrice'], kde=True)

# Dependent variable log transformation
df2['SalePrice'] = np.log1p(df2['SalePrice'])

# Result
sns.histplot(df2['SalePrice'], kde=True)

# Summary Statistics of all numeric variables
df2[df2.select_dtypes(exclude='object').columns].describe()

# Correlation Map
# Select only numeric variables
numeric_variables = df2.select_dtypes(include=np.number)

# Calculate the correlation matrix for numeric variables
corr = numeric_variables.corr()

# Select correlation values that are greater than 0.5
highly_corr_features = corr.index[abs(corr["SalePrice"]) > 0.5]

# Graph
plt.figure(figsize=(10, 10))
map = sns.heatmap(df2[highly_corr_features].corr(), annot=True, cmap="RdYlBu")

# All correlation values with 'SalePrice' in a descending order
corr_w_dependent = corr["SalePrice"]
corr_w_dependent.sort_values(ascending=False)

# Distribution of all numeric variables
for feature in highly_corr_features:
    plt.figure(figsize=(6, 4))
    plt.hist(df1[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# Create two new numeric variables based on 'HouseAge' & 'YearRemodAdd'
# Create house age variable (age of a house, not the year itself)
df2['HouseAge'] = df2['YrSold'] - df2['YearBuilt']

# Create remodled age variable
df2['RemodAdd_Age'] = df2['YrSold'] - df2['YearRemodAdd']

# Plot histograms for HouseAge and RemodAdd_Age
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df2['HouseAge'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of HouseAge')
plt.xlabel('HouseAge')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df2['RemodAdd_Age'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of RemodAdd_Age')
plt.xlabel('RemodAdd_Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Select categorical variables

categorical_variables = df2.select_dtypes(exclude=np.number)

# Show input percentage of each categorical variable

for column in categorical_variables.columns:
    plt.figure(figsize=(8, 6))
    df2[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of {column}')
    plt.ylabel('')
    plt.show()


# Check the NAs in all categorical variables
# Same process as what's done on numeric variables

total = categorical_variables.isnull().sum().sort_values(ascending=False)
percent = (categorical_variables.isnull().sum(
) / categorical_variables.isnull().count() * 100).sort_values(ascending=False)
categorical_missing_data = pd.concat(
    [total, percent], axis=1, keys=['Total', 'Percent'])
print(categorical_missing_data.head(10))


# Drop columns that have more than 5% of missing values
# Filter variables with more than 5% missing values
variables_to_drop = categorical_missing_data[categorical_missing_data['Percent'] > 5].index

# Drop the selected variables
categorical_cleaned = categorical_variables.drop(variables_to_drop, axis=1)

# show new categorical variables table
categorical_cleaned


# Fill categorical NAs with mode
# Find the mode of each column
mode_values = categorical_cleaned.mode().iloc[0]

# Fill missing values with the mode
categorical_cleaned = categorical_cleaned.fillna(mode_values)


# %% Data processing

# Function to process each dataset (train and test)
# %%Data Cleaning
def process_data(df):

    # Drop the 'Id' column if it exists
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Log transformation of the target variable if 'SalePrice' exists in df
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])

    # Create new numeric variables 'HouseAge' and 'RemodAdd_Age' if the necessary columns exist
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAdd_Age'] = df['YrSold'] - df['YearRemodAdd']

    # Drop columns with more than 5% missing values
    percent = (df.isnull().sum() / df.count() * 100)
    columns_to_drop = percent[percent > 5].index
    df.drop(columns=columns_to_drop, inplace=True)

    # Impute numeric variables with median
    numeric_vars = df.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    df[numeric_vars.columns] = imputer.fit_transform(numeric_vars)

    # Handle categorical variables: fill NAs with mode and encode
    categorical_vars = df.select_dtypes(exclude=[np.number])
    if not categorical_vars.empty:
        mode_values = categorical_vars.mode().iloc[0]
        categorical_vars = categorical_vars.fillna(mode_values)
        # Assuming 'categorical_vars' is a DataFrame of categorical variables
        dummy_vars = pd.get_dummies(categorical_vars, drop_first=True)
        # Explicitly convert to integers if necessary
        dummy_vars = dummy_vars.astype(int)
        # Concatenate with numeric variables
        df = pd.concat([df[numeric_vars.columns], dummy_vars], axis=1)

    return df


# Process both datasets
processed_train_df = process_data(train_df)
processed_test_df = process_data(test_df)

# Align the feature set in x_train and x_test to ensure they are identical
x_train, x_test = processed_train_df.align(
    processed_test_df, join='outer', axis=1, fill_value=0)  # fills missing dummies with 0
y_train = x_train.pop('SalePrice')
y_test = x_test.pop('SalePrice')

# Outputs for verification
print("Processed Train Features:\n", x_train.head())
print("Processed Train Target:\n", y_train.head())
print("Processed Test Features:\n", x_test.head())
print("Processed Test Target:\n", y_test.head())

y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)


# %% Random Forest

# Calculate baseline predictions (historical average of target variable)
baseline_preds = np.mean(y_test)

# Baseline errors (absolute difference between true target variable and baseline predictions)
baseline_errors = abs(y_test - baseline_preds)

# Display average baseline error
print('Mean Absolute Error (Baseline):', round(np.mean(baseline_errors), 2))


# Instantiate model with 200 decision trees
default_RF = RandomForestRegressor(n_estimators=200, random_state=23)
# Train the model on training data on the baseline parameters
default_RF.fit(x_train, y_train)
# Making predictions
default_RF_pred = default_RF.predict(x_test)
# Applying reverse of log, i.e exp, to see actual house value
default_RF_pred_actual = np.expm1(default_RF_pred)
# R^2 value
# perform the trained model on testing set
print(default_RF.score(x_test, y_test))

# Gridsearch
# Find best hyper-parameter value
RF_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": [6, 8, 10]
}

RF_grid_search = GridSearchCV(default_RF, RF_param_grid,
                              cv=5,
                              scoring="neg_mean_squared_error",
                              return_train_score=True)

RF_grid_search.fit(x_train, y_train)
RF_grid_search.best_estimator_
# Returns n_estimators = 200 & max_features = 10
# n_estimators = 'number of decision trees to be used in the model'

# Improve the RF model based on the last step
# Instantiate model with 300 decision trees
improved_forest = RandomForestRegressor(n_estimators=300, random_state=23)
# Train the model on training data
improved_forest.fit(x_train, y_train)
# Making predictions
improved_forest_pred = improved_forest.predict(x_test)
# Applying reverse of log, i.e exp, to see actual house value
improved_forest_pred = np.expm1(improved_forest_pred)

# Variable Importance
# Extract feature importances
RF_feature_importances = improved_forest.feature_importances_
# Create a DataFrame to store feature importances along with their corresponding feature names
RF_feature_importance_df = pd.DataFrame(
    {'Feature': x_train.columns, 'Importance': RF_feature_importances})
# Sort the DataFrame by importance in descending order
RF_feature_importance_df = RF_feature_importance_df.sort_values(
    by='Importance', ascending=False)
# Display the top features
print(RF_feature_importance_df.head(10))


# # Calculate the absolute errors
# errors = abs(prediction - y_test)

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)

# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# print('Accuracy:', round(accuracy, 2), '%.')


# # Calculate the absolute errors
# errors = abs(imp_prediction - y_test)

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)

# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# print('Accuracy:', round(accuracy, 2), '%.')


# %% Clustering Analysis

# Normalize and transform data
nrm = Normalizer()
normal_data = nrm.fit_transform(x_train)

# Apply t-SNE
tsn = TSNE(random_state=23)
res_tsne = tsn.fit_transform(normal_data)

# Plotting the t-SNE results
plt.figure(figsize=(8, 8))
# Corrected to use keyword arguments
sns.scatterplot(x=res_tsne[:, 0], y=res_tsne[:, 1])

# Hierarchical clustering on the t-SNE results
link = ward(res_tsne)
cluster_model = fcluster(link, t=300, criterion='distance')

# Plotting the results with cluster identification
fig = plt.figure(figsize=(25, 25))
ax1 = fig.add_subplot(3, 3, 1)
pd.value_counts(cluster_model).plot(kind='barh')

ax2 = fig.add_subplot(3, 3, 2)
sns.scatterplot(x=res_tsne[:, 0], y=res_tsne[:, 1], hue=cluster_model,
                palette="Set1", ax=ax2)  # Corrected with keyword args
ax2.legend_.remove()  # Correct usage to remove legend if unnecessary
# plt.savefig('Clustering.png', dpi=300)
plt.show()  # Ensure that the plots are displayed


sns.set(style='white')
plt.figure(figsize=(10, 7))
dendrogram(link)
ax = plt.gca()
bounds = ax.get_xbound()
# ax.plot(bounds, [300,300],'--', c='k')
ax.plot(bounds, '--', c='k')
plt.savefig('Dendrogram.png', dpi=300)
plt.show()







# %%XGBOOST
# %%
# #XGBoost Model for Other Data
# XGB_model = xgb.XGBRegressor(objective='reg:squarederror')

# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': range(0,10),
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'subsample': [1],
#     'colsample_bytree': [1]
# }

# # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
# # max_depth: The maximum depth of each tree, often values are between 1 and 10.
# # eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
# # subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
# # colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.

# grid_search_xgb = GridSearchCV(XGB_model, param_grid, cv=5)

# grid_search_xgb.fit(x_train, y_train)

# # Get the best parameters and the best score
# print("Best parameters:", grid_search_xgb.best_params_)
# print("Best cross-validation score:", grid_search_xgb.best_score_)

# # Evaluate the best model on the testing set
# print("Test set score:", grid_search_xgb.score(x_test, y_test))

# %%
# XGBoost Model with Best Parameters (BEST XGBoost Model used with this dataset)

# initializing models (using default parameters initially)
XGB_model_best = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1,
                                  learning_rate=0.1, max_depth=3, n_estimators=400, subsample=1, random_state=23)

# fitting the model
XGB_model_best.fit(x_train, y_train)

# make predictions
XGB_preds = XGB_model_best.predict(x_test)

# Applying reverse of log, i.e exp, to see actual house value
XGB_preds = np.expm1(XGB_preds)

# Calculate the absolute errors
XGB_errors = abs(XGB_preds - y_test_actual)

# Calculate mean absolute percentage error (MAPE)
XGB_mape = 100 * (XGB_errors / y_test_actual)

# Calculate and display accuracy
XGB_accuracy = 100 - np.mean(XGB_mape)

print('Accuracy:', round(XGB_accuracy, 2), '%.')

# accuracy measures (RMSE, R^2)
XGB_rmse = mean_squared_error(y_test_actual, XGB_preds, squared=False)
XGB_r2 = r2_score(y_test_actual, XGB_preds)
print("RMSE:", XGB_rmse)
print("R^2:", XGB_r2)

# Variable importance

# Extract feature importances
XGB_feature_importances = XGB_model_best.feature_importances_

# Create a DataFrame to store feature importances along with their corresponding feature names
XGB_feature_importance_df = pd.DataFrame(
    {'Feature': x_train.columns, 'Importance': XGB_feature_importances})

# Sort the DataFrame by importance in descending order
_XGB = XGB_feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(XGB_feature_importance_df.head(10))

# Accuracy: 99.26 %.
# RMSE: 0.12920813719468224
# R^2: 0.8918551943481917
#            Feature  Importance
# 0         1stFlrSF    0.006203
# 1         2ndFlrSF    0.006675
# 2        3SsnPorch    0.000566
# 3     BedroomAbvGr    0.000610
# 4  BldgType_2fmCon    0.000000
# 5  BldgType_Duplex    0.000541
# 6   BldgType_Twnhs    0.000000
# 7  BldgType_TwnhsE    0.001350
# 8      BsmtCond_Gd    0.003412
# 9      BsmtCond_Po    0.000576

#%%
# preds_XGB_real = np.expm1(XGB_preds)
# y_test_real = np.expm1(y_test)

# errors_XGB_real = abs(preds_XGB_real - y_test_real)

# # Calculate mean absolute percentage error (MAPE)
# mape_XGB_real = 100 * (errors_XGB_real / y_test_real)

# # Calculate and display accuracy
# accuracy_XGB_real = 100 - np.mean(mape_XGB_real)

# print('Accuracy:', round(accuracy_XGB_real, 2), '%.')

# # accuracy measures (RMSE, R^2)
# rmse_XGB_real = mean_squared_error(y_test_real, preds_XGB_real, squared=False)
# r2_XGB_real = r2_score(y_test_real, preds_XGB_real)
# print("RMSE:", rmse_XGB_real)
# print("R^2:", r2_XGB_real)


# Accuracy: 90.86 %.
# RMSE: 20904.85680607727
# R^2: 0.913325003221211
# %%
plt.scatter(y_test_actual, XGB_preds)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted values (XGBoost Regression)")
plt.show()

#%%
#visualizing tree


# Plot the first tree
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(XGB_model_best, num_trees=0, ax=ax)

# Plot the second tree
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(XGB_model_best, num_trees=1, ax=ax)

# Plot the third tree sideways
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(XGB_model_best, rankdir="LR", num_trees=2, ax=ax)

# Plot the 300th tree sideways
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(XGB_model_best, rankdir="LR", num_trees=299, ax=ax)

# Plot the last tree sideways
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(XGB_model_best, rankdir="LR", num_trees=399, ax=ax)

# %%
xgb.plot_importance(XGB_model_best, spacing=10)
plt.tight_layout()


# %%ACCURACY
# %% Define Adjusted R2 function
def adjusted_rsquared(r_squared, n, p):
    return 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))

# %% Performace calcualtions: mse, rmse, r2, and adjusted r2


def model_performance(models, x_train, y_train, x_test, y_test):
    performance = []

    # Undo log transformation on the test and training data
    y_train_actual = np.expm1(y_train)
    y_train_actual = y_train_actual.values
    y_train_actual = y_train_actual.flatten()

    y_test_actual = np.expm1(y_test)
    y_test_actual = y_test_actual.values
    y_test_actual = y_test_actual.flatten()

    for name, model in models.items():

        train_preds = model.predict(x_train)
        test_preds = model.predict(x_test)

        # Undo the log transformations of the predictions
        train_preds = np.expm1(train_preds)
        train_preds = train_preds.flatten()

        test_preds = np.expm1(test_preds)
        test_preds = test_preds.flatten()

        # Calculate the mse, rmse, r2, and adjr2 for the model against both the train and test set
        train_mse = mean_squared_error(y_train_actual, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train_actual, train_preds)
        train_adjr2 = adjusted_rsquared(
            train_r2, len(x_train), x_train.shape[1])

        test_mse = mean_squared_error(y_test_actual, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test_actual, test_preds)
        test_adjr2 = adjusted_rsquared(test_r2, len(x_test), x_test.shape[1])

        # Calculate the percentage within certain thresholds: 1%, 10%, 20%
        within_1_percent = np.sum(np.abs(
            test_preds - y_test_actual) / y_test_actual * 100 <= 1) / len(y_test_actual) * 100
        within_10_percent = np.sum(np.abs(
            test_preds - y_test_actual) / y_test_actual * 100 <= 10) / len(y_test_actual) * 100
        within_20_percent = np.sum(np.abs(
            test_preds - y_test_actual) / y_test_actual * 100 <= 20) / len(y_test_actual) * 100

        # Append the metrics into a list
        performance.append({
            "Model": name,
            "Train MSE": train_mse,
            "Train RMSE": train_rmse,
            "Train R2": train_r2,
            "Train AdjR2": train_adjr2,
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test R2": test_r2,
            "Test AdjR2": test_adjr2,
            "within 1%": within_1_percent,
            "within 10%": within_10_percent,
            "within 20%": within_20_percent})

        # Make the list into a dataframe
        performance_df = pd.DataFrame(performance)
    return performance_df

#%%
models = {"Random Forest": default_RF,
          "Improved Forest": improved_forest,
          "XGB":XGB_model_best}


performance_df = model_performance(models, x_train, y_train, x_test, y_test)

#%% #7 - Implementation and show off
"""
Agenda:
    - Final model should be loadable and able to predict on new examples
    - Take a random sample of the raw dataset
    - export it to a dictionary of values:
        - dictionary: raw.iloc[1].to_dict("records")
		- save it as pickle or json (with json.dumps) in a file
    - Prepare code to load dictionary into the dataframe
    In class:
        - Load that file/dictionary 
        - Colleagues will indicate what modifications to make to the dictionary 
        - You need to have code written (A.iii) to take that dictionary and create a dataframe
	    - Then pass the dataframe through cleaning/prep, then estimate your model on it (this single sample)
	    - Print the result
"""

# Create a random sample
random_record = df1.sample(n=1)

# Convert to dictionary
record_dict = random_record.to_dict(orient='records')[
    0]  # Convert to dictionary

# Save the dictionary with json
with open('record.json', 'w') as file:
    json.dump(record_dict, file)
    
#%% Update loaded dictionary


def update():
    def update_record():
        for key in entry_widgets:
            loaded_dict[key] = entry_widgets[key].get()
        root.destroy()

    root = tk.Tk()
    root.title("Modify Record")

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    entry_widgets = {}
    for key, value in loaded_dict.items():
        label = tk.Label(scrollable_frame, text=key)
        entry = tk.Entry(scrollable_frame)
        entry.insert(0, str(value))
        label.pack()
        entry.pack()
        entry_widgets[key] = entry

    submit_btn = tk.Button(
        scrollable_frame, text="Update Record", command=update_record)
    submit_btn.pack()

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()



# %% Implimentation during the final
# Load the dictionary
with open('record.json', 'r') as file:
    loaded_dict = json.load(file)

# Modify the dictionary
update()

# Convert dictionary to dataframe
modified_df = pd.DataFrame([loaded_dict])

# Store the actual SalePrice for later comparison
y_random = modified_df['SalePrice']

# Assuming you have a process_data function that prepares your DataFrame
modified_df = process_data(modified_df)

# Drop the 'SalePrice' column as it should not be included in the feature set for prediction
modified_df = modified_df.drop(columns=['SalePrice'])

# Align the features of processed_modified_df with those used by the model (x_train columns)
modified_df = modified_df.reindex(columns=x_train.columns, fill_value=0)

# Predict the SalePrice using the trained model (assuming the model is named 'forest' and is loaded)
predicted_log_price = forest.predict(modified_df)

# Since the original SalePrice was log-transformed, apply the exponential function to get the actual price
predicted_price = np.expm1(predicted_log_price)

# Print the predicted and actual SalePrice
print("Predicted SalePrice:", predicted_price[0])
print("Actual SalePrice:", y_random.iloc[0])