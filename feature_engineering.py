import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

#Select categorical columns
cate_cols = df.select_dtypes(include=['object']).columns
print(cate_cols)
# Replace missing values with the most frequent category (mode)
for col in cate_cols:
    mode = df[col].mode()[0]  # Get the most common value
    df.fillna({col: mode}, inplace=True)  # Fill missing values

#Implementation for detecting outliers in selected numerical columns

features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# Plot box plots to detect outliers
df[features] = np.log(df[features])
df[features].boxplot(figsize=(8, 4))

plt.title('Box Plot for Outlier Detection')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

# Replacing numerical columns with median values
# Select numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns

for col in numerical_columns:
    median = df[col].median()
    df.fillna({col: median}, inplace=True)  # Replace nulls with median

#Creating new features for more useful insights

# Get columns that contain 'Yr' or 'Year'
year_columns = [feature for feature in numerical_columns if 'Yr' in feature or 'Year' in feature]

# Convert year values into age-related features
for col in year_columns:
    df[col] = df['YrSold'] - df[col]

#Code implementation to identify skewed numerical features

# Get numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Identify columns containing zeros
numerical_0s = df.loc[:, (df == 0).any()].select_dtypes(include=['number']).columns

# Remove columns that contain zeros from consideration
numerical_columns = numerical_columns.difference(numerical_0s)

# Calculate skewness for the remaining numerical columns
skewness = df[numerical_columns].skew()

# Set threshold for skewness (e.g., absolute value > 1 indicates high skewness)
skewed_columns = skewness[abs(skewness) > 1]

# Display skewed columns
print("Skewed Columns:")
print(skewed_columns)

#Using log-normal distribution to convert the skewed columns into a Gaussian distribution

# The list of highly skewed features identified earlier
skew_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

# Apply log transformation to each skewed feature
for col in skew_features:
    df[col] = np.log(df[col])

#Applying Target Encoding

# Select categorical variables
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply target encoding

for col in categorical_columns:
    # Compute mean SalePrice for each category
    labels_ordered = df.groupby([col])['SalePrice'].mean().sort_values().index

    # Assign numerical values based on target variable mean
    labels_ordered = {x: i for i, x in enumerate(labels_ordered, 0)}

    # Map encoded values back to the dataframe
    df[col] = df[col].map(labels_ordered)



#DATA IS NOW READY FOR MACHINE LEARNING!