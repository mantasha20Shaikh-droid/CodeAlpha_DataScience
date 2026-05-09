# CodeAlpha - Car Price Prediction
# STEP 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# STEP 2: Upload Dataset
from google.colab import files
uploaded = files.upload()

# STEP 3: Get Uploaded File Name
file_name = next(iter(uploaded))
print("Uploaded File:", file_name)

# STEP 4: Read Excel Dataset
df = pd.read_excel(file_name)

# STEP 5: Show Dataset
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# STEP 6: Dataset Information
print("\n===== DATASET INFO =====")
print(df.info())

# STEP 7: Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# STEP 8: Drop Missing Values
df.dropna(inplace=True)

# STEP 9: Encode Categorical Columns
label_encoder = LabelEncoder()
for col in df.columns:

    if df[col].dtype == 'object':

        print(f"Encoding Column: {col}")

        # Convert to string
        df[col] = df[col].astype(str)

        # Encode
        df[col] = label_encoder.fit_transform(df[col])

print("\nEncoding Completed Successfully!")

# STEP 10: Features and Target
target_column = 'Selling_Price'
X = df.drop(target_column, axis=1)
y = df[target_column]

# STEP 11: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(    
    X,
    y,
    test_size=0.2,
    random_state=42
)
# STEP 12: Model Training
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# STEP 13: Predictions
y_pred = model.predict(X_test)

# STEP 14: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Mean Absolute Error:", mae)
print(" R2 Score:", r2)
print("================================")

# STEP 15: Actual vs Predicted Graph
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# STEP 16: Feature Importance
importance = model.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})
feature_df = feature_df.sort_values(  
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))
sns.barplot(
    
    x='Importance',
    
    y='Feature',
    
    data=feature_df
)

plt.title("Feature Importance")
plt.show()
