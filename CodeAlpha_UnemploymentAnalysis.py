# ============================================
# CodeAlpha - Unemployment Analysis
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Upload Dataset
from google.colab import files
uploaded = files.upload()

# Read CSV File
df = pd.read_csv(next(iter(uploaded)))

# Show Dataset
print(df.head())

# Dataset Info
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Rename Columns (if needed)
df.columns = [col.strip() for col in df.columns]

# Convert Date Column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Basic Statistics
print(df.describe())

# Plot Unemployment Trend
plt.figure(figsize=(12,6))

if 'Date' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
    sns.lineplot(
        x='Date',
        y='Estimated Unemployment Rate (%)',
        data=df
    )

plt.title("Unemployment Rate Over Time")
plt.xticks(rotation=45)
plt.show()

# State-wise Analysis
if 'Region' in df.columns:
    plt.figure(figsize=(14,6))

    sns.barplot(
        x='Region',
        y='Estimated Unemployment Rate (%)',
        data=df
    )

    plt.xticks(rotation=90)
    plt.title("State-wise Unemployment Rate")
    plt.show()

# Covid Impact Analysis
if 'Date' in df.columns:
    covid_period = df[df['Date'] >= '2020-03-01']

    plt.figure(figsize=(12,6))

    sns.lineplot(
        x='Date',
        y='Estimated Unemployment Rate (%)',
        data=covid_period
    )

    plt.title("Covid-19 Impact on Unemployment")
    plt.xticks(rotation=45)
    plt.show()

print("\nAnalysis Completed Successfully!")
