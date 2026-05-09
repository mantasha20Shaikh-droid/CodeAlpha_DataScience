# CodeAlpha - Unemployment Analysis
# STEP 1: Install KaggleHub

!pip install kagglehub -q

# STEP 2: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# STEP 3: Download Dataset

path = kagglehub.dataset_download("pantanjali/unemployment-dataset")
print("Dataset Folder Path:", path)

# STEP 4: Show Files Inside Dataset Folder

files = os.listdir(path)
print("\nFiles in Dataset Folder:")
print(files)

# STEP 5: Find CSV File Automatically

csv_file = None

for file in files:
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)

print("\nCSV File Path:")
print(csv_file)

# STEP 6: Read CSV File

df = pd.read_csv(csv_file)

# STEP 7: Show Dataset

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# STEP 8: Dataset Info

print("\n===== DATASET INFO =====")
print(df.info())

# STEP 9: Missing Values

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# STEP 10: Clean Column Names

df.columns = [col.strip() for col in df.columns]

print("\n===== COLUMN NAMES =====")
print(df.columns)

# STEP 11: Convert Date Column

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# STEP 12: Statistical Summary

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# STEP 13: Unemployment Trend

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

# STEP 14: State-wise Analysis

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

# STEP 15: Covid Impact Analysis

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

# STEP 16: Final Message

print(" Analysis Completed Successfully ")
