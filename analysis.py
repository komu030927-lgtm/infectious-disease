import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. SETUP & DATA LOADING
# Ensure your file is named 'patient_data.csv' and is in the same folder
file_path = 'patient_data.csv'

if not os.path.exists(file_path):
    print(f"❌ Error: '{file_path}' not found in {os.getcwd()}")
    print("Please download the dataset and rename it to 'patient_data.csv'.")
    exit()

print("🚀 Loading dataset and starting analysis...")
df = pd.read_csv(file_path)

# 2. FEATURE ENGINEERING (Business Objectives 3 & 5)
# Convert date strings to datetime objects
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'])
df['released_date'] = pd.to_datetime(df['released_date'])

# Calculate Recovery Duration (Days from confirmation to release)
df['recovery_duration'] = (df['released_date'] - df['confirmed_date']).dt.days

# Calculate Age based on the current year (2026)
df['age'] = 2026 - df['birth_year']

# 3. EXPLORATORY DATA ANALYSIS (EDA) - Visual Evidence
print("📊 Generating Demographic Plots... (CLOSE THE GRAPH WINDOW TO CONTINUE)")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Gender Distribution (Objective 1)
sns.countplot(data=df, x='sex', hue='sex', palette='viridis', ax=axes[0], legend=False)
axes[0].set_title('Infection Distribution by Gender', fontsize=14)

# Plot 2: Age Distribution (Objective 1)
sns.histplot(df['age'].dropna(), bins=20, kde=True, color='teal', ax=axes[1])
axes[1].set_title('Patient Age Distribution', fontsize=14)

plt.tight_layout()
plt.show() # THE SCRIPT PAUSES HERE UNTIL YOU CLOSE THE WINDOW

# 4. REGIONAL & INFECTION ANALYSIS (Objective 2 & 4)
plt.figure(figsize=(10, 6))
top_regions = df['region'].value_counts().head(10)
sns.barplot(x=top_regions.values, y=top_regions.index, hue=top_regions.index, palette='magma', legend=False)
plt.title('Top 10 Most Impacted Regions', fontsize=14)
plt.xlabel('Number of Confirmed Cases')
plt.show() # CLOSE THIS WINDOW TO SEE FINAL STATS

# 5. LINEAR REGRESSION MODEL (Objective 5)
print("🤖 Training Linear Regression Model to predict recovery...")

# Prepare data: Age and Contact Number are features, Recovery Duration is target
model_data = df.dropna(subset=['recovery_duration', 'age', 'contact_number'])
model_data = model_data[model_data['recovery_duration'] >= 0] # Filter valid data

if not model_data.empty:
    X = model_data[['age', 'contact_number']]
    y = model_data['recovery_duration']

    # Split data: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and Fit Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)

    # --- FINAL PROJECT REPORT OUTPUT ---
    print("\n" + "="*40)
    print("   HEALTHGUARD ANALYTICS FINAL REPORT")
    print("="*40)
    print(f"🔹 Average Recovery Time:   {df['recovery_duration'].mean():.2f} days")
    print(f"🔹 Most Frequent Region:    {df['region'].mode()[0]}")
    print("-" * 40)
    print(f"📈 Model R-Squared Score:   {r2_score(y_test, y_pred):.4f}")
    print(f"📉 Mean Squared Error:      {mean_squared_error(y_test, y_pred):.2f}")
    print("="*40)
    print("✅ Project Analysis Complete.")
else:
    print("\n⚠️ Not enough valid data to run the Linear Regression model.")