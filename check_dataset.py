import pandas as pd

csv_path = r"C:\Users\rishi\Downloads\Biomass_Project\train.csv"

df = pd.read_csv(csv_path)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())