import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_samples = 100

# Generate data
data = {
    'price': np.random.uniform(200000, 800000, n_samples),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples),
    'size': np.random.uniform(1000, 5000, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'garage': np.random.randint(0, 3, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Round numerical values
df['price'] = df['price'].round(2)
df['size'] = df['size'].round(0)

# Save to CSV
df.to_csv('house_data.csv', index=False)

print("Successfully created house_data.csv!")
print("\nFirst few rows of the dataset:")
print(df.head())