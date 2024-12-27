import sys 
import subprocess 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
print("Note: This script assumes you have a 'house_data.csv' file in the same directory.")
print("If you don't have this file, please create a sample CSV file with the required columns.")
print("Required columns: 'price', 'location', 'size', 'age', 'bedrooms', 'bathrooms', 'garage'")

try:
    data = pd.read_csv('house_data.csv')
except FileNotFoundError:
    print("\nError: 'house_data.csv' file not found.")
    print("Please create a CSV file named 'house_data.csv' with the required columns and run the script again.")
    sys.exit(1)

# Preprocess the data
# Convert 'location' to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['location'])

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Function to make predictions
def predict_price(features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    return model.predict(features_scaled)[0]

# Get feature importances
importances = model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=90)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importances.png')

# Visualize predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual House Prices')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')

print("Feature importances and prediction visualizations have been saved.")

# Example prediction
example_house = np.array([2000, 10, 3, 2, 1] + [0] * (len(X.columns) - 5))  # Assuming 2000 sqft, 10 years old, 3 bed, 2 bath, 1 garage, and the first location
predicted_price = predict_price(example_house)
print(f"Predicted price for the example house: ${predicted_price:,.2f}")