import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# synthetic data
np.random.seed(42)
num_samples = 1000
feature_names = [
    'area', 'bedrooms', 'bathrooms', 'stories',
    'parking', 'mainroad', 'guestroom'
]

X = pd.DataFrame({
    'area': np.random.randint(500, 3500, num_samples),
    'bedrooms': np.random.randint(1, 5, num_samples),
    'bathrooms': np.random.randint(1, 4, num_samples),
    'stories': np.random.randint(1, 4, num_samples),
    'parking': np.random.randint(0, 4, num_samples),
    'mainroad': np.random.randint(0, 2, num_samples), # binary, 0/1
    'guestroom': np.random.randint(0, 2, num_samples),
})

# Generate target with some relationship
coef = np.array([150, 10000, 12000, 8000, 5000, 10000, 5000])
y = (
    X['area']*150 + X['bedrooms']*10000 + X['bathrooms']*12000 +
    X['stories']*8000 + X['parking']*5000 +
    X['mainroad']*10000 + X['guestroom']*5000 +
    np.random.normal(0, 20000, num_samples)
)

# Save data for visualization
os.makedirs('data', exist_ok=True)
X.assign(price=y).to_csv('data/synthetic_houses.csv', index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

#  Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:.2f}")
print(f"Test R^2: {r2:.2f}")

#  Save scaler, model, feature_names
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("files saved")