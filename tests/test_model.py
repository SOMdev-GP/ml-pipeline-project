from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# Dummy dataset
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")