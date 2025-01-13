import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import yaml

# Step 1: Load the dataset
file_path = "parkinson.csv"  # Replace with the correct path if necessary
data = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
data.info()

# Step 2: Select Features
features = ["PPE", "RPDE"]  # Inputs
output = "status"  # Output
data = data[features + [output]]

# Step 3: Scale the Data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Step 4: Split the Data
X = data[features]
y = data[output]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose a Model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)

# Step 6: Test the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

if accuracy >= 0.8:
    print("Model meets the accuracy requirement.")
else:
    print("Model accuracy is below 0.8. Consider tuning hyperparameters.")

# Step 7: Save the Model
model_filename = "my_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Save the configuration
config = {
    "selected_features": features,
    "path": model_filename
}

with open("config.yaml", "w") as file:
    yaml.dump(config, file)
    print("Configuration saved to config.yaml")

