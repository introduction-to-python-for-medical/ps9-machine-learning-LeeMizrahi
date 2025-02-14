import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml

# Load the dataset
df = pd.read_csv('parkinsons.csv')  # Assuming 'parkinsons.csv' is in the same directory

# Select features
input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']
output_feature = 'status'
X = df[input_features]
y = df[output_feature]

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=input_features)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Choose a model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Test the accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")
if accuracy < 0.8:
    print("Accuracy is below the required threshold of 0.8.")

# Save the model
joblib.dump(model, 'parkinsons_model.joblib')

# Create config.yaml
config_content = {
    'selected_features': input_features,
    'path': 'parkinsons_model.joblib'
}
with open('config.yaml', 'w') as f:
    yaml.dump(config_content, f)

print("Model saved and config.yaml created.")
