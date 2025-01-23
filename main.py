%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv


  mport pandas as pd

df = pd.read_csv('/content/parkinsons.csv')
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Create pair plots to visualize relationships between features
sns.pairplot(df, vars=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                       'spread1', 'spread2', 'D2', 'PPE'], hue='status')
plt.show()

# Example feature selection based on observation (replace with your own analysis)
input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']  # Example: Two input features
output_feature = 'status'  # Example: Output feature (status)

from sklearn.preprocessing import MinMaxScaler

# Assuming 'input_features' and 'df' are defined as in the previous code
scaler = MinMaxScaler()
df[input_features] = scaler.fit_transform(df[input_features])
print(df.head())

from sklearn.model_selection import train_test_split

# Assuming 'input_features', 'output_feature', and 'df' are defined as in the previous code
X = df[input_features]
y = df[output_feature]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # Example: 80% train, 20% validation

from sklearn.linear_model import LogisticRegression

# Choose a model (Logistic Regression in this example)
model = LogisticRegression()

model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
  print("Accuracy is below the required threshold of 0.8. Try different features or models.")

import joblib

joblib.dump(model, 'my_model.joblib')

  
