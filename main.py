# prompt: After running the first cell of this notebook, the file parkinson.csv will appear in the Files folder. You need to loaded the file as a DataFrame

import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('/content/parkinsons.csv')

# Display the first few rows of the DataFrame (optional)
print(df.head())
# prompt: Choose two features as inputs for the model.
# Identify one feature to use as the output for the model.
# Advice:
# You can refer to the paper available in the GitHub repository for insights into the dataset and guidance on identifying key features for the input and output.
# Alternatively, consider creating pair plots or using other EDA methods we learned in the last lecture to explore the relationships between features and determine which ones are most relevant.

# Import necessary libraries (if not already imported)
import seaborn as sns
import matplotlib.pyplot as plt

# ... (Your existing code)

# Explore relationships between features using pair plots (example)
# You can customize the features to explore different relationships
sns.pairplot(df[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'status']], hue='status')
plt.show()


# Example feature selection based on observation from pairplot or domain knowledge (replace with your own choice):
# Input features
input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']  
# Output feature
output_feature = 'status'

# Now you have your input_features and output_feature defined.
# Proceed with the next steps in your model building process.

print("Selected Input Features:", input_features)
print("Selected Output Feature:", output_feature)
# prompt: Apply the MinMaxScaler to scale the two input columns to a range between 0 and 1.

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the input features and transform them
df[input_features] = scaler.fit_transform(df[input_features])

# Display the scaled DataFrame (optional)
print(df.head())
# prompt: Divide the dataset into a training set and a validation set.

# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X = df[input_features]
y = df[output_feature]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # You can adjust test_size and random_state

# Print the shapes of the resulting sets (optional)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
# prompt: Select a model to train on the data.
# Advice:
# Consider using the model discussed in the paper from the GitHub repository as a reference.

from sklearn.linear_model import LogisticRegression

# Initialize the model (you can choose other models as well)
model = LogisticRegression()
# prompt: Evaluate the model's accuracy on the test set. Ensure that the accuracy is at least 0.8.

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
  print("Accuracy is below the target threshold of 0.8. Consider trying different features, models or hyperparameters.")
