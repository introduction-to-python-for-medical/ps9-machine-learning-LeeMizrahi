import pandas as pd
df = pd.read_csv('parkinsons.csv')

import seaborn as sns
sns.pairplot(df, hue='status')

features = ['PPE', 'DFA']
target = 'status'
X = df[features]
y = df['status']

import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(max_depth= 3)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print(accuracy)





  
