from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load dataset
df = pd.read_csv('parkinsons.csv')

# Define features and labels
input_features = ['DFA', 'PPE']
output_feature = 'status'

# Handle missing values
df = df.dropna()

# Split data into features and labels
X = df[input_features]
y = df[output_feature]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
score = accuracy_score(y_test, predictions)
print(f'Accuracy: {score}')
