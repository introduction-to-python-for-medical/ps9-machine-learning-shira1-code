import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

input_features = ['DFA', 'PPE']
output_feature = 'status'

df = df.dropna()
# Create input and output datasets
X = df[input_features]
y = df[output_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) 

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score
