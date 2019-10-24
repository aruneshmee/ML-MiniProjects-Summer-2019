import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

passengers = pd.read_csv('train.csv')
passengers.head()

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})

# Fill the nan values in the age column
passengers['Age'].fillna(value= round(passengers['Age'].mean()) , inplace=True)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

scaler = StandardScaler()
features = scaler.fit_transform(features)

model = LogisticRegression()
model.fit(features, survival)

print(model.score(features, survival))

Jack = np.array([0.0, 20.0, 0.0, 0.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0])
You = np.array([0.0, 29, 1.0, 0.0])

# Combine passenger arrays
sample_passengers = np.array([ Jack , Rose, You ])
print(sample_passengers)

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
