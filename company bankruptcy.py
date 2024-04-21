import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv("D:/Disha/Diploma/AI ML/Company Bankruptcy Prediction/data.csv")

missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# Split the dataset into features and target variable
X = df.drop('Bankrupt?', axis=1)
y = df['Bankrupt?']

# Oversampling using RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)


from sklearn.metrics import mean_squared_error

# Calculate the mean square error
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error:", mse)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)